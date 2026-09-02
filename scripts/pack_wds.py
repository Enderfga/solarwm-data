#!/usr/bin/env python3
"""Pack a corpus into WebDataset shards plus each partition's `meta.jsonl`.

    <out>/<owner>/<partition>/shards/shard-000000.tar
    <out>/<owner>/<partition>/meta.jsonl

PARTITION BY TIER, because the physical release stores every canonical row under one of
`kept-high`, `kept-xhigh` or `rejected`. Rejected rows are part of the release, not
debris: they carry the same annotations and explicit rejection provenance as kept rows,
which is what makes a different threshold, a dropped metric or a new source mixture
answerable from the corpus instead of from another GPU run. Keeping them in their own
partition is what lets a reader take the kept tiers alone without reading past them.

`--partition none` writes the flat `<out>/<owner>/shards/` layout instead, for a corpus
that has not been judged yet.

One sample is the group of files sharing a key, written CONTIGUOUSLY and in sorted order
inside one shard::

    <key>.video.mp4        <key>.poses.npy      <key>.intrinsics.npy
    <key>.meta.json        <key>.prompt.txt
    <key>.audio.m4a        # iff the source had audio
    <key>.gt_depth.npz     # iff the source ships GT depth
    <key>.action.npy       # iff the source ships actions

    python3 scripts/pack_wds.py --corpus <root> --owner miradata --out <dir>

THE KEY IS NOT THE CLIP ID. A WebDataset reader splits a member name at its FIRST dot to
get the key, so a clip id containing a dot would merge two different clips into one sample
and no reader would report an error — it would simply train on a video paired with another
clip's poses. Keys are therefore sanitised, the mapping is recorded in each sample's
meta.json as ``wds_key``, and an injectivity check aborts the run if two ids ever collide.

A sample never spans two shards, grouping is deterministic (sorted ids, size-bounded), and
a shard is written to a temporary name and renamed only once complete — so an interrupted
run leaves finished shards and no half-written one, and re-running resumes rather than
duplicating.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Written in this order inside a sample. Sorted by name, which is what readers expect,
#: and stable so two packs of the same corpus are byte-comparable.
MEMBERS = (
    "action.npy", "audio.m4a", "gt_depth.npz",
    "intrinsics.npy", "meta.json", "poses.npy", "prompt.txt", "video.mp4",
)
REQUIRED = ("intrinsics.npy", "meta.json", "poses.npy", "prompt.txt", "video.mp4")

DEFAULT_SHARD_BYTES = 2 * 1024**3


def wds_key(clip_id: str) -> str:
    """A dot-free, reader-safe key. Injectivity is checked by the caller, not assumed."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", clip_id)


def clip_files(clip_dir: Path) -> dict[str, Path] | None:
    """The members present for this clip, or None when a required one is missing.

    A clip missing a payload is skipped rather than packed short: per-clip upload is
    several writes, so a worker that died mid-clip leaves meta and poses with no video,
    and a directory-level count calls that complete.
    """
    have = {name: clip_dir / name for name in MEMBERS if (clip_dir / name).is_file()}
    if any(r not in have for r in REQUIRED):
        return None
    return have


#: Tier value in meta.json -> release partition directory.
PARTITIONS = {"xhigh": "kept-xhigh", "high": "kept-high", None: "rejected"}


def partition_of(meta: dict) -> str:
    """The release partition this clip belongs in, from its own recorded verdict.

    A MISSING `kept_tier` key is not "rejected" — it means nothing ever judged this clip,
    and filing it under `rejected` would publish an unjudged clip as a rejected one, which
    reads as a quality verdict that was never made. A present-but-null `kept_tier` IS
    rejected: that is how the schema records a clip the policy turned down.
    """
    if "kept_tier" not in meta:
        raise KeyError("meta.json has no kept_tier: clip was never judged")
    tier = meta["kept_tier"]
    if tier not in PARTITIONS:
        raise ValueError(f"unknown kept_tier {tier!r}; expected one of "
                         f"{sorted(k for k in PARTITIONS if k)} or null")
    return PARTITIONS[tier]


def plan_shards(clips: list[tuple[str, dict[str, Path]]], shard_bytes: int
                ) -> list[list[tuple[str, dict[str, Path]]]]:
    """Group clips into shards without ever splitting one. Deterministic given the input.

    A single clip larger than the target still gets its own shard rather than being
    dropped: the bound is a target, not a limit, and silently losing the biggest clips
    would be a far worse failure than an oversized shard.
    """
    shards: list[list[tuple[str, dict[str, Path]]]] = []
    cur: list[tuple[str, dict[str, Path]]] = []
    cur_bytes = 0
    for cid, files in clips:
        size = sum(p.stat().st_size for p in files.values())
        if cur and cur_bytes + size > shard_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append((cid, files))
        cur_bytes += size
    if cur:
        shards.append(cur)
    return shards


def write_shard(path: Path, samples: list[tuple[str, dict[str, Path]]]) -> None:
    """Write one shard atomically: temp name, then rename."""
    tmp = path.with_suffix(".tar.partial")
    with tarfile.open(tmp, "w") as tf:
        for cid, files in samples:
            key = wds_key(cid)
            for name in MEMBERS:            # sorted order, contiguous per sample
                src = files.get(name)
                if src is not None:
                    tf.add(str(src), arcname=f"{key}.{name}")
    tmp.rename(path)


def read_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pack_partition(name: str, out_root: Path, clips: list[tuple[str, dict[str, Path]]],
                   metas: dict[str, dict], shard_bytes: int) -> tuple[int, int, int]:
    """Write one partition's shards and meta.jsonl. Returns (shards, written, skipped)."""
    shard_dir = out_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = plan_shards(clips, shard_bytes)
    written = skipped = 0
    with open(out_root / "meta.jsonl", "w", encoding="utf-8") as mf:
        for i, samples in enumerate(shards):
            shard_name = f"shard-{i:06d}.tar"
            target = shard_dir / shard_name
            if target.exists():
                skipped += 1                      # resume: already complete
            else:
                write_shard(target, samples)
                written += 1
            for cid, files in samples:
                rec = dict(metas[cid])
                rec["wds_key"] = wds_key(cid)
                rec["wds_partition"] = name
                rec["wds_shard"] = shard_name
                rec["wds_members"] = sorted(files)
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(shards), written, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="root holding <owner>/<clip_id>/ dirs")
    ap.add_argument("--owner", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    ap.add_argument("--partition", choices=("tier", "none"), default="tier",
                    help="tier: kept-xhigh / kept-high / rejected (the release layout); "
                         "none: one flat set of shards, for an unjudged corpus")
    a = ap.parse_args()

    src_root = Path(a.corpus) / a.owner
    if not src_root.is_dir():
        raise SystemExit(f"no such owner directory: {src_root}")
    out_root = Path(a.out) / a.owner

    clips: list[tuple[str, dict[str, Path]]] = []
    incomplete: list[str] = []
    for d in sorted(p for p in src_root.iterdir() if p.is_dir()):
        files = clip_files(d)
        (clips.append((d.name, files)) if files else incomplete.append(d.name))

    # Injectivity: two clip ids must never sanitise to the same key. Checked across the
    # whole owner, not per partition — a colliding pair that lands in two partitions today
    # would collide the moment anyone re-judges one of them into the other.
    seen: dict[str, str] = {}
    for cid, _ in clips:
        k = wds_key(cid)
        if k in seen:
            raise SystemExit(f"key collision: {cid!r} and {seen[k]!r} both -> {k!r}")
        seen[k] = cid

    metas = {cid: read_meta(files["meta.json"]) for cid, files in clips}

    if a.partition == "none":
        groups = {"": clips}
    else:
        groups = {}
        unjudged: list[str] = []
        for cid, files in clips:
            try:
                groups.setdefault(partition_of(metas[cid]), []).append((cid, files))
            except (KeyError, ValueError):
                unjudged.append(cid)
        if unjudged:
            # Fail rather than guess. Packing these as `rejected` would publish clips that
            # were never judged as clips the policy turned down.
            raise SystemExit(
                f"{len(unjudged)} clips have no usable kept_tier in meta.json, e.g. "
                f"{', '.join(unjudged[:5])}. Run the assembler over this corpus first, or "
                f"pass --partition none.")

    total_shards = 0
    for name in sorted(groups):
        part_clips = groups[name]
        n_shards, written, skipped = pack_partition(
            name or a.owner, out_root / name if name else out_root,
            part_clips, metas, a.shard_bytes)
        total_shards += n_shards
        label = name or "(flat)"
        print(f"  {a.owner}/{label}: {len(part_clips)} clips -> {n_shards} shards "
              f"({written} written, {skipped} already present)")

    print(f"  {len(clips)} clips, {total_shards} shards under {out_root}")
    if incomplete:
        # Name missing payloads so users can locate the affected clips.
        print(f"  SKIPPED {len(incomplete)} clips missing a required payload: "
              f"{', '.join(incomplete[:5])}{' ...' if len(incomplete) > 5 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
