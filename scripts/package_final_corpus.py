#!/usr/bin/env python3
"""Pack a finalized training list into deterministic fixed-size tar archives.

The command supports on-disk archives and streaming emission. It stops before the
configured free-space floor. Source reclamation is optional and applies only to archives
that have passed verification and are explicitly listed by the user.

Grouping is by POSITION IN THE LIST, which is safe precisely because the list is finished:
tar N always holds the same 200 clips no matter when it is built, so an interrupted run
resumes without re-packing and without assigning a clip to multiple archives.

    package_final_corpus.py --list train_list.jsonl --corpus <root> --out <dir>
    package_final_corpus.py --out <dir> --reclaim collected.txt      # free the sources
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

GROUP = 200
# Files required by the data contract. prompt.txt is carried when available so workflows
# that attach captions separately can package the remaining annotations.
REQUIRED = ("video.mp4", "poses.npy", "intrinsics.npy", "meta.json")
OPTIONAL = ("prompt.txt",)


def log(msg: str) -> None:
    print(msg, flush=True)


def free_tb(path: Path) -> float:
    """Return free space reported for the output directory's filesystem."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e12


def clip_dir(corpus: Path, rec: dict) -> Path:
    """Locate a clip from its train_list record."""
    sp = rec.get("store_path")
    if sp:
        return corpus / sp.rstrip("/")
    return corpus / rec["source"] / "clips" / rec["clip_id"]


def build(recs: list[dict], corpus: Path, tar_path: Path, stage: Path) -> tuple[int, list]:
    """Hard-link the clips into a staging tree and tar it. Returns (bytes, skipped)."""
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    skipped = []
    try:
        for r in recs:
            src = clip_dir(corpus, r)
            missing = [f for f in REQUIRED if not (src / f).exists()]
            if missing:
                skipped.append({"clip_id": r["clip_id"], "source": r["source"],
                                "missing": missing})
                continue
            d = stage / f"{r['source']}__{r['clip_id']}"
            d.mkdir(parents=True, exist_ok=True)
            for f in REQUIRED + OPTIONAL:
                p = src / f
                if p.exists():
                    os.link(p, d / f)          # same filesystem: no second copy while staging
        with tarfile.open(tar_path, "w") as tf:        # already-compressed payloads
            for d in sorted(stage.iterdir()):
                tf.add(d, arcname=d.name)
        return tar_path.stat().st_size, skipped
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def verify_tar(tar_path: Path, expect: dict[str, int]) -> tuple[bool, str]:
    """Verify that an archive is a faithful, complete copy of its source files.

    Three checks, because after the sources go this file is the only copy:
      * every expected member is present, and nothing extra is;
      * each member's recorded size equals the source file's size (catches truncation);
      * every member's DATA reads back without error (catches a half-written archive that
        still has a valid-looking index).
    """
    try:
        with tarfile.open(tar_path, "r") as tf:
            got = {}
            for m in tf.getmembers():
                if m.isfile():
                    got[m.name] = m.size
            if got.keys() != expect.keys():
                miss = sorted(set(expect) - set(got))[:3]
                extra = sorted(set(got) - set(expect))[:3]
                return False, f"member mismatch (missing e.g. {miss}, extra e.g. {extra})"
            bad = [n for n, sz in expect.items() if got[n] != sz]
            if bad:
                return False, f"{len(bad)} members differ in size, e.g. {bad[0]}"
            with tarfile.open(tar_path, "r") as tf2:
                for m in tf2.getmembers():
                    if not m.isfile():
                        continue
                    f = tf2.extractfile(m)
                    if f is None:
                        return False, f"cannot read {m.name}"
                    n = 0
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        n += len(chunk)
                    if n != m.size:
                        return False, f"{m.name} truncated: read {n} of {m.size}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, "ok"


def release_sources(recs: list[dict], corpus: Path, packed: set[str]) -> int:
    """Delete the source files of the clips actually packed. Returns bytes freed."""
    freed = 0
    for r in recs:
        if f"{r['source']}__{r['clip_id']}" not in packed:
            continue                     # skipped clip: its source stays put
        d = clip_dir(corpus, r)
        if not d.is_dir():
            continue
        for f in d.iterdir():
            freed += f.stat().st_size
            f.unlink()
        d.rmdir()
    return freed


def reclaim(out: Path, corpus: Path, collected: Path) -> None:
    """Delete the source clips of tars named in `collected` — the ONLY deleting path here.

    Guarded twice over: the tar must have a manifest entry (so we know exactly what it
    holds) and must no longer be present in the output directory (i.e. it really was taken
    away, not merely listed). Freeing the sources while the only other copy still sits on
    the same disk would release no space and lose the redundancy at the same time.
    """
    man = {}
    with open(out / "manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            man[r["tar"]] = r
    freed = 0
    for name in collected.read_text().split():
        rec = man.get(name)
        if rec is None:
            log(f"SKIP {name}: no manifest entry — refusing to delete sources for it")
            continue
        if (out / name).exists():
            log(f"SKIP {name}: still present in {out} — not collected yet")
            continue
        for c in rec["clips"]:
            src = corpus / c["store_path"].rstrip("/")
            if src.is_dir():
                for p in src.iterdir():
                    freed += p.stat().st_size
                    p.unlink()
                src.rmdir()
        log(f"{name}: released {len(rec['clips'])} clips")
    log(f"reclaimed {freed/1e12:.2f} TB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="archive output directory (not used by --emit)")
    ap.add_argument("--list", help="train_list.jsonl (finished)")
    ap.add_argument("--corpus", default=os.environ.get("SOLAR_WM_LOCAL_ROOT", ""),
                    help="root that store_path is relative to")
    ap.add_argument("--group", type=int, default=GROUP)
    ap.add_argument("--free-floor-tb", type=float, default=1.0,
                    help="stop packing while free space is under this")
    ap.add_argument("--max-tars", type=int, default=0, help="0 = as many as fit")
    ap.add_argument("--release-sources", action="store_true",
                    help="after a tar VERIFIES, delete the source clips it contains. Keeps "
                         "peak usage at one tar so the whole corpus can be packed on a disk "
                         "that cannot hold sources and tars at once. The tar becomes the "
                         "only local copy, so verification is mandatory and a failed "
                         "check leaves the sources untouched.")
    ap.add_argument("--reclaim", help="file listing collected tar names; frees their sources")
    ap.add_argument("--emit", type=int, default=-1, metavar="N",
                    help="write tar N to stdout and exit without staging an archive on disk")
    a = ap.parse_args()

    corpus = Path(a.corpus)
    if a.emit < 0 and not a.out:
        ap.error("--out is required unless --emit is given")
    # Only touch the output dir for the modes that use it: --emit writes to stdout and must
    # not create anything, least of all on a filesystem this exists to keep clear.
    out = Path(a.out) if a.out else Path(".")
    if a.emit < 0:
        out.mkdir(parents=True, exist_ok=True)
    if a.reclaim:
        reclaim(out, corpus, Path(a.reclaim))
        return 0
    if not a.list:
        ap.error("--list is required unless --reclaim is given")

    recs = [json.loads(l) for l in Path(a.list).read_text().splitlines() if l.strip()]
    groups = [recs[i:i + a.group] for i in range(0, len(recs), a.group)]

    if a.emit >= 0:
        # Straight to stdout without a staging archive. The same
        # grouping as the on-disk mode, so tar N is byte-for-byte the same set either way.
        if a.emit >= len(groups):
            ap.error(f"--emit {a.emit} is out of range (0..{len(groups)-1})")
        import sys as _sys
        with tarfile.open(fileobj=_sys.stdout.buffer, mode="w|") as tf:
            for r in groups[a.emit]:
                src = clip_dir(corpus, r)
                if any(not (src / f).exists() for f in REQUIRED):
                    continue
                for f in REQUIRED + OPTIONAL:
                    if (src / f).exists():
                        tf.add(src / f, arcname=f"{r['source']}__{r['clip_id']}/{f}")
        return 0

    # Prove the paths resolve BEFORE packing anything. store_path is relative to a corpus
    # root given on the command line, and if that root is wrong every clip "skips" - which
    # this would otherwise report as a series of successful, empty tars. Fail loudly with
    # the path actually tried instead of shipping nothing and calling it done.
    probe = recs[:: max(1, len(recs) // 20)][:20]
    found = [r for r in probe if all((clip_dir(corpus, r) / f).exists() for f in REQUIRED)]
    if not found:
        log(f"ABORT: none of {len(probe)} sampled clips resolve under --corpus {corpus}")
        log(f"  tried: {clip_dir(corpus, probe[0])}")
        return 2
    if len(found) < len(probe):
        log(f"NOTE: {len(probe)-len(found)} of {len(probe)} sampled clips are incomplete; "
            f"they will be skipped and recorded per tar")
    log(f"{len(recs)} clips -> {len(groups)} tars of {a.group}")

    man_path, built = out / "manifest.jsonl", 0
    done = set()
    if man_path.exists():
        with open(man_path) as f:
            done = {json.loads(l)["tar"] for l in f if l.strip()}

    # Catch up on groups packed BEFORE releasing was switched on: their tar exists and is
    # recorded, but their sources were never freed, so both copies sit on the disk. Verify
    # each one exactly as a fresh group is verified, then release.
    if a.release_sources and man_path.exists():
        caught = freed_all = 0
        with open(man_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                tp = out / rec["tar"]
                if not tp.exists():
                    continue                      # already collected and reclaimed
                live = [c for c in rec["clips"]
                        if (corpus / c["store_path"].rstrip("/")).is_dir()]
                if not live:
                    continue                      # sources already gone: nothing to do
                expect = {}
                for c in rec["clips"]:
                    nm = f"{c['source']}__{c['clip_id']}"
                    d = corpus / c["store_path"].rstrip("/")
                    for fn in REQUIRED + OPTIONAL:
                        if (d / fn).exists():
                            expect[f"{nm}/{fn}"] = (d / fn).stat().st_size
                ok, why = verify_tar(tp, expect)
                if not ok:
                    log(f"{rec['tar']}: catch-up VERIFY FAILED ({why}) — sources kept")
                    continue
                packed = {f"{c['source']}__{c['clip_id']}" for c in rec["clips"]}
                freed_all += release_sources(
                    [{"source": c["source"], "clip_id": c["clip_id"],
                      "store_path": c["store_path"]} for c in rec["clips"]], corpus, packed)
                caught += 1
        if caught:
            log(f"catch-up: released sources for {caught} already-packed groups "
                f"({freed_all/1e9:.1f} GB)")
    for i, g in enumerate(groups):
        name = f"final_{i:05d}.tar"
        if name in done:
            continue
        free = free_tb(out)
        if free < a.free_floor_tb:
            log(f"STOP: {free:.2f} TB free is under the {a.free_floor_tb} TB floor — "
                f"{built} tars built this pass, {len(groups)-len(done)-built} still to go. "
                f"Move completed archives off this filesystem, run --reclaim if desired, "
                f"then run this command again.")
            break
        if a.max_tars and built >= a.max_tars:
            log(f"stopping at --max-tars {a.max_tars}")
            break
        size, skipped = build(g, corpus, out / name, out / "_stage")
        if len(skipped) == len(g):
            # Every clip in this group is gone or incomplete. Recording it keeps resume from
            # retrying it forever, but leaving the file would put a valid-looking, empty
            # package in the output directory.
            (out / name).unlink(missing_ok=True)
            log(f"{name}: EMPTY — all {len(g)} clips incomplete/missing; no file written")
        packed = {f"{r['source']}__{r['clip_id']}" for r in g
                  if not any(s["clip_id"] == r["clip_id"] for s in skipped)}
        if a.release_sources and packed:
            expect = {}
            for r in g:
                nm = f"{r['source']}__{r['clip_id']}"
                if nm not in packed:
                    continue
                d = clip_dir(corpus, r)
                for fn in REQUIRED + OPTIONAL:
                    fp = d / fn
                    if fp.exists():
                        expect[f"{nm}/{fn}"] = fp.stat().st_size
            ok, why = verify_tar(out / name, expect)
            if not ok:
                # The tar is the only thing that would remain; if it does not check out,
                # throw the TAR away, never the sources, and stop rather than plough on.
                (out / name).unlink(missing_ok=True)
                log(f"{name}: VERIFY FAILED ({why}) — tar deleted, sources untouched, stopping")
                return 3
        with open(man_path, "a") as f:
            f.write(json.dumps({
                "tar": name, "bytes": size,
                "clips": [{"clip_id": r["clip_id"], "source": r["source"],
                           "store_path": r["store_path"]} for r in g
                          if not any(s["clip_id"] == r["clip_id"] for s in skipped)],
                "skipped": skipped}) + "\n")
        if a.release_sources and packed:
            # Manifest first, then delete: if this dies mid-way the record of what the tar
            # holds already exists on disk.
            freed = release_sources(g, corpus, packed)
            log(f"{name}: verified, released {freed/1e9:.1f} GB of sources")
        built += 1
        if len(skipped) != len(g):
            log(f"{name}: {len(g)-len(skipped)} clips, {size/1e9:.1f} GB"
                + (f", {len(skipped)} skipped (incomplete)" if skipped else ""))
    log(f"done: {built} tars this pass, {len(done)+built}/{len(groups)} total, "
        f"{free_tb(out):.2f} TB free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
