#!/usr/bin/env python3
"""Build a model view — fixed-length windows cut from the canonical corpus.

The corpus separates three namespaces. The PHYSICAL corpus holds canonical samples and
annotations; a LOGICAL recipe (`build_recipe.py`) holds split membership, tier policy and
source weights; a MODEL VIEW — this script — holds the backbone-specific windows a reader
actually consumes. Keeping them apart is what makes each cheap: a new mixture is a new
recipe, a new context length is a new view, and neither copies a single video.

    python3 scripts/build_window_view.py --meta-dir <dir of <owner>.jsonl> \\
        --window 153 --out views/w153 --max-frames 956
    ... --window 957 --out views/w957 --min-frames 957

WINDOWS ARE NON-OVERLAPPING AND CONTIGUOUS. A clip of N frames yields floor(N / W) windows
at offsets 0, W, 2W, ...; the remainder is dropped rather than stretched. Never subsample a
long clip down to W frames: that is a hidden N-fold speed-up which inflates optical flow
and fails the motion gate. The `--min-frames` and `--max-frames` options express an
explicit length band. Use the released window index when exact release membership is
required; rebuilding a view applies the rules selected on this command.

Split membership ranks the CLIP, so every window cut from one clip lands on the same side.
Windows of one clip share nearly all their frames, so splitting them across train and test
leaks the test set while every id-level overlap check still reads clean.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_wm_data.split import split_rank      # noqa: E402  the one split rule

TIERS = ("xhigh", "high")


def windows_for(num_frames: int, window: int) -> list[int]:
    """Start frames of the non-overlapping windows a clip of this length can serve."""
    if window <= 0:
        raise ValueError("window must be positive")
    return [i * window for i in range(num_frames // window)]


def eligible_rows(rows: list[dict], tiers: set[str], window: int,
                  min_frames: int, max_frames: int | None) -> list[dict]:
    out = []
    for r in rows:
        if r.get("kept_tier") not in tiers:
            continue
        n = r.get("num_frames")
        if not isinstance(n, int) or n < max(window, min_frames):
            continue
        if max_frames is not None and n > max_frames:
            continue
        out.append(r)
    return out


def load_owner(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, required=True, help="window length in frames")
    ap.add_argument("--tier", default="xhigh,high")
    ap.add_argument("--owners", default="")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="ignore clips shorter than this (default: the window length)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="ignore clips longer than this, e.g. to reserve long clips for a "
                         "long-video view")
    ap.add_argument("--test-per-owner", type=int, default=100,
                    help="held-out CLIPS per owner; all of a clip's windows go with it")
    a = ap.parse_args()

    tiers = {t.strip() for t in a.tier.split(",") if t.strip()}
    unknown = tiers - set(TIERS)
    if unknown:
        raise SystemExit(f"unknown tier(s) {sorted(unknown)}; valid: {list(TIERS)}")
    wanted = {o.strip() for o in a.owners.split(",") if o.strip()}

    files = sorted(Path(a.meta_dir).glob("*.jsonl"))
    if wanted:
        missing = wanted - {f.stem for f in files}
        if missing:
            raise SystemExit(f"--owners names {sorted(missing)}, not present in {a.meta_dir}")
        files = [f for f in files if f.stem in wanted]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train: list[dict] = []
    test: list[dict] = []
    report: dict = {"window": a.window, "tiers": sorted(tiers),
                    "min_frames": a.min_frames or a.window, "max_frames": a.max_frames,
                    "owners": {}}

    for f in files:
        owner = f.stem
        rows = eligible_rows(load_owner(f), tiers, a.window, a.min_frames, a.max_frames)
        # Rank the CLIP, not the window: a clip is held out whole or not at all.
        rows.sort(key=lambda r: (split_rank(str(r.get("clip_id", ""))),
                                 str(r.get("clip_id", ""))))
        n_test = min(a.test_per_owner, len(rows))
        n_win_test = n_win_train = 0
        for i, r in enumerate(rows):
            side, cid = (test if i < n_test else train), str(r.get("clip_id", ""))
            for start in windows_for(int(r["num_frames"]), a.window):
                side.append({"owner": owner, "clip_id": cid,
                             "start_frame": start, "num_frames": a.window,
                             "kept_tier": r.get("kept_tier")})
                if i < n_test:
                    n_win_test += 1
                else:
                    n_win_train += 1
        report["owners"][owner] = {
            "clips": len(rows), "test_clips": n_test,
            "train_windows": n_win_train, "test_windows": n_win_test,
        }

    # The property a training curve cannot reveal. Checked at CLIP level, because two
    # windows of one clip overlap in content even when their (clip_id, start) ids differ.
    train_clips = {(r["owner"], r["clip_id"]) for r in train}
    test_clips = {(r["owner"], r["clip_id"]) for r in test}
    overlap = train_clips & test_clips
    if overlap:
        raise SystemExit(f"a clip appears in both splits: {sorted(overlap)[:3]}")
    for name, rows in (("train", train), ("test", test)):
        ids = {(r["owner"], r["clip_id"], r["start_frame"]) for r in rows}
        if len(ids) != len(rows):
            raise SystemExit(f"duplicate window within {name}")

    for name, rows in (("train", train), ("test", test)):
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")

    report["train_windows"] = len(train)
    report["test_windows"] = len(test)
    report["train_clips"] = len(train_clips)
    report["test_clips"] = len(test_clips)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  window {a.window}, {len(report['owners'])} owners")
    print(f"  train {len(train)} windows from {len(train_clips)} clips")
    print(f"  test  {len(test)} windows from {len(test_clips)} clips, zero clip overlap")
    per = collections.Counter(r["owner"] for r in train)
    if per:
        top = ", ".join(f"{o}:{n}" for o, n in per.most_common(4))
        print(f"  largest contributors: {top}")
    print(f"  {out}/train.jsonl  {out}/test.jsonl  {out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
