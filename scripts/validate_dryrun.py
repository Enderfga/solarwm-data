#!/usr/bin/env python3
"""Standalone end-to-end dry-run validation of the whole pipeline.

Exercises the full pipeline on synthetic samples — ingest -> pose -> filter ->
caption -> package — and asserts the outputs are
well-formed. Prints a summary and exits non-zero on any failure. Intended to run
anywhere the core deps (numpy, pyyaml) are present, e.g. a fresh debug node.

    python3 scripts/validate_dryrun.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# allow running from a checkout without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar_wm_data.config import load_config
from solar_wm_data.manifest import write_manifest, read_manifest
from solar_wm_data.driver import run_pipeline, STAGES
from solar_wm_data.ingest import ingest_clip_dir, mode_for
from solar_wm_data.caption import contains_camera_motion

MODELS = {"dry_run": True, "depth_fusion": {"ema_momentum": 0.99}}


def _mk(root: Path, source: str, name: str) -> Path:
    d = root / source / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "video.mp4").write_bytes(b"")
    return d


def main() -> int:
    filters_cfg = load_config("filters")
    tmp = Path(tempfile.mkdtemp(prefix="solarwm_validate_"))
    print(f"[validate] workdir: {tmp}")

    # --- main pipeline over a few sources ---
    sources = ["miradata", "omniworld", "sekai_game", "spatialvid", "sekai_walking"]
    records = []
    for s in sources:
        for i in range(3):
            d = _mk(tmp / "raw", s, f"{s}_{i:03d}")
            rec = ingest_clip_dir(d, s, fallback={"num_frames": 16, "width": 128, "height": 72})
            assert rec.mode == mode_for(s), f"mode mismatch {s}"
            records.append(rec)
    manifest = tmp / "m.jsonl"
    write_manifest(manifest, records)

    recs = read_manifest(manifest)
    run_pipeline(recs, tmp / "out", STAGES, filters_cfg, MODELS)

    n_kept = 0
    for r in recs:
        assert r.pose_path and Path(r.pose_path).exists(), f"no pose for {r.clip_id}"
        poses = np.load(r.pose_path)
        intr = np.load(r.intrinsics_path)
        assert poses.shape[1:] == (4, 4), f"bad pose shape {poses.shape}"
        assert intr.shape[1] == 4, f"bad intrinsics shape {intr.shape}"
        assert len(r.scale_factors) == r.num_frames
        assert r.kept is not None
        assert r.caption and not contains_camera_motion(r.caption), \
            f"caption leaks camera motion: {r.caption!r}"
        if r.kept:
            n_kept += 1
            pd = Path(r.extra["packaged_dir"])
            for f in ("poses.npy", "intrinsics.npy", "prompt.txt"):
                assert (pd / f).exists(), f"missing {f} in {pd}"
    assert n_kept > 0, "no clips survived filtering"
    print(f"[validate] main pipeline: {len(recs)} clips, {n_kept} kept, all annotated+captioned OK")

    # --- gt_pose metric-scale recovery sanity (Umeyama) ---
    from solar_wm_data.pose.alignment import recover_metric_scale
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(40, 3))
    s = recover_metric_scale(pred, 2.3 * pred, inlier_percentile=80)
    assert abs(s - 2.3) < 1e-3, f"scale recovery off: {s}"
    print(f"[validate] Umeyama scale recovery: {s:.4f} (expected 2.3) OK")

    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
