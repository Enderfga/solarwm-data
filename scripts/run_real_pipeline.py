#!/usr/bin/env python3
"""Run the REAL (non-dry-run) pipeline on a few extracted DL3DV clips.

Ingests clip dirs (source=dl3dv, gt_pose mode), then runs pose (real Pi3 +
Umeyama metric scale against GT poses) -> filter (real cv2 motion/flow +
saturation + PySceneDetect + camera filters) -> caption (real Qwen2.5-VL) ->
package. Prints a per-clip summary of the real artifacts produced.

    SOLAR_WM_WEIGHTS=<weights> python3 scripts/run_real_pipeline.py <clips_dir> <out_dir>
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar_wm_data.config import load_config
from solar_wm_data.manifest import write_manifest
from solar_wm_data.driver import package_clip
from solar_wm_data.ingest import ingest_source
from solar_wm_data.pose.stage import annotate_pose
from solar_wm_data.filter.stage import filter_clip
from solar_wm_data.caption import caption_clip


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    clips_dir, out_dir = sys.argv[1], sys.argv[2]
    source = sys.argv[3] if len(sys.argv) > 3 else "dl3dv"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    records = ingest_source(clips_dir, source)
    if limit:
        records = records[:limit]
    force_mode = os.environ.get("SOLAR_WM_FORCE_MODE")
    if force_mode:
        for r in records:
            r.mode = force_mode  # exercise default / gt_depth path on these clips
        _log(f"forcing pose mode = {force_mode}")
    _log(f"ingested {len(records)} clips from {clips_dir} (source={source})")

    filters_cfg = load_config("filters")
    models_cfg = {"dry_run": False, "depth_fusion": {"ema_momentum": 0.99},
                  "caption_nframes": 8}
    out = Path(out_dir)

    for r in records:
        _log(f"== {r.clip_id}: pose (gt_pose, Pi3+Umeyama) ...")
        annotate_pose(r, out / "work" / "pose", models_cfg)
        _log(f"   pose done: metric_scale={r.scale_factors[0]:.4f}, poses={np.load(r.pose_path).shape}")
        _log(f"   filter (real cv2 metrics + camera) ...")
        filter_clip(r, filters_cfg, models_cfg)
        _log(f"   filter done: kept={r.kept} reasons={r.reject_reasons}")
        if os.environ.get("SOLAR_WM_SKIP_CAPTION"):
            r.caption = ""
            _log("   caption SKIPPED (SOLAR_WM_SKIP_CAPTION set)")
        else:
            _log("   caption (Qwen2.5-VL) ...")
            try:
                r.caption = caption_clip(r, models_cfg)
                _log(f"   caption: {r.caption!r}")
            except Exception as e:  # noqa: BLE001 - caption best-effort
                r.caption = ""
                _log(f"   caption SKIPPED ({type(e).__name__}: {str(e)[:120]})")
        package_clip(r, out / "corpus")
        _log(f"   packaged: {r.extra.get('packaged_dir')}")

    write_manifest(out / "manifest.jsonl", records)

    print("\n===== REAL PIPELINE RESULTS =====")
    for r in records:
        sc = np.load(r.pose_path) if r.pose_path else None
        print(f"\n# {r.clip_id}")
        print(f"  pose: {sc.shape if sc is not None else None} mode={r.pose_mode} "
              f"metric_scale={r.scale_factors[0]:.4f}" if r.scale_factors else "  pose: none")
        c = r.camera
        print(f"  camera: fov_x={_f(c.fov_x)} fov_y={_f(c.fov_y)} "
              f"focal_div={_f(c.focal_div)} scale_cov={_f(c.scale_cov)}")
        m = r.metrics
        print(f"  metrics: sat={_f(m.saturation)} flow={_f(m.unimatch)} "
              f"motion={_f(m.vmaf)} scene_cuts={m.scene_cuts} dover={_f(m.dover_tech)}")
        if r.extra.get("metrics_skipped"):
            print(f"  metrics_skipped: {r.extra['metrics_skipped']}")
        print(f"  kept: {r.kept}  reject: {r.reject_reasons}")
        print(f"  caption: {r.caption!r}")
        if r.kept and r.extra.get("packaged_dir"):
            print(f"  packaged -> {r.extra['packaged_dir']}")

    kept = sum(1 for r in records if r.kept)
    print(f"\n[real] DONE: {len(records)} clips, {kept} kept and packaged.")


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


if __name__ == "__main__":
    main()
