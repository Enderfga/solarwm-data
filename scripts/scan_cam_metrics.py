#!/usr/bin/env python3
"""Distribution scan for the camera-dynamic selector (calibration step A2).

select_hq_cam_dynamic.py applies a quality gate AND a camera-motion gate, but its
motion thresholds were first set by guess. This script samples each source and dumps
EVERY clip's raw metrics (trajectory motion + quality) WITHOUT gating, so the real
per-source / per-pose-mode distributions can be inspected and the gates calibrated
against data instead of assumptions.

Reuses the selector's verified trajectory maths (c2w camera-centre convention) and
S3 listing so the scan and the final selection agree bit-for-bit.

Usage (on an in-region pod with S3 creds):
    SOLAR_WM_STORAGE=s3 SOLAR_WM_S3_BUCKET=<your-bucket> SOLAR_WM_CORPUS_PREFIX=corpus \\
    PYTHONPATH=. python3 scripts/scan_cam_metrics.py --out /tmp/scan --sample 1500
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)

from solar_wm_data import cos_io  # noqa: E402
from select_hq_cam_dynamic import (  # noqa: E402  reuse the verified pieces
    DEFAULT_SOURCES, clip_ids, pose_mode, traj_metrics,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/scan")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--sample", type=int, default=1500,
                    help="per-source clip cap; sources with fewer are scanned whole")
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    c, b = cos_io.client()
    pool = ThreadPoolExecutor(max_workers=a.threads)

    fout = open(out / "scan_metrics.jsonl", "w")
    summary = {}

    def fetch(cid, pre):
        base = f"{pre}{cid}/"
        try:
            meta = json.loads(c.get_object(Bucket=b, Key=base + "meta.json")["Body"].read())
            pb = c.get_object(Bucket=b, Key=base + "poses.npy")["Body"].read()
            return cid, meta, np.load(io.BytesIO(pb))
        except Exception:  # noqa: BLE001 - mid-upload / missing; counted, not fatal
            return cid, None, None

    for src in sources:
        mode = pose_mode(src)
        ids = clip_ids(src, a.sample, 0, 1)
        pre = f"{cos_io.corpus_prefix(src)}/clips/"
        n_seen = n_err = n_ok = 0
        for cid, meta, poses in pool.map(lambda cid: fetch(cid, pre), ids):
            n_seen += 1
            if meta is None or poses is None:
                n_err += 1
                continue
            tm = traj_metrics(poses)
            if tm is None:
                n_err += 1
                continue
            m = meta.get("metrics") or {}
            cam = meta.get("camera") or {}
            rec = {
                "src": src, "mode": mode, "id": cid,
                "n_frames": tm["n_frames"],
                "avg_motion_m": round(tm["avg_motion_m"], 6),
                "span_m": round(tm["span_m"], 4),
                "path_length": round(tm["path_length"], 4),
                "tortuosity": round(tm["tortuosity"], 4),
                "max_seg": round(tm["max_seg"], 4),
                "jump_ratio": round(tm["jump_ratio"], 3),
                "ang_med": round(tm["ang_med"], 4),
                "ang_p95": round(tm["ang_p95"], 4),
                "ang_max": round(tm["ang_max"], 4),
                "dover_tech": m.get("dover_tech"),
                "dover_aes": m.get("dover_aes"),
                "unimatch": m.get("unimatch"),
                "scene_cuts": m.get("scene_cuts"),
                "scale_cov": cam.get("scale_cov"),
                "fov_x": cam.get("fov_x"),
                "fov_y": cam.get("fov_y"),
                "focal_div": cam.get("focal_div"),
            }
            fout.write(json.dumps(rec) + "\n")
            n_ok += 1
        fout.flush()
        summary[src] = dict(mode=mode, seen=n_seen, ok=n_ok, err=n_err)
        print(f"[{time.strftime('%H:%M:%S')}] {src:14s} mode={mode:8s} "
              f"seen={n_seen} ok={n_ok} err={n_err}", flush=True)

    fout.close()
    json.dump(summary, open(out / "scan_summary.json", "w"), indent=2)
    print(f"\n# wrote {out/'scan_metrics.jsonl'}")


if __name__ == "__main__":
    main()
