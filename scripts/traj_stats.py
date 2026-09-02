#!/usr/bin/env python3
"""Per-clip camera-trajectory statistics for the whole corpus.

WHY. Every gate the corpus has judges the PIXELS
(quality, flow, saturation, cuts) or the intrinsics. None of them judges the thing this
corpus exists to teach: the camera's motion. A clip can be pristine by every visual metric
and still carry a trajectory that is useless (the camera never moves) or unlearnable (it
jitters in place). Optical flow is not a substitute: flow is large whenever the SCENE moves,
and small whenever the camera translates toward a distant background.

Poses are c2w and metric-scaled (Umeyama against Pi3 structure), so the camera centre is
`M[:3,3]` directly — NOT -R^T t — and lengths are in metres, comparable across sources.

Emitted per clip (one JSON object per line, per source):
  path      Σ‖ΔC‖                     total distance travelled (m)
  disp      ‖C[-1]-C[0]‖              net displacement (m)
  diag      bbox diagonal of C        extent of the trajectory (m)
  tort      path / diag               tortuosity; clean trajectories sit at ~1-8
  straight  disp / path               1 = straight line, ~0 = returns to start
  rot_deg   Σ per-step rotation       total rotation along the clip (deg)
  fov_sweep angle(first fwd,last fwd) net change of viewing direction (deg)
  vel_med   median ‖ΔC‖ per frame     typical per-frame speed (m/frame)
  vel_p95   95th pct ‖ΔC‖             jitter/spike detector against vel_med

Usage:
  python3 scripts/traj_stats.py --out <dir> [--sources a,b] [--sample N] [--threads 48]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)

from solar_wm_data import cos_io  # noqa: E402
from solar_wm_data.ingest import SOURCE_MODE  # noqa: E402


def stats_from_poses(P: np.ndarray) -> dict | None:
    """P: (N,4,4) camera-to-world. Returns None if the array is not a usable trajectory."""
    if P.ndim != 3 or P.shape[1:] != (4, 4) or len(P) < 2:
        return None
    C = P[:, :3, 3].astype(np.float64)
    if not np.isfinite(C).all():
        return None
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)   # axis=0! np.diff(C,0) does nothing
    path = float(step.sum())
    disp = float(np.linalg.norm(C[-1] - C[0]))
    diag = float(np.linalg.norm(C.max(0) - C.min(0)))
    R = P[:, :3, :3].astype(np.float64)
    # per-step rotation angle from the trace of R_k^T R_{k+1}
    rel = np.einsum("nij,nik->njk", R[:-1], R[1:])
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos)).sum())
    fwd0, fwd1 = R[0][:, 2], R[-1][:, 2]              # OpenCV convention: +Z is forward
    c = float(np.clip(fwd0 @ fwd1 / (np.linalg.norm(fwd0) * np.linalg.norm(fwd1) + 1e-12),
                      -1.0, 1.0))
    return {
        "path": path, "disp": disp, "diag": diag,
        "tort": (path / diag) if diag > 1e-9 else float("inf"),
        "straight": (disp / path) if path > 1e-9 else 0.0,
        "rot_deg": rot_deg,
        "fov_sweep": float(np.degrees(np.arccos(c))),
        "vel_med": float(np.median(step)), "vel_p95": float(np.percentile(step, 95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default=",".join(sorted(SOURCE_MODE)))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for src in [s.strip() for s in a.sources.split(",") if s.strip()]:
        pre = f"{cos_io.corpus_prefix(src)}/clips/"
        ids = sorted({k[len(pre):].split("/")[0] for k in cos_io.list_keys(pre)
                      if k.endswith("/poses.npy")})
        if not ids:
            continue
        if a.sample:
            step = max(1, len(ids) // a.sample)
            ids = ids[::step][:a.sample]

        def one(cid, _pre=pre):
            try:
                import io
                P = np.load(io.BytesIO(cos_io.get_bytes(f"{_pre}{cid}/poses.npy")))
            except Exception:  # noqa: BLE001 - unreadable clip is counted, not fatal
                return None
            s = stats_from_poses(P)
            if s is None:
                return None
            s["clip_id"] = cid
            return s

        n_bad = 0
        with ThreadPoolExecutor(a.threads) as ex, \
                open(out / f"{src}.jsonl", "w") as fh:
            for cid, r in zip(ids, ex.map(one, ids)):
                # Emit a record for EVERY clip, including the unusable ones. Silently
                # omitting them makes "absent" mean both "not measured" and "measured and
                # bad", and a consumer cannot tell a corrupt clip from one this pass never
                # reached. 1,643 clips of the 5s corpus carry non-finite poses — a file of
                # the right shape and size holding NaN, which every count calls complete.
                if r is None:
                    n_bad += 1
                    fh.write(json.dumps({"clip_id": cid, "ok": False}) + "\n")
                    continue
                r["ok"] = True
                fh.write(json.dumps(r) + "\n")
        print(f"{src}: {len(ids) - n_bad}/{len(ids)} trajectories "
              f"({n_bad} unusable) -> {out}/{src}.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
