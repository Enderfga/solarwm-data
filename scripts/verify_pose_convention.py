#!/usr/bin/env python3
"""Pose-convention arbiter for new gt_pose sources (GPU; runs in the fleet env).

For produced corpus clips of a source, runs Pi3 on the video and checks the stored GT
poses against Pi3's estimate two ways:

1. Sim(3) Umeyama on camera CENTERS -> normalised RMS residual. A wrong trajectory
   convention (w2c stored as c2w, bad axis chain) deforms the curve: residual blows up.
   (A global rotation/scale/translation is gauge freedom — absorbed, as it should be.)
2. Camera-AXIS consistency: with the Umeyama gauge R_g, expect R_pi3_i ≈ R_g · R_gt_i.
   The per-frame deviation angle(R_pi3_iᵀ · R_g · R_gt_i) should be small and stable;
   a fixed-permutation camera-frame error shows up as a large constant offset (~90°+).

Verdict per clip: PASS if resid_norm < 0.15 and median axis-dev < 25°.
Static clips (GT span ~0) are skipped (Umeyama degenerate; nothing to verify).

Usage (fleet env): python3 scripts/verify_pose_convention.py <source> [n_clips=3]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)
sys.path.insert(0, f"{WM}/third_party/Pi3")

from solar_wm_data import cos_io  # noqa: E402
from solar_wm_data.pose import _real, adapters  # noqa: E402


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Sim(3) aligning src->dst (N,3). Returns (s, R, t, rms)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    rms = float(np.sqrt(((s * (R @ src.T).T + t - dst) ** 2).sum(1).mean()))
    return s, R, t, rms


def rot_angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def verify_clip(clip_dir: Path, n_pi3: int = 64):
    gt = np.load(clip_dir / "poses.npy")              # (N,4,4) c2w
    N = gt.shape[0]
    idx = adapters.even_indices(N, min(n_pi3, N))
    gt_sub = gt[idx]
    span = float(np.linalg.norm(gt_sub[:, :3, 3].max(0) - gt_sub[:, :3, 3].min(0)))
    if span < 0.05:
        return None                                    # static — degenerate, skip
    # near-COLLINEAR trajectory (straight drive): the position-Umeyama rotation gauge is
    # under-determined AROUND the line -> axis_dev becomes an arbitrary constant offset
    # (observed on zod straight segments: constant 30°/175° within a clip). Skip like
    # static — convention must be judged on clips with actual turns.
    cen = gt_sub[:, :3, 3] - gt_sub[:, :3, 3].mean(0)
    ev = np.linalg.eigvalsh(cen.T @ cen / len(cen))    # ascending
    if ev[1] / max(ev[2], 1e-12) < 0.02:
        return None                                    # collinear — gauge degenerate, skip
    frames = adapters.read_frames(str(clip_dir / "video.mp4"), len(idx))
    pi3, _ = _real.pi3_infer(frames)                   # (n,4,4) c2w, scale-ambiguous
    s, Rg, t, rms = umeyama_sim3(gt_sub[:, :3, 3], pi3[:, :3, 3])
    resid_norm = rms / max(span * s, 1e-9)             # residual in PI3 units / gt span
    devs = [rot_angle(pi3[i, :3, :3].T @ Rg @ gt_sub[i, :3, :3]) for i in range(len(idx))]
    return {"span_m": span, "scale": s, "resid_norm": resid_norm,
            "axis_dev_med": float(np.median(devs)), "axis_dev_max": float(np.max(devs))}


def main():
    source = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    pre = f"{cos_io.corpus_prefix(source)}/clips/"
    clip_ids, seen = [], set()
    for k in cos_io.list_keys(pre):
        cid = k[len(pre):].split("/")[0]
        if cid not in seen:
            seen.add(cid)
            clip_ids.append(cid)
        if len(clip_ids) >= want * 4:                  # spares for static skips
            break
    print(f"[verify] {source}: corpus clips found {len(seen)}+, checking up to {want}")
    done = 0
    verdicts = []
    for cid in clip_ids:
        if done >= want:
            break
        local = Path(f"/tmp/vpc/{cid}")
        local.mkdir(parents=True, exist_ok=True)
        try:
            cos_io.get_file(f"{pre}{cid}/video.mp4", str(local / "video.mp4"))
            cos_io.get_file(f"{pre}{cid}/poses.npy", str(local / "poses.npy"))
            r = verify_clip(local)
        except Exception as e:  # noqa: BLE001
            print(f"  {cid}: ERROR {e}")
            continue
        if r is None:
            print(f"  {cid}: static (span<5cm) — skipped")
            continue
        ok = r["resid_norm"] < 0.15 and r["axis_dev_med"] < 25.0
        verdicts.append(ok)
        done += 1
        print(f"  {cid}: span={r['span_m']:.2f}m scale={r['scale']:.3f} "
              f"resid_norm={r['resid_norm']:.3f} axis_dev med/max="
              f"{r['axis_dev_med']:.1f}/{r['axis_dev_max']:.1f}° -> {'PASS' if ok else 'FAIL'}")
    if not verdicts:
        print(f"[verify] {source}: NO verifiable clips (all static/errors)")
        sys.exit(2)
    if all(verdicts):
        print(f"[verify] {source}: CONVENTION OK ({len(verdicts)} clips)")
        sys.exit(0)
    print(f"[verify] {source}: CONVENTION SUSPECT — {verdicts.count(False)}/{len(verdicts)} fail")
    sys.exit(1)


if __name__ == "__main__":
    main()
