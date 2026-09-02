#!/usr/bin/env python3
"""Check whether generated depth and pose share one metric scale using ground truth.

Photometric scale checks depend on parallax and can be unreliable for individual clips.

OmniWorld clips ship both GT camera poses and GT metric depth. Run the depth path on one
of these clips and two independent
factors fall out:

    alpha = our_depth / gt_depth              (is the emitted depth metric?)
    beta  = our_translation / gt_translation  (is the emitted trajectory metric?)

"Depth and pose share one scale" is exactly alpha == beta. Both being ~1 additionally says
the shared scale is metres. This is the claim a downstream user actually depends on when
they sample anchor frames by depth and expect the poses to agree with them.

    verify_depth_pose_scale_gt.py <gt_clip_dir> <produced_clip_dir>
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def load_exr_depth(zip_path: Path, idx: list[int]) -> np.ndarray:
    """VIPE's depth/video.zip -> (len(idx),h,w) metres. The channel is 'Z'; cv2 reads these
    as empty because it looks for RGB, which is silent and reads as "depth is all zero"."""
    import Imath
    import OpenEXR
    z = zipfile.ZipFile(zip_path)
    names = sorted(n for n in z.namelist() if n.endswith(".exr"))
    out, tmp = [], Path("/tmp/_gtverify.exr")
    for i in idx:
        tmp.write_bytes(z.read(names[i]))
        f = OpenEXR.InputFile(str(tmp))
        dw = f.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        out.append(np.frombuffer(f.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT)),
                                 dtype=np.float32).reshape(h, w))
    tmp.unlink(missing_ok=True)
    return np.stack(out)


def umeyama_scale(A: np.ndarray, B: np.ndarray) -> float:
    """Similarity scale taking A onto B (both (N,3) camera centres), Umeyama 1991.

    Scale only - the rotation and offset between the two trajectories are free and carry no
    information about metric size, which is the one thing being measured.
    """
    A0, B0 = A - A.mean(0), B - B.mean(0)
    H = A0.T @ B0 / len(A)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return float(np.trace(np.diag(D) @ S) / (A0 ** 2).sum() * len(A))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", help="corpus clip dir: gt_depth.npz + poses.npy (GT, metric)")
    ap.add_argument("produced", help="depth-batch output dir for the SAME clip")
    ap.add_argument("--frames", type=int, default=12)
    a = ap.parse_args()

    gt, pr = Path(a.gt), Path(a.produced)
    gt_poses, our_poses = np.load(gt / "poses.npy"), np.load(pr / "poses.npy")
    n = min(len(gt_poses), len(our_poses))
    if len(gt_poses) != len(our_poses):
        print(f"NOTE: {len(gt_poses)} GT poses vs {len(our_poses)} ours — comparing {n}")

    # poses.npy is camera-to-world, so the camera centre is the translation column as it
    # stands. Taking -R^T t here would silently compare two different curves.
    ours_C, gt_C = our_poses[:n, :3, 3], gt_poses[:n, :3, 3]
    beta = umeyama_scale(ours_C, gt_C)
    # A ratio of two trajectory sizes says nothing when the trajectory has no size: over a
    # 5s window a camera that barely moves gives tiny/tiny, and the answer is noise wearing
    # a decimal point. Report how far the GT camera actually travelled so the number can be
    # interpreted only when the camera travelled far enough.
    gt_disp = float(np.linalg.norm(gt_C[-1] - gt_C[0]))
    gt_path = float(np.linalg.norm(np.diff(gt_C, axis=0), axis=1).sum())

    idx = sorted(set(np.linspace(0, n - 1, a.frames).astype(int).tolist()))
    gd = np.load(gt / "gt_depth.npz")
    gt_depth = gd[gd.files[0]][idx].astype(np.float64)
    our_depth = load_exr_depth(pr / "depth_exr.zip", idx).astype(np.float64)
    if our_depth.shape[1:] != gt_depth.shape[1:]:
        import cv2
        our_depth = np.stack([cv2.resize(d, (gt_depth.shape[2], gt_depth.shape[1]),
                                         interpolation=cv2.INTER_NEAREST) for d in our_depth])

    ok = (np.isfinite(gt_depth) & np.isfinite(our_depth)
          & (gt_depth > 0.1) & (gt_depth < 200) & (our_depth > 0.1))
    if ok.sum() < 1000:
        print("too few valid depth pixels to compare")
        return 1
    ratio = our_depth[ok] / gt_depth[ok]
    alpha = float(np.median(ratio))
    spread = float(np.percentile(ratio, 84) / np.percentile(ratio, 16))

    # One median hides the difference between the two failures that matter. A constant
    # factor across distance is a scale error and is correctable by one multiply; a factor
    # that drifts with distance is a non-linearity (a depth aligned in disparity rather than
    # in metres, say) and no single number fixes it. Report the ratio per GT-depth decile.
    gtv = gt_depth[ok]
    edges = np.percentile(gtv, np.linspace(0, 100, 11))
    print(f"clip {gt.name}: {n} frames, {ok.sum():,} depth pixels compared")
    print("  ours/GT by GT-depth decile:")
    for i in range(10):
        sel = (gtv >= edges[i]) & (gtv < edges[i + 1] if i < 9 else gtv <= edges[10])
        if sel.sum() > 100:
            print(f"    {edges[i]:7.1f}-{edges[i+1]:7.1f} m : {np.median(ratio[sel]):.3f}")
    # If the deciles drift, test the specific suspicion: a depth that is affine in DISPARITY
    # rather than in metres, i.e. 1/ours = a*(1/GT) + b. That is what an alignment carried
    # out in inverse-depth space leaves behind, and it is not repairable by any single
    # multiplier - which is the difference that decides whether the shipped depth can be
    # rescaled onto the pose scale or has to be regenerated.
    inv_g, inv_o = 1.0 / gtv, 1.0 / our_depth[ok]
    A = np.stack([inv_g, np.ones_like(inv_g)], 1)
    (fa, fb), *_ = np.linalg.lstsq(A, inv_o, rcond=None)
    resid = np.median(np.abs(inv_o - (fa * inv_g + fb)) / np.maximum(inv_o, 1e-9))
    print(f"  disparity fit: 1/ours = {fa:.3f}*(1/GT) + {fb:.4f}   "
          f"(median residual {resid:.1%}; b=0 would mean a pure scale)")
    print(f"  depth  : ours / GT        = {alpha:.3f}   (per-pixel 16-84% spread {spread:.2f}x)")
    print(f"  poses  : ours / GT        = {beta:.3f}   "
          f"(GT camera travelled {gt_path:.1f} m, net {gt_disp:.1f} m)")
    print(f"  ratio  : depth vs pose    = {alpha / beta:.3f}")
    if gt_path < 2.0:
        print(f"\nUNRELIABLE: the GT camera moved {gt_path:.1f} m in this window — the pose "
              f"scale is a ratio of two near-zero numbers. Use a clip that moves.")
        return 0
    off = abs(np.log(alpha / beta))
    if off < np.log(1.15):
        print("\ndepth and pose are in ONE scale (within 15%)")
        if abs(np.log(alpha)) < np.log(1.15):
            print("and that shared scale is metres — both match GT directly")
    else:
        print(f"\nMISMATCH: the depth is {alpha / beta:.2f}x the pose scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
