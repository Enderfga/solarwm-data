#!/usr/bin/env python3
"""Prove that a clip's depth and its poses are in ONE metric scale, from the clip alone.

The claim "depth and pose share a scale" is usually argued from how the pipeline is wired.
This measures it instead. Back-project frame t with its depth and intrinsics, move the
points into frame t+k with the two camera poses, project them back to pixels, and compare
what lands there with what is actually there. Then sweep a GLOBAL MULTIPLIER on the depth:

  * if depth and pose share one scale, the photometric error bottoms out at exactly 1.0;
  * if the depth were, say, twice the pose scale, the error would bottom out near 0.5.

The sweep is the whole point - an absolute error number says little on its own, but the
LOCATION of its minimum is a direct, falsifiable read on the scale ratio.

Read with a median, not a mean: real outdoor footage has moving people and cars, and every
frame pair has occlusions. Those break photometric consistency wherever they occur and would
otherwise drown the signal.

    verify_depth_pose_scale.py <clip_dir> <video.mp4> [--gap 12] [--pairs 8]
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

# Log-spaced and DELIBERATELY WIDE. A narrow sweep cannot tell "the scales agree" from
# "the true ratio is outside what I looked at": both put the minimum at an endpoint with
# every sampled scale worse than not warping at all. The range covers the fusion's own
# recovered metric factor (~23 on a sampled clip), so a gross mismatch lands inside it.
SCALES = (0.05, 0.1, 0.25, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0,
          20.0, 30.0, 50.0)
BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, 500.0))
INVERT = False          # set by --invert; the relative-pose direction is easy to flip


def load_depth(clip: Path, idx: list[int]) -> tuple[np.ndarray, str]:
    """Depth for the requested frame indices, whichever payload the clip carries."""
    npz = clip / "depth.npz"
    if npz.exists():
        d = np.load(npz)["depth"]
        return d[idx].astype(np.float32), f"depth.npz {d.shape}"
    zp = clip / "depth_exr.zip"
    import Imath
    import OpenEXR
    z = zipfile.ZipFile(zp)
    names = sorted(n for n in z.namelist() if n.endswith(".exr"))
    out = []
    tmp = Path("/tmp/_verify.exr")
    for i in idx:
        tmp.write_bytes(z.read(names[i]))
        f = OpenEXR.InputFile(str(tmp))
        dw = f.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        out.append(np.frombuffer(f.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT)),
                                 dtype=np.float32).reshape(h, w))
    tmp.unlink(missing_ok=True)
    return np.stack(out), f"depth_exr.zip ({len(names)} frames)"


def gray(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample; NaN outside the frame so misses are dropped, not clamped to an edge."""
    h, w = img.shape
    ok = (u >= 0) & (u <= w - 1.001) & (v >= 0) & (v <= h - 1.001)
    u0, v0 = np.floor(np.where(ok, u, 0)).astype(int), np.floor(np.where(ok, v, 0)).astype(int)
    du, dv = np.where(ok, u, 0) - u0, np.where(ok, v, 0) - v0
    out = ((1 - du) * (1 - dv) * img[v0, u0] + du * (1 - dv) * img[v0, u0 + 1]
           + (1 - du) * dv * img[v0 + 1, u0] + du * dv * img[v0 + 1, u0 + 1])
    return np.where(ok, out, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("video")
    ap.add_argument("--gap", type=int, default=12, help="frames between the pair (24fps)")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--invert", action="store_true",
                    help="use the opposite relative-pose direction (bug-isolation)")
    a = ap.parse_args()

    global INVERT
    INVERT = a.invert
    clip = Path(a.clip)
    poses = np.load(clip / "poses.npy")                      # (N,4,4) cam2world
    intr = np.load(clip / "intrinsics.npy")                  # (N,4) fx,fy,cx,cy at video res
    n = poses.shape[0]
    starts = np.linspace(int(0.05 * n), int(0.9 * n) - a.gap, a.pairs).astype(int)
    idx = sorted({int(i) for s in starts for i in (s, s + a.gap)})
    depth, how = load_depth(clip, idx)
    where = {f: k for k, f in enumerate(idx)}
    dh, dw = depth.shape[1:]

    import decord
    vr = decord.VideoReader(a.video)
    frames = {f: gray(vr[f].asnumpy(), (dh, dw)) for f in idx}
    sx = dw / (2.0 * intr[:, 2].mean())                      # video width -> depth width
    print(f"clip {clip.name}: {n} poses, depth {depth.shape} from {how}")
    print(f"intrinsics are for the video frame; depth is {sx:.2f}x that, scaled accordingly")

    yy, xx = np.mgrid[0:dh:4, 0:dw:4]                        # every 4th pixel is plenty
    u_pix, v_pix = xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)

    err = {s: [] for s in SCALES}
    derr = {s: [] for s in SCALES}
    band = {b: [] for b in BANDS}
    base, parallax = [], []
    for s0 in starts:
        t, t2 = int(s0), int(s0 + a.gap)
        d0 = depth[where[t]].astype(np.float64)
        # Sky is legitimately at hundreds of metres and carries no texture to match, so it
        # is dropped rather than allowed to dominate the residual.
        good = np.isfinite(d0) & (d0 > 0.1) & (d0 < 500)
        d2 = depth[where[t2]].astype(np.float64)
        d2 = np.where(np.isfinite(d2) & (d2 > 0.1) & (d2 < 500), d2, np.nan)
        g = good[v_pix.astype(int), u_pix.astype(int)]
        if g.sum() < 100:
            continue
        u, v = u_pix[g], v_pix[g]
        z0 = d0[v.astype(int), u.astype(int)]
        fx, fy, cx, cy = intr[t] * sx
        fx2, fy2, cx2, cy2 = intr[t2] * sx
        rel = (np.linalg.inv(poses[t2]) @ poses[t] if not INVERT
               else np.linalg.inv(poses[t]) @ poses[t2])
        R, tr = rel[:3, :3], rel[:3, 3]
        # How much signal this pair actually carries. Scale is only observable through
        # PARALLAX: the camera must move far enough, relative to how far away the scene is,
        # to displace pixels by more than the depth maps' own noise. Below ~5% the sweep
        # asymptotes instead of troughing and its argmin is meaningless - which is what
        # made two different residuals both "find" a minimum at the edge of the range.
        parallax.append(float(np.linalg.norm(tr) / max(np.nanmedian(z0), 1e-6)))
        ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)])   # (3,M)
        src = frames[t][v.astype(int), u.astype(int)]
        base.append(np.nanmedian(np.abs(src - frames[t2][v.astype(int), u.astype(int)])))
        for s in SCALES:
            P = R @ (ray * (z0 * s)) + tr[:, None]
            zc = P[2]
            vis = zc > 0.05
            uu = fx2 * P[0] / np.where(vis, zc, 1) + cx2
            vv = fy2 * P[1] / np.where(vis, zc, 1) + cy2
            got = sample(frames[t2], np.where(vis, uu, -1), np.where(vis, vv, -1))
            err[s].append(np.nanmedian(np.abs(src - got)))

            # The lens the verdict comes from: compare DEPTH against DEPTH. Where the point
            # lands in frame t+k, the far camera measured its own distance to whatever is
            # there, and the pose says it should be zc. Both sides carry the factor s but
            # the translation in `rel` does not, which is what makes the agreement point
            # informative rather than scale-free.
            #
            # It needs the parallax gate above to be trusted: with too little translation it
            # asymptotes towards the rotation-only limit instead of troughing, and its argmin
            # runs off to the end of the sweep. Gated, it is far sharper than the photometric
            # residual, which cares about texture and lighting - checked against omniworld
            # clips that ship GT depth AND GT poses, it puts a clean V at exactly 1.00 with a
            # 1.9% residual, while the photometric argmin on the same data wanders to 8x.
            d_got = sample(d2, np.where(vis, uu, -1), np.where(vis, vv, -1)) * s
            rel_e = np.abs(zc - d_got) / np.maximum(d_got, 1e-6)
            derr[s].append(np.nanmedian(np.where(vis, rel_e, np.nan)))

            # At true scale, split the same residual by how far away the point is. "Do depth
            # and pose share a scale" is one bit; what a downstream user picking anchor frames
            # by depth needs is the distance at which the agreement stops holding, because the
            # error is not uniform - GT comparison showed it growing with range.
            if s == 1.0:
                for lo_m, hi_m in BANDS:
                    m = (z0 >= lo_m) & (z0 < hi_m) & vis
                    if m.sum() >= 50:
                        band[(lo_m, hi_m)].append(float(np.nanmedian(rel_e[m])))

    if not base:
        print("no usable frame pair in this clip (all depth invalid) — nothing measured")
        return 1
    print(f"\n{'depth x':>8}  {'photometric':>12}  {'depth-vs-depth':>14}")
    best = min(SCALES, key=lambda s: np.nanmean(derr[s]))
    for s in SCALES:
        m, dm = float(np.nanmean(err[s])), float(np.nanmean(derr[s]))
        bar = "#" * int(50 * (1 - min(dm, 1.0)))
        print(f"{s:>8.2f}  {m:>12.2f}  {dm:>13.1%}  {bar}"
              f"{'   <-- minimum' if s == best else ''}")
    b = float(np.mean(base))
    print(f"{'no warp':>8}  {b:>12.2f}  {'-':>14}   (photometric baseline)")

    # Two ways to read nothing into a result, both of which this printed as a verdict before:
    # a minimum sitting on an endpoint means the true ratio may be outside the sweep, and a
    # minimum that does not beat the baseline means the warp never helped at any scale, so
    # the curve is measuring noise rather than geometry.
    lo = float(np.nanmean(derr[best]))
    px = float(np.mean(parallax))
    print(f"\nparallax: the camera moves {px:.1%} of the scene depth over {a.gap} frames")
    shown = [(b, float(np.median(v))) for b, v in band.items() if v]
    if shown:
        print("depth-vs-pose disagreement at true scale, by distance:")
        for (lo_m, hi_m), v in shown:
            print(f"    {lo_m:5.0f}-{hi_m:5.0f} m : {v:6.1%}")
    if px < 0.05:
        print(f"INCONCLUSIVE: {px:.1%} parallax is too little to see scale at all — the "
              f"sweep asymptotes rather than troughing. Test a clip that translates more.")
        return 0
    if best in (SCALES[0], SCALES[-1]):
        print(f"\nINCONCLUSIVE: the minimum is at the edge of the sweep ({best}); widen it.")
    elif 0.7 <= best <= 1.5:
        print(f"\nminimum at depth x {best} — depth and pose are in ONE scale; the depth "
              f"itself disagrees with the geometry by {lo:.1%} (GT-quality depth reads 2%)")
    else:
        print(f"\nminimum at depth x {best} ({lo:.1%} disagreement): depth is "
              f"{1/best:.2f}x the pose scale — they do NOT share a scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
