#!/usr/bin/env python3
"""Validate per-frame intrinsics BA: compare VIPE's recovered per-frame focal
against the exact GT focal ramp of the synthetic zoom clip.

VIPE optimises intrinsics per KEYFRAME; the GT is per input frame. For a steadily
moving clip the keyframes are ~evenly spaced, so we resample the GT focal to the
number of recovered entries (linspace) and compare. The decisive signal: recovered
focal must be monotonically increasing and span ~the GT range (a per-VIEW baseline
would be ~constant).

    python3 compare_intrinsics.py <recovered.npy> <gt_intrinsics.npy>
"""
import sys

import numpy as np

rec = np.load(sys.argv[1])   # (K,4) fx,fy,cx,cy  (recovered, original-res)
gt = np.load(sys.argv[2])    # (N,4) fx,fy,cx,cy  (exact)

rec_f = rec[:, 0]
gt_f = gt[:, 0]
K, N = len(rec_f), len(gt_f)

# resample GT focal to the K keyframe positions (linspace over the clip)
idx = np.linspace(0, N - 1, K).round().astype(int)
gt_at_kf = gt_f[idx]

print(f"recovered keyframes K={K}, GT frames N={N}")
print(f"GT focal ramp:        {gt_f[0]:.1f} -> {gt_f[-1]:.1f}  (x{gt_f[-1]/gt_f[0]:.2f})")
print(f"recovered focal:      {rec_f.min():.1f} -> {rec_f.max():.1f}  (x{rec_f.max()/max(rec_f.min(),1e-6):.2f})")
print()
print("kf |   GT_f  | recovered_f | abs_err | rel_err")
for k in range(K):
    e = rec_f[k] - gt_at_kf[k]
    print(f"{k:2d} | {gt_at_kf[k]:7.1f} | {rec_f[k]:11.1f} | {e:+7.1f} | {e/gt_at_kf[k]*100:+5.1f}%")

# verdicts
diffs = np.diff(rec_f)
monotonic = bool(np.all(diffs > -5))      # allow tiny noise
rec_range = rec_f.max() - rec_f.min()
gt_range = gt_f.max() - gt_f.min()
range_ratio = rec_range / gt_range
mae = float(np.mean(np.abs(rec_f - gt_at_kf)))
mape = float(np.mean(np.abs(rec_f - gt_at_kf) / gt_at_kf)) * 100

print()
print(f"monotonic increasing: {monotonic}")
print(f"recovered range / GT range: {range_ratio:.2f}  (per-view baseline would be ~0)")
print(f"MAE={mae:.1f}px  MAPE={mape:.1f}%")
tracks = monotonic and range_ratio > 0.6 and mape < 15
print()
print("VERDICT:", "PASS — per-frame BA tracks the zoom" if tracks
      else "CHECK — recovered focal does not cleanly track GT (see table)")
