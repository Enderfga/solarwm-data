"""Pi3X + MoGe-2 depth fusion.

Pi3X gives long-sequence-consistent (but scale-ambiguous) depth; MoGe-2 gives a
per-frame metric-scale anchor. We fuse them by solving for a per-frame scale
factor ``s`` minimising

    sum_i w_i (s * d_Pi3X_i - d_MoGe_i)^2 ,   w_i = 1 / d_i

with inverse-depth weights, then smoothing the per-frame scale temporally with
an exponential moving average (momentum 0.99). The fused depth for a frame is
``s_smoothed * d_Pi3X``: it keeps Pi3X's temporally-consistent structure while
adopting MoGe-2's metric scale.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def solve_frame_scale(d_pi3x: np.ndarray, d_moge: np.ndarray) -> float:
    """Weighted-least-squares scale for one frame.

    Minimises ``sum_i w_i (s*a_i - b_i)^2`` with ``a=d_pi3x``, ``b=d_moge`` and
    inverse-depth weights ``w_i = 1/b_i`` (the metric reference depth). The
    closed-form solution is ``s = sum(w a b) / sum(w a^2)``.
    """
    a = np.asarray(d_pi3x, dtype=np.float64).ravel()
    b = np.asarray(d_moge, dtype=np.float64).ravel()
    # only use valid, positive-depth pixels
    mask = np.isfinite(a) & np.isfinite(b) & (a > _EPS) & (b > _EPS)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan")
    w = 1.0 / (b + _EPS)
    num = np.sum(w * a * b)
    den = np.sum(w * a * a) + _EPS
    return float(num / den)


def robust_sequence_scale(
    d_pi3x: np.ndarray, d_moge: np.ndarray
) -> tuple[float, np.ndarray, dict[str, float | int]]:
    """Robust Pi3-to-metre scale over a shared multi-frame reconstruction.

    MoGe is monocular and its metric head can vary from frame to frame.  Translation
    needs one scalar for the whole GT trajectory, so estimate raw per-frame factors,
    reject log-scale outliers with a median/MAD rule, and use their median.  Return the
    raw factors as quality evidence rather than hiding disagreement behind the EMA used
    by the dense-depth path.
    """
    a = np.asarray(d_pi3x, dtype=np.float64)
    b = np.asarray(d_moge, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError(f"Pi3/MoGe depth shape mismatch: {a.shape} vs {b.shape}")
    pixel_valid = np.isfinite(a) & np.isfinite(b) & (a > _EPS) & (b > _EPS)
    pixel_fraction = pixel_valid.reshape((a.shape[0], -1)).mean(axis=1)
    raw = np.asarray([
        solve_frame_scale(a[t], b[t]) if pixel_fraction[t] >= 0.05 else float("nan")
        for t in range(a.shape[0])
    ])
    valid = np.isfinite(raw) & (raw > _EPS)
    min_valid = max(3, int(np.ceil(0.5 * len(raw))))
    if int(valid.sum()) < min_valid:
        raise ValueError(
            f"only {int(valid.sum())}/{len(raw)} frames have a valid Pi3/MoGe metric scale"
        )

    logs = np.log(raw[valid])
    med = float(np.median(logs))
    mad = float(np.median(np.abs(logs - med)))
    sigma = 1.4826 * mad
    tol = max(4.0 * sigma, 1e-6)
    valid_idx = np.flatnonzero(valid)
    inlier_local = np.abs(logs - med) <= tol
    inliers = np.zeros_like(valid)
    inliers[valid_idx[inlier_local]] = True
    if int(inliers.sum()) < 3:
        raise ValueError("fewer than three inlier frames remain in Pi3/MoGe scale fit")

    chosen = raw[inliers]
    raw_valid = raw[valid]
    scale = float(np.median(chosen))
    diagnostics: dict[str, float | int] = {
        "sampled_frames": int(len(raw)),
        "valid_frames": int(valid.sum()),
        "inlier_frames": int(inliers.sum()),
        "rejected_frames": int(len(raw) - inliers.sum()),
        "min_valid_pixel_fraction": float(np.min(pixel_fraction)),
        "median_valid_pixel_fraction": float(np.median(pixel_fraction)),
        "pi3_to_metric_median": scale,
        "log_mad_sigma": sigma,
        "p10": float(np.percentile(chosen, 10.0)),
        "p90": float(np.percentile(chosen, 90.0)),
        "p90_p10_ratio": float(np.percentile(chosen, 90.0) / np.percentile(chosen, 10.0)),
        "raw_p90_p10_ratio": float(
            np.percentile(raw_valid, 90.0) / np.percentile(raw_valid, 10.0)
        ),
    }
    filtered = raw.copy()
    filtered[~inliers] = np.nan
    return scale, filtered, diagnostics


def fuse_depth_sequence(
    d_pi3x: np.ndarray, d_moge: np.ndarray, ema_momentum: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse a (T, ...) depth sequence.

    Returns ``(fused_depth, scales)`` where ``scales`` is the EMA-smoothed
    per-frame scale (length T) and ``fused_depth[t] = scales[t] * d_pi3x[t]``.
    """
    d_pi3x = np.asarray(d_pi3x, dtype=np.float64)
    d_moge = np.asarray(d_moge, dtype=np.float64)
    T = d_pi3x.shape[0]

    scales = np.empty(T, dtype=np.float64)
    ema = None
    for t in range(T):
        s_raw = solve_frame_scale(d_pi3x[t], d_moge[t])
        if np.isfinite(s_raw) and s_raw > _EPS:
            ema = s_raw if ema is None else ema_momentum * ema + (1 - ema_momentum) * s_raw
        scales[t] = float("nan") if ema is None else ema

    fused = scales.reshape((T,) + (1,) * (d_pi3x.ndim - 1)) * d_pi3x
    return fused, scales
