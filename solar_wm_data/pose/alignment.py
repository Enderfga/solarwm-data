"""Similarity (Sim(3)) alignment between predicted and reference trajectories.

Pi3X predicts scene structure up to scale. A Umeyama Sim(3) fit recovers the
factor mapping its camera positions into a reference trajectory's gauge, using
80th-percentile inlier filtering for robustness. The returned factor is metric
only when the reference trajectory is already in metres; non-metric GT sources
must add a metric anchor such as MoGe-2 before claiming metric output.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def umeyama_sim3(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) mapping ``src -> dst`` (Umeyama 1991).

    Finds ``s, R, t`` minimising ``sum_k || s R src_k + t - dst_k ||^2``.
    Returns ``(s, R(3x3), t(3,))``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / (var_src + _EPS))
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def recover_metric_scale(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    inlier_percentile: float = 80.0,
) -> float:
    """Scale factor aligning predicted to reference camera positions.

    Two-pass: fit Sim(3) on all points, keep points whose residual is below the
    ``inlier_percentile`` percentile, then re-fit and return the scale. This
    rejects gross trajectory outliers from imperfect structure prediction.
    """
    scale, _ = recover_scale_with_diagnostics(
        pred_positions, gt_positions, inlier_percentile=inlier_percentile
    )
    return scale


def recover_scale_with_diagnostics(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    inlier_percentile: float = 80.0,
) -> tuple[float, dict[str, float | int]]:
    """Robust Pi3-to-reference scale plus dimensionless fit diagnostics."""
    pred = np.asarray(pred_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1:] != (3,) or pred.shape != gt.shape:
        raise ValueError(f"trajectory shape mismatch: Pi3 {pred.shape}, GT {gt.shape}")
    if len(pred) < 3 or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("trajectory alignment needs at least three finite matched positions")

    pred_rms = float(np.sqrt(np.mean(np.sum((pred - pred.mean(0)) ** 2, axis=1))))
    gt_rms = float(np.sqrt(np.mean(np.sum((gt - gt.mean(0)) ** 2, axis=1))))
    if pred_rms <= _EPS or gt_rms <= _EPS:
        raise ValueError(
            f"degenerate trajectory for scale recovery (Pi3 rms {pred_rms}, GT rms {gt_rms})"
        )

    s, R, t = umeyama_sim3(pred, gt)
    resid = np.linalg.norm((s * (R @ pred.T)).T + t - gt, axis=1)
    thresh = np.percentile(resid, inlier_percentile)
    inliers = resid <= thresh
    if inliers.sum() >= 3:
        s, R, t = umeyama_sim3(pred[inliers], gt[inliers])
    resid = np.linalg.norm((s * (R @ pred.T)).T + t - gt, axis=1)
    inlier_resid = resid[inliers]
    diagnostics: dict[str, float | int] = {
        "matched_frames": int(len(pred)),
        "inlier_frames": int(inliers.sum()),
        "pi3_center_rms": pred_rms,
        "gt_center_rms": gt_rms,
        "inlier_rmse": float(np.sqrt(np.mean(inlier_resid ** 2))),
        "normalized_inlier_rmse": float(np.sqrt(np.mean(inlier_resid ** 2)) / gt_rms),
        "all_p90_residual": float(np.percentile(resid, 90.0)),
    }
    return float(s), diagnostics
