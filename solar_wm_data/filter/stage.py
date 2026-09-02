"""Filter stage: compute metrics, apply the camera and per-dataset gates, set the tier.

Reads the per-frame intrinsics and scale factors produced by the pose stage to
evaluate the camera filters, computes the visual/motion/VLM metrics (real or
dry-run), then marks each clip ``kept`` / ``reject_reasons``.
"""

from __future__ import annotations

import numpy as np

from ..manifest import ClipRecord, CameraMetrics
from .camera import scale_cov
from .metrics import compute_metrics
from .thresholds import judge_clip

_EPS = 1e-8


def _load_intrinsics(rec: ClipRecord) -> np.ndarray | None:
    """Full per-frame intrinsics (N, 4): fx, fy, cx, cy, from intrinsics.npy."""
    if not rec.intrinsics_path:
        return None
    K = np.load(rec.intrinsics_path)
    if K.ndim == 1:
        K = K[None, :]
    return K


def _load_poses(rec: ClipRecord) -> np.ndarray | None:
    """Camera-to-world poses (N, 4, 4) from poses.npy; None if the stage produced none.

    ``None`` means no camera evidence and causes a rejection. Read errors from an
    existing file propagate because corrupt data is a defect rather than an absence.
    """
    if not rec.pose_path:
        return None
    return np.load(rec.pose_path)


def filter_clip(rec: ClipRecord, filters_cfg: dict, models_cfg: dict) -> ClipRecord:
    reasons: list[str] = []

    # 1. metrics (visual / motion / VLM)
    ds_cfg = filters_cfg["datasets"].get(rec.source)
    if ds_cfg is None:
        raise KeyError(
            f"no thresholds for source '{rec.source}' in this filters config. A source "
            f"gets a row calibrated from its own measured distribution; do not copy the "
            f"nearest row or run it ungated. "
            f"Use configs/filters_released.yaml if you want the published policy, or run "
            f"scripts/calibrate_filters.py on this source first.")
    rec.metrics = compute_metrics(rec, models_cfg)
    tier, q_reasons = judge_clip(rec, ds_cfg)
    reasons += q_reasons

    # 2. camera filters (need intrinsics + per-frame scale from pose stage).
    #    Gate FOV / focal divergence PER FRAME (worst frame), not on the median,
    #    so zoom / non-square clips with out-of-range end-frames are caught
    #    Scale-CoV rejects clips with no metric scale recovered.
    # A dataset may narrow or widen the global camera block for itself (one source needs
    # a 125-degree FOV ceiling). Merge rather than replace, so a per-source override
    # naming only fov_deg still inherits the focal-divergence and scale-CoV bounds.
    cam_cfg = {**filters_cfg["camera"], **(ds_cfg.get("camera") or {})}
    # 1b. FINITENESS. Non-finite poses or intrinsics are unusable for camera
    #     conditioning and are rejected explicitly.
    # Missing is not passing. An absent array means the gate was never evaluated, and an
    # unevaluated gate rejects — the same rule the metric thresholds follow.
    for _name, _arr in (("poses", _load_poses(rec)), ("intrinsics", _load_intrinsics(rec))):
        if _arr is None:
            reasons.append(f"{_name} missing (camera evidence unevaluable)")
        elif not np.isfinite(_arr).all():
            reasons.append(f"{_name} contain NaN/Inf")

    # Fail closed on missing camera evidence. A clip with unevaluable camera gates
    # has not passed them and is rejected.
    K = _load_intrinsics(rec)
    if K is None or not rec.width or not rec.height:
        missing = ("intrinsics" if K is None else
                   f"frame size ({rec.width}x{rec.height})")
        reasons.append(f"camera gates unevaluable: {missing} missing")
    else:
        fx_all = K[:, 0].astype(np.float64)
        fy_all = K[:, 1].astype(np.float64)
        fov_x_all = np.degrees(2.0 * np.arctan(rec.width / (2.0 * fx_all)))
        fov_y_all = np.degrees(2.0 * np.arctan(rec.height / (2.0 * fy_all)))
        fdiv_all = np.abs(fx_all - fy_all) / ((fx_all + fy_all) / 2.0 + _EPS)

        fov_lo, fov_hi = cam_cfg["fov_deg"]
        if fov_x_all.min() < fov_lo or fov_x_all.max() > fov_hi:
            reasons.append(
                f"fov_x in [{fov_x_all.min():.1f},{fov_x_all.max():.1f}]deg outside [{fov_lo},{fov_hi}]"
            )
        if fov_y_all.min() < fov_lo or fov_y_all.max() > fov_hi:
            reasons.append(
                f"fov_y in [{fov_y_all.min():.1f},{fov_y_all.max():.1f}]deg outside [{fov_lo},{fov_hi}]"
            )
        if float(fdiv_all.max()) > cam_cfg["focal_div_max"]:
            reasons.append(f"focal_div max={fdiv_all.max():.3f} > {cam_cfg['focal_div_max']}")

        scales = rec.scale_factors or []
        cov = scale_cov(scales)  # inf when no scale recovered -> rejects
        if cov > cam_cfg["scale_cov_max"]:
            reasons.append(f"scale_cov={cov:.3f} > {cam_cfg['scale_cov_max']}")

        # Store the extremes ALONGSIDE the medians. The gates above compared the
        # extremes; storing only the medians meant no later re-judgement could reach the
        # same verdict, and the assembler -- which decides what actually trains --
        # silently re-admitted clips this stage had rejected on a single bad frame.
        rec.camera = CameraMetrics(
            fov_x=float(np.median(fov_x_all)), fov_y=float(np.median(fov_y_all)),
            focal_div=float(np.median(fdiv_all)),
            scale_cov=(float(cov) if scales else None),
            fov_x_min=float(fov_x_all.min()), fov_x_max=float(fov_x_all.max()),
            fov_y_min=float(fov_y_all.min()), fov_y_max=float(fov_y_all.max()),
            focal_div_max=float(fdiv_all.max()),
        )

    rec.reject_reasons = reasons
    rec.kept = len(reasons) == 0
    # A camera failure rejects outright: the xhigh extras never relax a camera gate, so a
    # clip that failed one is not "high", it is out.
    rec.kept_tier = tier if rec.kept else None
    return rec


def run_filter_stage(
    records: list[ClipRecord], filters_cfg: dict, models_cfg: dict
) -> list[ClipRecord]:
    return [filter_clip(r, filters_cfg, models_cfg) for r in records]
