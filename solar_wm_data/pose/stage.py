"""Pose stage: modified VIPE engine with three dataset-specific modes.

Modes:
* ``default``   internet video: Pi3X+MoGe-2 fused depth -> VIPE SLAM + per-frame BA.
* ``gt_depth``  metric GT depth in SLAM, with MoGe-2 scale validation.
* ``gt_pose``   keep the GT trajectory; use it directly when it is metric, otherwise
                bridge its gauge to MoGe-2 metres through Pi3X.

Outputs per clip: ``poses.npy`` (N,4,4), ``intrinsics.npy`` (N,4 per-frame), and
``scale_factors`` (per-frame metric scale, used later by the scale-CoV filter).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ingest import gt_pose_is_metric
from ..manifest import ClipRecord
from . import adapters
from .alignment import recover_scale_with_diagnostics
from .fusion import fuse_depth_sequence
from .intrinsics import constant_intrinsics, to_per_frame_2d


def _default_hw(rec: ClipRecord) -> tuple[int, int]:
    # small synthetic resolution in dry-run; real runs read the decoded frames
    h = rec.height or 72
    w = rec.width or 128
    return (min(h, 72), min(w, 128))


def _seed_intrinsics(rec: ClipRecord, n_frames: int) -> np.ndarray:
    w = rec.width or 128
    h = rec.height or 72
    fx = fy = 0.9 * w  # ~ moderate FoV starting guess
    return constant_intrinsics(fx, fy, w / 2.0, h / 2.0, n_frames)


def annotate_pose(
    rec: ClipRecord, out_dir: str | Path, models_cfg: dict
) -> ClipRecord:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dry = models_cfg.get("dry_run", True)
    fusion_cfg = models_cfg.get("depth_fusion", {})
    momentum = fusion_cfg.get("ema_momentum", 0.99)

    import os
    # OUTPUT poses MUST be frame-aligned to EVERY video frame. The max_frames cap applies
    # ONLY to the subsample fed to Pi3/MoGe (GPU memory) — it must NEVER truncate the
    # output (a 64-cap desynced pose_i from video frame_i and produced unusable clips).
    N = rec.num_frames or 24
    max_frames = int(os.environ.get("SOLAR_WM_MAX_FRAMES", "64"))
    n_scale = min(N, max_frames)
    hw = _default_hw(rec)
    mode = rec.mode

    if mode == "gt_pose":
        # GT trajectory stays at FULL length (N frames). Metric-GT sources need one
        # ordinary Pi3 subsample for validation. Non-metric sources use several short
        # Pi3+MoGe windows to recover one robust scalar without long-trajectory drift.
        #
        # WHICH DIRECTION THAT SCALE GOES matters. `recover_metric_scale(pred, gt)` returns
        # the factor taking Pi3 units to GT units: the Umeyama fit brings Pi3's structure
        # INTO the trajectory's gauge. The trajectory is the reference, not the thing being
        # corrected -- which is sound exactly when the trajectory is already metric, and is
        # why `gt_pose_is_metric` has to be answered per source. Pi3 is scale-ambiguous, so
        # aligning it to a COLMAP-scale trajectory yields the ratio of two arbitrary scales
        # and metricises nothing.
        metric_gt = gt_pose_is_metric(rec.source)   # raises if the source never declared
        gt_poses = _load_gt_poses(rec, N)           # FULL-length GT poses, aligned to video
        pred_pos = None
        if metric_gt or gt_poses is None:
            pred_pos = adapters.run_pi3x_trajectory(
                rec.video_path, rec.clip_id, n_scale, models_cfg, dry
            )
            n_scale = pred_pos.shape[0]             # Pi3 may return fewer
        if gt_poses is None:
            if not dry:
                raise FileNotFoundError(
                    f"gt_pose clip {rec.clip_id!r} has no ground-truth poses on disk. The "
                    f"proxy trajectory below exists only so the dry run has something "
                    f"shaped like a trajectory; emitting it for real would ship a "
                    f"fabricated camera path that passes every structural check.")
            # Dry-run proxy at n_scale.
            gt_centers = 1.7 * np.asarray(pred_pos)
            s, alignment_diag = recover_scale_with_diagnostics(
                pred_pos, gt_centers, inlier_percentile=80.0
            )
            poses = _positions_to_poses(gt_centers)
            gt_to_metric = 1.0
            pi3_to_metric = float(s)
            pi3_to_metric_raw = np.full(len(pred_pos), pi3_to_metric, dtype=np.float64)
            metric_diag = {
                "sampled_frames": int(len(pred_pos)),
                "valid_frames": int(len(pred_pos)),
                "inlier_frames": int(len(pred_pos)),
                "pi3_to_metric_median": pi3_to_metric,
                "log_mad_sigma": 0.0,
                "p10": pi3_to_metric,
                "p90": pi3_to_metric,
                "p90_p10_ratio": 1.0,
            }
            gauge = "dry-run-proxy"
        elif metric_gt:
            gt_sub = gt_poses[adapters.even_indices(gt_poses.shape[0], n_scale)]  # match Pi3 frames
            s, alignment_diag = recover_scale_with_diagnostics(
                pred_pos, gt_sub[:, :3, 3], inlier_percentile=80.0
            )
            poses = gt_poses
            gt_to_metric = 1.0
            pi3_to_metric = float(s)
            gauge = "native"
        else:
            # Long global Pi3 reconstructions drift on 10--60 second Sekai clips.  Fit
            # several local reconstructions and require their independently metricised
            # GT-scale estimates to agree before claiming metres.
            bridge = adapters.run_windowed_metric_bridge(
                rec.video_path,
                rec.clip_id,
                gt_poses[:, :3, 3],
                rec.fps or 16.0,
                hw,
                models_cfg,
                dry,
            )
            gt_to_metric = float(bridge["gt_to_metric"])
            pi3_to_metric = float(bridge["pi3_to_metric"])
            s = float(bridge["pi3_to_gt"])
            metric_diag = bridge["metric_scale_diagnostics"]
            alignment_diag = bridge["trajectory_alignment_diagnostics"]
            poses = gt_poses.copy()
            poses[:, :3, 3] *= gt_to_metric         # rotations are scale-free
            gauge = "moge2-windowed"
        m = poses.shape[0]
        intr = _load_real_intrinsics(rec, m)
        if intr is None:
            intr = _seed_intrinsics(rec, m)
        # ``scale_factors`` is documented as Pi3-to-metre scale. For a non-metric
        # trajectory, ``s`` is only Pi3-to-GT-gauge; retain that ratio in provenance but
        # publish the actual Pi3-to-metre factor used by the metric bridge.
        if metric_gt:
            scales = [pi3_to_metric] * m
        else:
            # This is the one scalar actually applied to the preserved GT trajectory.
            # Per-window Pi3 gauges are independent, so their Pi3-to-metre factors must
            # not be interpolated into a fictitious per-frame series.
            scales = [gt_to_metric] * m
            rec.extra["metric_scale_diagnostics"] = metric_diag
            rec.extra["trajectory_alignment_diagnostics"] = alignment_diag
        rec.extra["gt_to_metric"] = gt_to_metric
        rec.extra["gt_pose_metric_source"] = gauge
        rec.extra["pi3_to_gt"] = float(s)
        rec.extra["pi3_to_metric"] = float(pi3_to_metric)

    elif mode == "gt_depth":
        # GT depth in SLAM; MoGe-2 recovers metric scale via point-cloud align.
        n = n_scale
        intr0 = _load_real_intrinsics(rec, n)
        if intr0 is None:
            intr0 = _seed_intrinsics(rec, n)
        gt_depth = _load_gt_depth(rec, n, hw, dry)
        moge = adapters.run_moge2_depth(rec.video_path, rec.clip_id, n, hw, models_cfg, dry)
        _, scales_arr = fuse_depth_sequence(gt_depth, moge, ema_momentum=momentum)
        poses, intr = adapters.run_vipe_slam(
            rec.video_path, rec.clip_id, n, hw, gt_depth, intr0, models_cfg, dry
        )
        scales = scales_arr.tolist()

    else:  # default
        n = n_scale
        intr0 = _load_real_intrinsics(rec, n)
        if intr0 is None:
            intr0 = _seed_intrinsics(rec, n)
        pi3x = adapters.run_pi3x_depth(rec.video_path, rec.clip_id, n, hw, models_cfg, dry)
        moge = adapters.run_moge2_depth(rec.video_path, rec.clip_id, n, hw, models_cfg, dry)
        fused, scales_arr = fuse_depth_sequence(pi3x, moge, ema_momentum=momentum)
        poses, intr = adapters.run_vipe_slam(
            rec.video_path, rec.clip_id, n, hw, fused, intr0, models_cfg, dry
        )
        scales = scales_arr.tolist()

    pose_arr = np.asarray(poses, dtype=np.float64)
    intr_arr = to_per_frame_2d(np.asarray(intr, dtype=np.float64))
    # Reject non-finite solver output even when its shape and file size are valid.
    for name, arr in (("poses", pose_arr), ("intrinsics", intr_arr)):
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).any(axis=tuple(range(1, arr.ndim))).sum())
            raise ValueError(
                f"{rec.clip_id}: {mode} produced non-finite {name} "
                f"({n_bad}/{len(arr)} frames) — pose solve diverged")
    np.save(pose_path := out_dir / f"{rec.clip_id}.poses.npy", pose_arr)
    np.save(intr_path := out_dir / f"{rec.clip_id}.intrinsics.npy", intr_arr)

    rec.pose_path = str(pose_path.resolve())
    rec.intrinsics_path = str(intr_path.resolve())
    rec.scale_factors = [float(x) for x in scales]
    rec.pose_mode = mode
    # Every path that reaches here emits metres: `default` and `gt_depth` because the
    # depth handed to SLAM was metricised against MoGe-2, `gt_pose` because the trajectory
    # was either metric already or was rescaled above. Stated on the record so a consumer
    # never has to infer it, and so the validator can check the claim rather than trust it.
    rec.pose_units = "metric"
    return rec


def _positions_to_poses(positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    poses = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    poses[:, :3, 3] = positions
    return poses


def _load_real_intrinsics(rec: ClipRecord, n: int) -> np.ndarray | None:
    """Load the clip's real per-frame intrinsics (N,4) if ingest captured them,
    frame-aligned to n. Returns (n,1,4) tensor or None."""
    p = rec.extra.get("gt_intrinsics_path")
    if not (p and Path(p).exists()):
        return None
    K = np.load(p).astype(np.float64)
    if K.ndim == 1:
        K = K[None, :]
    if K.shape[-1] != 4:  # only (fx,fy,cx,cy) layout supported here
        return None
    idx = np.linspace(0, K.shape[0] - 1, n).round().astype(int)
    return K[idx][:, None, :]  # (n,1,4)


def _load_gt_poses(rec: ClipRecord, n: int) -> np.ndarray | None:
    """Load GT poses as (n,4,4), frame-aligned to Pi3's n sampled frames.

    Accepts (M,4,4), (M,3,4), or (M,3) positions on disk. Returns None if no GT
    poses are stored (dry-run / pure-prediction sources).
    """
    p = rec.extra.get("gt_positions_path")
    if not (p and Path(p).exists()):
        return None
    arr = np.load(p).astype(np.float64)
    m = arr.shape[0]
    poses = np.tile(np.eye(4), (m, 1, 1))
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        poses = arr
    elif arr.ndim == 3 and arr.shape[1:] == (3, 4):
        poses[:, :3, :4] = arr
    elif arr.ndim == 2 and arr.shape[1] == 3:
        poses[:, :3, 3] = arr
    else:
        raise ValueError(f"unrecognized GT pose array shape {arr.shape}")
    # frame-align: subsample GT to the same n frames Pi3 used. MUST use the same
    # sampling rule as adapters.read_frames or Pi3 frame i and GT pose i diverge.
    idx = adapters.even_indices(m, n)
    return poses[idx]


def _load_gt_depth(rec: ClipRecord, n: int, hw, dry: bool) -> np.ndarray:
    p = rec.extra.get("gt_depth_path")
    if p and Path(p).exists():
        return np.load(p)
    return adapters.run_pi3x_depth(rec.video_path, rec.clip_id, n, hw, {}, dry_run=True)


def run_pose_stage(
    records: list[ClipRecord], out_dir: str | Path, models_cfg: dict
) -> list[ClipRecord]:
    return [annotate_pose(r, out_dir, models_cfg) for r in records]
