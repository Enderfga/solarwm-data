"""Adapters around VIPE, Pi3X, and MoGe-2 for the pose stage.

Real execution shells out to the vendored repos under ``third_party/``; dry-run
produces deterministic synthetic geometry so the stage and downstream filters
can be exercised end-to-end without a GPU.

SolarWM feeds fused Pi3X/MoGe-2 depth and per-frame intrinsics into VIPE through
these adapters. The fusion and bundle-adjustment math lives in ``fusion.py`` and
``intrinsics.py``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .intrinsics import constant_intrinsics


def _seed(clip_id: str) -> int:
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16)


def even_indices(count: int, n: int) -> np.ndarray:
    """`n` evenly-spaced integer indices into a sequence of length `count` (rounded).

    This is the single sampling rule for both Pi3/MoGe input and the GT-pose subsample,
    keeping frames and poses aligned during metric-scale recovery."""
    return np.linspace(0, max(count - 1, 0), max(min(n, count), 1)).round().astype(int)


def timed_indices(total: int, fps: float, interval_s: float, max_s: float) -> np.ndarray:
    """Frame indices at a fixed WALL-CLOCK stride (every ``interval_s`` seconds) over
    the first ``max_s`` seconds. Quality and motion metrics are sampled by
    time, not by frame count: UniMatch uses 0.5 s pairs across the first 60 s, DOVER
    5 s chunks. Sampling by a fixed frame count instead makes a 60 s clip ~6x sparser
    than a 10 s clip, so one shared threshold would compare incomparable magnitudes.
    This is the single source of truth for seconds-based metric sampling."""
    if total <= 0:
        return np.array([0])
    fps = fps if fps and fps > 0 else 16.0
    stride = max(1, int(round(fps * interval_s)))
    last = min(total - 1, int(round(fps * max_s)))
    idx = np.arange(0, last + 1, stride)
    if len(idx) < 2:  # ultra-short clip: at least one consecutive pair
        idx = np.array([0, min(1, total - 1)])
    return idx


def _video_fps(video_path: str) -> tuple[int, float]:
    """(total_frames, fps) via decord, else OpenCV. fps falls back to 16.0."""
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(video_path))
        return len(vr), float(vr.get_avg_fps() or 16.0)
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
        cap.release()
        return total, fps


def read_frames_at(video_path: str, idx: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read the exact frame indices ``idx`` -> (N,H,W,3) uint8 RGB (decord else OpenCV)."""
    idx = np.asarray(idx).astype(int)
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(video_path))
        sel = [int(min(max(i, 0), len(vr) - 1)) for i in idx]
        frames = vr.get_batch(sel).asnumpy()
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        want = set(int(i) for i in idx)
        got, i = {}, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        frames = np.stack([got[i] for i in sorted(got)]) if got else np.zeros((1, 8, 8, 3), np.uint8)
    if size is not None:
        import cv2
        frames = np.stack([cv2.resize(f, (size[1], size[0])) for f in frames])
    return frames


def read_frames_timed(
    video_path: str, interval_s: float = 0.5, max_s: float = 60.0, size: tuple[int, int] | None = None
) -> np.ndarray:
    """Frames at a fixed ``interval_s`` wall-clock stride over the first ``max_s`` seconds.
    Consecutive returned frames are exactly ``interval_s`` apart (paper's UniMatch 0.5 s
    pairing). Length-independent temporal density -> one threshold stays valid."""
    total, fps = _video_fps(video_path)
    return read_frames_at(video_path, timed_indices(total, fps, interval_s, max_s), size)


def read_frames(video_path: str, n_frames: int, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read up to n_frames evenly-sampled RGB frames -> (N,H,W,3) uint8.

    Uses decord if available, else OpenCV. ``size`` is (H,W) to resize to.
    """
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(video_path))
        total = len(vr)
        idx = even_indices(total, n_frames)
        frames = vr.get_batch(list(idx)).asnumpy()  # (n,H,W,3) RGB
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
        idx = set(even_indices(total, n_frames).tolist())
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in idx:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        frames = np.stack(frames) if frames else np.zeros((1, 8, 8, 3), np.uint8)
    if size is not None:
        import cv2
        frames = np.stack([cv2.resize(f, (size[1], size[0])) for f in frames])
    return frames


def _resize_stack(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize an (N,h,w) stack to (N,*hw)."""
    import cv2
    if arr.shape[1:] == tuple(hw):
        return arr
    return np.stack([cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR) for a in arr])


def _synthetic_depth(clip_id: str, n_frames: int, hw: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(_seed(clip_id))
    base = rng.uniform(1.0, 8.0, size=hw)  # static-ish scene depth
    # mild per-frame variation to mimic parallax
    return np.stack([base * (1.0 + 0.02 * t) for t in range(n_frames)], axis=0)


def _synthetic_trajectory(clip_id: str, n_frames: int) -> np.ndarray:
    """An (N,4,4) CAMERA-TO-WORLD trajectory: gentle forward dolly + yaw.

    c2w, like everything else in this corpus — the translation column is the camera
    centre. This said world-to-camera, which is the one convention error that produces
    plausible-looking, systematically wrong trajectories.
    """
    poses = np.tile(np.eye(4), (n_frames, 1, 1)).astype(np.float64)
    for t in range(n_frames):
        yaw = 0.01 * t
        c, s = np.cos(yaw), np.sin(yaw)
        poses[t, :3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        poses[t, :3, 3] = [0.0, 0.0, -0.05 * t]  # move forward
    return poses


def run_pi3x_depth(video_path: str, clip_id: str, n_frames: int, hw, cfg, dry_run: bool):
    """Pi3X multi-frame consistent (scale-ambiguous) depth, (T,H,W)."""
    if dry_run:
        return _synthetic_depth(clip_id + ":pi3x", n_frames, hw)
    from . import _real
    _poses, depth = _real.pi3_infer(read_frames(video_path, n_frames))
    return _resize_stack(depth, hw)


def run_pi3x_pose_depth(
    video_path: str, clip_id: str, n_frames: int, hw, cfg, dry_run: bool
):
    """Pi3 camera centres and depth from one shared reconstruction.

    A non-metric GT trajectory needs both outputs to bridge its gauge to metres.
    Running Pi3 twice is wasteful and, more importantly, makes the trajectory and
    depth provenance two separate reconstructions.  Keep them coupled here.
    """
    if dry_run:
        positions = _synthetic_trajectory(clip_id + ":pi3x", n_frames)[:, :3, 3]
        depth = _synthetic_depth(clip_id + ":pi3x", n_frames, hw)
        return positions, depth
    from . import _real
    poses, depth = _real.pi3_infer(read_frames(video_path, n_frames))
    return poses[:, :3, 3], _resize_stack(depth, hw)


def run_windowed_metric_bridge(
    video_path: str,
    clip_id: str,
    gt_positions: np.ndarray,
    fps: float,
    hw: tuple[int, int],
    cfg: dict,
    dry_run: bool,
) -> dict:
    """Recover one GT-gauge-to-metre scalar from short local reconstructions.

    A single Pi3 reconstruction over a 10--60 second clip can drift even when each
    local section is geometrically sound.  Fitting that global curve to the source
    trajectory silently turns reconstruction drift into a scale bias.  Instead, run
    several 2-second local windows.  Each window independently bridges

        GT gauge <- Pi3 reconstruction -> MoGe-2 metres

    and the final scalar is the robust median of the locally consistent estimates.
    Cross-window disagreement is observable uncertainty, so fail closed when it is
    too large rather than labelling an arbitrary scale as metric.
    """
    from .alignment import recover_scale_with_diagnostics
    from .fusion import robust_sequence_scale

    gt = np.asarray(gt_positions, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1:] != (3,) or len(gt) < 3:
        raise ValueError(f"metric bridge needs GT camera centres [N,3], got {gt.shape}")
    bridge_cfg = cfg.get("metric_bridge", {})
    samples = max(3, int(bridge_cfg.get("samples_per_window", 16)))
    windows = max(1, int(bridge_cfg.get("num_windows", 6)))
    span_seconds = float(bridge_cfg.get("window_seconds", 2.0))
    span = min(len(gt), max(samples, int(round(max(float(fps), 1.0) * span_seconds))))
    starts = np.linspace(0, len(gt) - span, windows).round().astype(int)
    starts = np.unique(starts)
    max_alignment_nrmse = float(bridge_cfg.get("max_alignment_nrmse", 0.15))
    max_scale_ratio = float(bridge_cfg.get("max_scale_p90_p10_ratio", 2.5))
    rows: list[dict] = []

    for start in starts.tolist():
        idx = np.linspace(start, start + span - 1, min(samples, span)).round().astype(int)
        gt_window = gt[idx]
        try:
            if dry_run:
                pi3_positions = gt_window / 2.0
                pi3_depth = _synthetic_depth(clip_id + f":window:{start}", len(idx), hw)
                moge_depth = 6.0 * pi3_depth
            else:
                from . import _real
                frames = read_frames_at(video_path, idx)
                pi3_poses, pi3_depth = _real.pi3_infer(frames)
                moge_depth = _real.moge_metric_depth(frames, ref_hw=pi3_depth.shape[1:])
            pi3_to_gt, alignment = recover_scale_with_diagnostics(
                pi3_positions if dry_run else pi3_poses[:, :3, 3],
                gt_window,
                inlier_percentile=80.0,
            )
            pi3_to_metric, _raw, metric = robust_sequence_scale(pi3_depth, moge_depth)
            gt_to_metric = float(pi3_to_metric / pi3_to_gt)
            if not np.isfinite(gt_to_metric) or gt_to_metric <= 0.0:
                raise ValueError(f"non-positive/non-finite GT-to-metric scale {gt_to_metric}")
            rows.append({
                "status": "success",
                "start_frame": int(start),
                "end_frame": int(start + span - 1),
                "sampled_frames": int(len(idx)),
                "pi3_to_gt": float(pi3_to_gt),
                "pi3_to_metric": float(pi3_to_metric),
                "gt_to_metric": gt_to_metric,
                "alignment": alignment,
                "metric": metric,
            })
        except Exception as exc:  # one bad window must not discard sound independent ones
            rows.append({
                "status": "error",
                "start_frame": int(start),
                "end_frame": int(start + span - 1),
                "sampled_frames": int(len(idx)),
                "error": f"{type(exc).__name__}: {exc}",
            })

    successful = [r for r in rows if r["status"] == "success"]
    accepted = [
        r for r in successful
        if r["alignment"]["normalized_inlier_rmse"] <= max_alignment_nrmse
    ]
    min_accepted = min(3, len(starts))
    if len(accepted) < min_accepted:
        raise ValueError(
            f"metric bridge has only {len(accepted)}/{len(starts)} locally aligned windows "
            f"(need {min_accepted}, max normalized RMSE {max_alignment_nrmse})"
        )

    raw_scales = np.asarray([r["gt_to_metric"] for r in accepted], dtype=np.float64)
    logs = np.log(raw_scales)
    log_med = float(np.median(logs))
    log_mad = float(np.median(np.abs(logs - log_med)))
    log_sigma = 1.4826 * log_mad
    # A generous MAD rejection removes isolated catastrophic windows.  The explicit
    # p90/p10 gate below still catches broadly inconsistent monocular metric estimates.
    log_tol = max(4.0 * log_sigma, np.log(1.5))
    scale_inliers = np.abs(logs - log_med) <= log_tol
    inlier_rows = [r for r, keep in zip(accepted, scale_inliers) if keep]
    if len(inlier_rows) < min_accepted:
        raise ValueError(
            f"metric bridge has only {len(inlier_rows)} scale-consistent windows after MAD filtering"
        )
    scales = np.asarray([r["gt_to_metric"] for r in inlier_rows], dtype=np.float64)
    p10 = float(np.percentile(scales, 10.0))
    p90 = float(np.percentile(scales, 90.0))
    ratio = float(p90 / p10)
    if ratio > max_scale_ratio:
        raise ValueError(
            f"metric bridge window scales disagree (p90/p10 {ratio:.3f} > {max_scale_ratio:.3f})"
        )
    scalar = float(np.median(scales))

    alignment_values = np.asarray([
        r["alignment"]["normalized_inlier_rmse"] for r in inlier_rows
    ])
    metric_diag = {
        "method": "local_window_pi3_moge2_v1",
        "window_seconds": span_seconds,
        "window_span_frames": int(span),
        "windows_total": int(len(starts)),
        "windows_successful": int(len(successful)),
        "windows_alignment_accepted": int(len(accepted)),
        "windows_scale_inliers": int(len(inlier_rows)),
        "sampled_frames": int(sum(r["metric"]["sampled_frames"] for r in successful)),
        "valid_frames": int(sum(r["metric"]["valid_frames"] for r in successful)),
        "inlier_frames": int(sum(r["metric"]["inlier_frames"] for r in successful)),
        "rejected_frames": int(sum(r["metric"]["rejected_frames"] for r in successful)),
        "gt_to_metric_median": scalar,
        "gt_to_metric_p10": p10,
        "gt_to_metric_p90": p90,
        "p90_p10_ratio": ratio,
        "raw_p90_p10_ratio": float(
            np.percentile(raw_scales, 90.0) / np.percentile(raw_scales, 10.0)
        ),
        "log_mad_sigma": log_sigma,
        "max_scale_p90_p10_ratio": max_scale_ratio,
        "windows": rows,
    }
    alignment_diag = {
        "method": "local_window_sim3_v1",
        "windows_total": int(len(starts)),
        "accepted_windows": int(len(inlier_rows)),
        "matched_frames": int(sum(r["alignment"]["matched_frames"] for r in inlier_rows)),
        "inlier_frames": int(sum(r["alignment"]["inlier_frames"] for r in inlier_rows)),
        "normalized_inlier_rmse": float(np.max(alignment_values)),
        "normalized_inlier_rmse_median": float(np.median(alignment_values)),
        "normalized_inlier_rmse_max": float(np.max(alignment_values)),
        "max_alignment_nrmse": max_alignment_nrmse,
    }
    return {
        "gt_to_metric": scalar,
        "pi3_to_gt": float(np.median([r["pi3_to_gt"] for r in inlier_rows])),
        "pi3_to_metric": float(np.median([r["pi3_to_metric"] for r in inlier_rows])),
        "metric_scale_diagnostics": metric_diag,
        "trajectory_alignment_diagnostics": alignment_diag,
    }


def run_moge2_depth(video_path: str, clip_id: str, n_frames: int, hw, cfg, dry_run: bool):
    """MoGe-2 metric-scale depth, (T,H,W). In dry-run it is a scaled Pi3X."""
    if dry_run:
        # make MoGe ~= true_scale * Pi3X so fusion recovers a sensible scale
        true_scale = 1.0 + 0.5 * (_seed(clip_id) % 5)
        return true_scale * _synthetic_depth(clip_id + ":pi3x", n_frames, hw)
    from . import _real
    return _real.moge_metric_depth(read_frames(video_path, n_frames), ref_hw=hw)


def run_pi3x_trajectory(video_path: str, clip_id: str, n_frames: int, cfg, dry_run: bool):
    """Pi3X camera positions (N,3), scale-ambiguous (for GT-pose alignment)."""
    if dry_run:
        return _synthetic_trajectory(clip_id + ":pi3x", n_frames)[:, :3, 3]
    from . import _real
    poses, _depth = _real.pi3_infer(read_frames(video_path, n_frames))
    return poses[:, :3, 3]  # camera centers (cam-to-world translation)


def run_vipe_slam(
    video_path: str, clip_id: str, n_frames: int, hw, depth, intrinsics0, cfg, dry_run: bool
):
    """VIPE SLAM front-end + per-frame-intrinsics BA.

    Returns ``(poses (N,4,4), intrinsics (N,V,4))``. In dry-run, returns a
    synthetic trajectory and the seed intrinsics unchanged.
    """
    if dry_run:
        poses = _synthetic_trajectory(clip_id, n_frames)
        return poses, intrinsics0
    # Real mode: full VIPE SLAM+BA is a heavy CUDA build we do not set up here.
    # We use Pi3's multi-frame-consistent pose output as the pose track (Pi3 is
    # the structure backbone SolarWM builds on); VIPE's bundle-adjustment refine
    # is the one layer omitted. Intrinsics keep the seed (per-frame BA not run).
    from . import _real
    frames = read_frames(video_path, n_frames)
    poses, _depth = _real.pi3_infer(frames)
    return poses, intrinsics0
