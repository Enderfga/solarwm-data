"""Faithful real-VIPE pose backend (the modified engine built by setup_vipe.sh).

``stage.annotate_pose`` covers all three modes, but its ``default``/``gt_depth``
real path uses a Pi3 *stand-in* (``adapters.run_vipe_slam`` returns Pi3 poses and
the seed intrinsics — "VIPE's bundle-adjustment refine is the one layer
omitted"). For a high-fidelity reproduction those two modes must run the REAL
modified VIPE engine: Pi3X+MoGe-2 fused depth (mod #1) + per-frame intrinsics BA
(mod #2). VIPE needs its own CUDA-built venv, so it is invoked via its CLI.

This module shells out to that CLI and writes the SAME ClipRecord fields
``annotate_pose`` does — ``pose_path`` (N,4,4 cam2world float64), ``intrinsics_path``
(N,4 fx,fy,cx,cy float64), ``scale_factors``, ``pose_mode`` — so the downstream
filter / caption / package stages are byte-for-byte identical regardless of which
backend produced the pose. ``gt_pose`` sources keep ``stage.annotate_pose`` (its
gt_pose path is already faithful: real Pi3 + real GT trajectory + Umeyama scale).

Two-process handoff (deliberate, see vipe_patches/pi3x_moge_depth.py): Pi3/MoGe
run in the torch-2.8 "real" env, VIPE runs in ``.venv-vipe``. ``precompute`` (real
env) writes the fused depth dir; ``vipe infer`` (vipe env) consumes it. This
function is called from the real-env fleet worker and spawns both as subprocesses
so a CUDA fault in either is isolated to one clip.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..manifest import ClipRecord


def _vipe_cfg(models_cfg: dict) -> dict:
    """Resolve VIPE backend paths (models_cfg['vipe'] overrides env overrides default)."""
    v = dict(models_cfg.get("vipe") or {})
    wm = v.get("wm_root") or os.environ.get("SOLAR_WM_ROOT", "")
    return {
        "wm_root": wm,
        "vipe_bin": v.get("vipe_bin") or os.environ.get("SOLAR_WM_VIPE_BIN") or f"{wm}/.venv-vipe/bin/vipe",
        "weights": v.get("weights") or os.environ.get("SOLAR_WM_WEIGHTS") or f"{wm}/weights",
        "gpu": str(v.get("gpu", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))),
        "max_frames": int(v.get("max_frames", os.environ.get("SOLAR_WM_MAX_FRAMES", "64"))),
    }


def _real_env(cfg: dict) -> dict:
    """Env for the torch-2.8 real-env subprocesses (precompute): Pi3 on PYTHONPATH."""
    wm = cfg["wm_root"]
    # APPEND to the caller's PYTHONPATH rather than replacing it: on shared-filesystem
    # deployments the pipeline's extra site-packages (decord, moge, ...) live on the
    # shared volume and are on PYTHONPATH, not in the interpreter's own site-packages.
    # Overwriting hid them from the subprocess and every clip died on ModuleNotFoundError.
    _pp = [f"{wm}/third_party/Pi3", wm]
    if os.environ.get("PYTHONPATH"):
        _pp.append(os.environ["PYTHONPATH"])
    return dict(
        os.environ,
        SOLAR_WM_ROOT=wm,
        SOLAR_WM_WEIGHTS=cfg["weights"],
        HF_HOME=f"{cfg['weights']}/hf",
        PYTHONPATH=os.pathsep.join(_pp),
        CUDA_VISIBLE_DEVICES=cfg["gpu"],
        SOLAR_WM_MAX_FRAMES=str(cfg["max_frames"]),
    )


def _load_vipe_pose(out_dir: Path) -> np.ndarray:
    """Load VIPE's cam2world pose track (N,4,4), frame-ordered by `inds`."""
    npzs = sorted(Path(out_dir).glob("pose/*.npz"))
    if not npzs:
        raise FileNotFoundError(f"no VIPE pose npz under {out_dir}/pose/")
    z = np.load(npzs[0])
    data = np.asarray(z["data"], dtype=np.float64)        # (N,4,4) cam2world (OpenCV)
    inds = np.asarray(z["inds"]).ravel()
    order = np.argsort(inds)                               # ensure frame order
    return data[order]


def _load_perframe_intrinsics(pf_dump: Path, n_target: int) -> np.ndarray:
    """Per-frame intrinsics (N,4) [fx,fy,cx,cy] for the N pose frames.

    mod #2 dumps the BA-optimised per-keyframe intrinsics (K,4). The per-frame
    intrinsics optimisation exists precisely to capture VARYING focal length / non-square
    pixels on internet video (zoom etc.), so we must NOT collapse to a single constant:
    a constant track erases the per-frame focal the downstream camera conditioning
    (Plücker / UCPE) is supposed to see. We keep the per-frame signal:
      * K == N  -> use the dump as-is (already per-frame);
      * K == 1  -> broadcast the single intrinsic;
      * 1<K<N   -> the dump has no keyframe->frame map, so assume keyframes are spread
                   across the clip and linearly interpolate each of (fx,fy,cx,cy) to the
                   N frames. This preserves the focal trend/range (a zoom stays a ramp)
                   instead of flattening it; for a fixed-lens clip it reduces to the
                   constant case. (A future VIPE-patch dump of keyframe indices would
                   make the frame mapping exact.)
    """
    pf = np.load(pf_dump).astype(np.float64)               # (K,4)
    if pf.ndim == 1:
        pf = pf[None, :]
    pf = _sanitize_focal(pf)                                # repair BA-diverged keyframe focals first
    K = pf.shape[0]
    if K == n_target:
        return pf
    if K == 1:
        return np.tile(pf[0], (n_target, 1))               # (N,4)
    src = np.linspace(0.0, 1.0, K)
    dst = np.linspace(0.0, 1.0, n_target)
    return np.stack([np.interp(dst, src, pf[:, j]) for j in range(pf.shape[1])], axis=1)  # (N,4)


def _sanitize_focal(pf: np.ndarray) -> np.ndarray:
    """Repair non-physical focal lengths in the BA keyframe dump (pf is (K,4) [fx,fy,cx,cy]).

    Per-frame-intrinsics BA on hard internet-video frames occasionally diverges, emitting a
    non-positive or wildly out-of-range focal on some keyframes (observed fx from -6998 to a
    spurious spike) while cx,cy stay sane. ``np.interp`` does not extrapolate, so a bad keyframe
    is faithfully carried into the per-frame track -> negative/garbage focal in the emitted
    intrinsics (validate_clip flags it; the assembler's FOV/focal gate then rejects the whole
    clip — wasted compute). Focal must be positive: rebuild each bad keyframe's fx/fy from the
    GOOD keyframes of the same clip (in-band linear interp), preserving a genuine zoom ramp while
    dropping the divergence spikes. If a column has no usable focal at all, fall back to a ~58deg
    FoV guess from the (reliable) image centre: fx ~= 0.9 * width = 0.9 * 2*cx. cx,cy untouched."""
    pf = pf.copy()
    K = pf.shape[0]
    x = np.arange(K, dtype=np.float64)
    for j, c_col in ((0, 2), (1, 3)):                      # fx<-cx, fy<-cy
        v = pf[:, j]
        pos = v[v > 0]
        if pos.size == 0:                                  # no usable focal -> image-centre fallback
            pf[:, j] = 0.9 * 2.0 * np.maximum(pf[:, c_col], 1.0)
            continue
        med = np.median(pos)
        ok = (v > 0.2 * med) & (v < 5.0 * med)             # positive AND within the clip's own focal band
        if ok.all():
            continue
        if ok.any():
            pf[~ok, j] = np.interp(x[~ok], x[ok], v[ok])   # rebuild bad keyframes from good ones
        else:
            pf[:, j] = 0.9 * 2.0 * np.maximum(pf[:, c_col], 1.0)
    return pf


def annotate_pose_vipe_cli(rec: ClipRecord, out_dir, models_cfg: dict) -> ClipRecord:
    """Faithful real-VIPE pose for `default` (and, later, `gt_depth`) clips.

    Writes ``<clip>.poses.npy`` / ``<clip>.intrinsics.npy`` under ``out_dir`` and
    sets the same ClipRecord fields ``stage.annotate_pose`` does.
    """
    if rec.mode == "gt_pose":
        raise ValueError("gt_pose is handled by stage.annotate_pose, not the VIPE CLI backend")
    if rec.mode == "gt_depth":
        # A faithful gt_depth route would feed GT depth as VIPE's keyframe depth plus
        # MoGe metric scale. OmniWorld-Game ships only
        # GT camera poses+intrinsics, RGB, masks and captions in <uid>_others.tar.gz /
        # <uid>_rgb_*.tar.gz — NO depth maps. So OmniWorld is routed to `gt_pose`, which
        # IS faithful to the GT it actually provides (GT trajectory + Pi3 structure +
        # Umeyama metric scale). This guard stays so a full run can never silently fall
        # back to the default (predicted Pi3+MoGe) recipe. Wire this only if a GT-depth
        # release becomes available; the backend would then mirror the default path with
        # precompute_fused_depth replaced by the GT depth loader.
        raise NotImplementedError(
            "gt_depth needs GT depth maps; OmniWorld-Game ships GT poses only -> use gt_pose"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _vipe_cfg(models_cfg)
    if not cfg["wm_root"]:
        raise RuntimeError("SOLAR_WM_ROOT (or models_cfg['vipe']['wm_root']) required for VIPE CLI")
    env = _real_env(cfg)
    video = rec.video_path
    work = out_dir / rec.clip_id
    depth_dir = work / "depth"
    vipe_out = work / "vipe_out"
    pf_dump = work / "intr_pf.npy"
    work.mkdir(parents=True, exist_ok=True)

    # 1. Pi3X+MoGe-2 fused metric depth (real env) -> fused.npy + sig.npy + scales.npy
    subprocess.run(
        [sys.executable, f"{cfg['wm_root']}/scripts/precompute_fused_depth.py", video, str(depth_dir)],
        check=True, env=env,
    )

    # 2. real modified VIPE SLAM + per-frame-intrinsics BA (vipe venv)
    vipe_env = dict(
        env,
        SOLAR_WM_FUSED_DEPTH_DIR=str(depth_dir),
        SOLAR_WM_PF_DUMP=str(pf_dump),
        # vipe venv inherits container torch; keep its own site-packages clean of Pi3
        PYTHONPATH=f"{cfg['wm_root']}",
    )
    subprocess.run(
        [cfg["vipe_bin"], "infer", video, "-o", str(vipe_out), "-p", "solarwm"],
        check=True, env=vipe_env,
    )

    # 3. convert VIPE outputs -> the uniform corpus format
    poses = _load_vipe_pose(vipe_out)                      # (N,4,4) c2w
    n = poses.shape[0]
    intr = _load_perframe_intrinsics(pf_dump, n)           # (N,4)
    scales_npy = depth_dir / "scales.npy"
    # [] when no scales were recovered, NEVER [1.0]: scale_cov([1.0]) is 0.0, which reads
    # as a perfectly stable metric scale and passes the gate that exists to catch exactly
    # the case where no scale was recovered at all. scale_cov([]) is inf and rejects.
    scales = np.load(scales_npy).astype(np.float64).tolist() if scales_npy.exists() else []

    pose_path = out_dir / f"{rec.clip_id}.poses.npy"
    intr_path = out_dir / f"{rec.clip_id}.intrinsics.npy"
    np.save(pose_path, poses)
    np.save(intr_path, intr)
    rec.pose_path = str(pose_path.resolve())
    rec.intrinsics_path = str(intr_path.resolve())
    rec.scale_factors = [float(x) for x in scales]
    rec.pose_mode = rec.mode
    rec.extra["pose_backend"] = "vipe_cli"
    return rec
