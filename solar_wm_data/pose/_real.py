"""Real Pi3 + MoGe-2 inference backends for the pose stage.

Loaded lazily and cached. Pi3 gives multi-frame-consistent structure + cam poses
(scale-ambiguous); MoGe-2 gives metric-scale depth. Weights are read from the
repo ``weights/`` dir (downloaded by scripts/setup_real_env.sh) if present, else
pulled from HuggingFace.

Shape contract (the part that bites): Pi3 consumes ``(B,N,3,H,W)`` in [0,1] and
returns ``camera_poses (B,N,4,4)`` cam-to-world (OpenCV) and ``local_points
(B,N,H,W,3)`` (depth = z channel). MoGe-2 consumes ``(3,H,W)`` in [0,1] per
frame and returns metric ``depth (H,W)`` + normalized ``intrinsics (3,3)``.
Pi3 and MoGe operate at different resolutions, so depth maps are resized to a
common grid before fusion.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_WEIGHTS = Path(os.environ.get(
    "SOLAR_WM_WEIGHTS", str(Path(__file__).resolve().parents[2] / "weights")))


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast_dtype():
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


@lru_cache(maxsize=1)
def _pi3():
    import torch
    from pi3.models.pi3 import Pi3
    local = _WEIGHTS / "pi3"
    src = str(local) if local.exists() else "yyfz233/Pi3"
    model = Pi3.from_pretrained(src).to(_device()).eval()
    return model


@lru_cache(maxsize=1)
def _moge():
    from moge.model.v2 import MoGeModel
    local = _WEIGHTS / "moge2"
    # MoGe-2 from_pretrained wants a checkpoint FILE (or HF repo id), not a dir.
    src = "Ruicheng/moge-2-vitl-normal"
    if local.is_dir():
        cands = (list(local.glob("*.pt")) + list(local.glob("*.pth"))
                 + list(local.glob("*.safetensors")))
        if cands:
            src = str(cands[0])
    elif local.exists():
        src = str(local)
    return MoGeModel.from_pretrained(src).to(_device()).eval()


_PI3_PATCH = 14  # DINOv2 patch size; Pi3 requires H, W multiples of this
# Pi3/DUSt3R/VGGT-style dense models are designed for a roughly 512-pixel long side.
# Cap larger sources before patch rounding to bound global-attention memory. For
# ``gt_pose`` sources Pi3 supplies only the Umeyama scale and a coarse depth track.
_PI3_MAX_SIDE = int(os.environ.get("SOLAR_WM_PI3_MAX_SIDE", "518"))


def _round_to_patch(frames: np.ndarray) -> np.ndarray:
    """Resize (N,H,W,3) so the long side <= SOLAR_WM_PI3_MAX_SIDE and H,W are
    multiples of the DINOv2 patch size."""
    import cv2
    frames = np.asarray(frames)
    oh, ow = frames.shape[1:3]
    h, w = oh, ow
    long_side = max(h, w)
    if _PI3_MAX_SIDE and long_side > _PI3_MAX_SIDE:
        s = _PI3_MAX_SIDE / float(long_side)
        h, w = int(round(h * s)), int(round(w * s))
    nh = max(_PI3_PATCH, (h // _PI3_PATCH) * _PI3_PATCH)
    nw = max(_PI3_PATCH, (w // _PI3_PATCH) * _PI3_PATCH)
    if (nh, nw) == (oh, ow):
        return frames
    return np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames])


def _frames_to_pi3_tensor(frames: np.ndarray):
    """(N,H,W,3) uint8 RGB -> (N,3,H,W) float [0,1] on device, H/W multiple of 14."""
    import torch
    frames = _round_to_patch(np.asarray(frames))
    arr = frames.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(_device())


def pi3_infer(frames: np.ndarray):
    """Run Pi3 on a clip. Returns (poses (N,4,4) cam2world, depth (N,h,w))."""
    import torch
    model = _pi3()
    imgs = _frames_to_pi3_tensor(frames)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=_autocast_dtype()):
            res = model(imgs[None])  # (1,N,3,H,W)
    poses = res["camera_poses"][0].float().cpu().numpy()        # (N,4,4) cam2world
    local = res["local_points"][0].float().cpu().numpy()         # (N,h,w,3)
    depth = local[..., 2]                                        # (N,h,w)
    return poses, depth


def moge_metric_depth(frames: np.ndarray, ref_hw: tuple[int, int] | None = None):
    """Per-frame MoGe-2 metric depth, optionally resized to ref_hw (Pi3 grid).

    MoGe publishes a validity mask alongside depth.  Invalid/sky pixels are not a
    metric observation and must not participate in the Pi3-to-metre fit.  Preserve
    that fact as NaN after resizing; the scale solver already ignores non-finite
    samples.
    """
    import torch
    model = _moge()
    out = []
    for f in frames:
        t = torch.from_numpy(np.asarray(f, np.float32) / 255.0).permute(2, 0, 1).to(_device())
        with torch.no_grad():
            result = model.infer(t)
            d = result["depth"].float().cpu().numpy()           # (H,W) metric metres
            mask = result["mask"].bool().cpu().numpy()
        if ref_hw is not None and d.shape != tuple(ref_hw):
            d = _resize_depth(d, ref_hw)
            mask = _resize_mask(mask, ref_hw)
        d = np.where(mask & np.isfinite(d) & (d > 0.0), d, np.nan)
        out.append(d)
    return np.stack(out)                                         # (N, *ref_hw)


def _resize_depth(d: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize a depth map to (h,w) with area/linear interpolation."""
    try:
        import cv2
        return cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    except Exception:
        import torch
        t = torch.from_numpy(d)[None, None]
        t = torch.nn.functional.interpolate(t, size=tuple(hw), mode="bilinear",
                                             align_corners=False)
        return t[0, 0].numpy()


def _resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize a validity mask to ``(h, w)`` without inventing valid pixels."""
    try:
        import cv2
        return cv2.resize(
            mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    except Exception:
        import torch
        t = torch.from_numpy(mask.astype(np.float32))[None, None]
        t = torch.nn.functional.interpolate(t, size=tuple(hw), mode="nearest")
        return (t[0, 0].numpy() > 0.5)
