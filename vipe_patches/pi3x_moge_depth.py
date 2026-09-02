# SPDX-License-Identifier: Apache-2.0
"""SolarWM modified depth backend for VIPE: Pi3X + MoGe-2 fused metric depth.

Replaces VIPE's default depth (Metric3D-Small) with the
Pi3X (multi-frame consistent) + MoGe-2 (per-frame metric anchor) fusion.

**Decoupled design (deliberate):** Pi3X/MoGe-2 need torch 2.5/cu124 while VIPE
needs torch 2.7/cu128 (CUDA extensions are ABI-locked) — they can't share a
process. So the fusion is precomputed OFFLINE in the torch-2.5 env
(``scripts/precompute_fused_depth.py``) and written to a dir holding
``fused.npy`` (S,h,w) + ``sig.npy`` (S,64) 8x8-gray signatures of the sampled
frames. This backend LOADS that dir and, per VIPE ``estimate`` call, matches the
incoming keyframe RGB to its fused depth by signature (VIPE's SLAM does not pass
a frame index, so content matching is the robust key).

Dir via ``SOLAR_WM_FUSED_DEPTH_DIR``. Deploy: copy to
``vipe/priors/depth/pi3x_moge.py`` + register ``pi3xmoge`` in make_depth_model.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from vipe.utils.cameras import CameraType

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType

_DEPTH_DIR = os.environ.get("SOLAR_WM_FUSED_DEPTH_DIR", "")


class Pi3xMogeModel(DepthEstimationModel):
    """Loads precomputed Pi3X+MoGe-2 fused metric depth, matched by signature."""

    def __init__(self) -> None:
        super().__init__()
        if not _DEPTH_DIR or not os.path.isdir(_DEPTH_DIR):
            raise RuntimeError(
                "SOLAR_WM_FUSED_DEPTH_DIR must point at a precompute dir "
                "(fused.npy + sig.npy); see scripts/precompute_fused_depth.py"
            )
        self._fused = np.load(os.path.join(_DEPTH_DIR, "fused.npy"))  # (S,h,w)
        self._sig = np.load(os.path.join(_DEPTH_DIR, "sig.npy"))      # (S,64)

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH

    @property
    def supported_camera_types(self):
        return [CameraType.PINHOLE]

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        assert src.camera_type == CameraType.PINHOLE
        rgb = _rgb_hwc(src.rgb)                       # (H,W,3) float/uint8
        h, w = rgb.shape[:2]
        si = self._match(rgb)
        depth = self._fused[si].astype(np.float32)    # (h0,w0)
        depth = _resize(depth, (h, w))
        # VIPE expects metric_depth shaped (V, H, W) (it does disp[:, 3::8, 3::8])
        return DepthEstimationResult(metric_depth=torch.from_numpy(depth)[None].float())

    def _match(self, rgb_hwc: np.ndarray) -> int:
        # Match the incoming keyframe to its precomputed fused depth by nearest signature.
        # RGB 16x16 (768-d) — identical to precompute_fused_depth.py — so near-duplicate
        # frames stay separable and BA never gets a wrong-frame depth (8x8 gray collided).
        import cv2
        g = rgb_hwc
        if g.dtype != np.uint8:
            g = (np.clip(g, 0, 1) * 255).astype(np.uint8) if g.max() <= 1.0 else g.astype(np.uint8)
        sig = cv2.resize(g, (16, 16)).astype(np.float32).ravel()
        d = np.linalg.norm(self._sig - sig[None], axis=1)
        return int(np.argmin(d))


def _rgb_hwc(rgb) -> np.ndarray:
    a = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    a = np.squeeze(a)                       # drop singleton view/batch dims
    if a.ndim == 3 and a.shape[0] == 3:     # (C,H,W) -> (H,W,C)
        a = np.transpose(a, (1, 2, 0))
    return a


def _resize(d: np.ndarray, hw):
    import cv2
    if d.shape == tuple(hw):
        return d
    return cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
