"""Real UniMatch (GMFlow) optical-flow magnitude over a clip.

Loads the GMFlow scale1 checkpoint and computes mean flow magnitude (pixels)
over consecutive sampled frames. The UniMatch repo must be importable (added to
sys.path) and the checkpoint present. Constructor/forward hyperparameters match
the scale1 mixdata checkpoint (verified against the repo).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np

_ROOT = os.environ.get(
    "SOLAR_WM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_REPO = os.path.join(_ROOT, "third_party", "unimatch")
_CKPT = os.path.join(_ROOT, "weights", "unimatch", "gmflow-scale1.pth")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _model():
    import torch
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from unimatch.unimatch import UniMatch
    m = UniMatch(num_scales=1, feature_channels=128, upsample_factor=8,
                 num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
                 reg_refine=False, task="flow").to(_device()).eval()
    ck = torch.load(_CKPT, map_location="cpu")
    m.load_state_dict(ck["model"])
    return m


def _pad_to(x, m):
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    return F.pad(x, (0, pw, 0, ph)), (h, w)


def mean_flow_magnitude(frames: np.ndarray) -> float:
    """Mean optical-flow magnitude (px) over consecutive frames, real GMFlow."""
    import torch
    model = _model()
    dev = _device()
    fwd = dict(attn_type="swin", attn_splits_list=[2], corr_radius_list=[-1],
               prop_radius_list=[-1], num_reg_refine=1)
    mags = []
    with torch.no_grad():
        for a, b in zip(frames[:-1], frames[1:]):
            i0 = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].to(dev)
            i1 = torch.from_numpy(b.astype(np.float32)).permute(2, 0, 1)[None].to(dev)
            i0p, (h, w) = _pad_to(i0, 32)
            i1p, _ = _pad_to(i1, 32)
            out = model(i0p, i1p, task="flow", **fwd)
            flow = out["flow_preds"][-1][..., :h, :w]  # (1,2,H,W)
            mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
            mags.append(float(mag.mean().item()))
    return float(np.mean(mags)) if mags else 0.0
