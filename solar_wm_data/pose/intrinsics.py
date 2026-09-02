"""Per-frame intrinsics representation.

The original VIPE shares one set of intrinsics across all frames. SolarWM
extends bundle adjustment to treat ``(fx, fy, cx, cy)`` as independent variables
per frame, stored as an ``(N, V, D)`` tensor (frames x views x intrinsics-dim),
enabling calibration on internet video with non-square pixels and varying focal
length. For the monocular sources here ``V = 1`` and ``D = 4``.

This module owns the storage layout and small helpers; the actual per-frame BA
optimisation lives in the VIPE adapter (we patch VIPE's BA to expose this
tensor). Here we provide construction, validation, and the ``(N, 4)`` view that
the rest of the pipeline (filters, packaging) consumes.
"""

from __future__ import annotations

import numpy as np

INTRINSICS_DIM = 4  # fx, fy, cx, cy


def make_intrinsics_tensor(
    fx: np.ndarray, fy: np.ndarray, cx: np.ndarray, cy: np.ndarray, n_views: int = 1
) -> np.ndarray:
    """Build an (N, V, 4) per-frame intrinsics tensor from per-frame arrays."""
    fx, fy, cx, cy = (np.asarray(a, dtype=np.float64).ravel() for a in (fx, fy, cx, cy))
    n = fx.shape[0]
    if not (fy.shape[0] == cx.shape[0] == cy.shape[0] == n):
        raise ValueError("fx, fy, cx, cy must share length N")
    flat = np.stack([fx, fy, cx, cy], axis=-1)  # (N, 4)
    return np.repeat(flat[:, None, :], n_views, axis=1)  # (N, V, 4)


def constant_intrinsics(
    fx: float, fy: float, cx: float, cy: float, n_frames: int, n_views: int = 1
) -> np.ndarray:
    """Seed an (N, V, 4) tensor with shared intrinsics (BA initialisation)."""
    one = np.array([fx, fy, cx, cy], dtype=np.float64)
    return np.tile(one, (n_frames, n_views, 1))


def to_per_frame_2d(tensor: np.ndarray, view: int = 0) -> np.ndarray:
    """Reduce an (N, V, 4) tensor to the (N, 4) view used downstream."""
    t = np.asarray(tensor)
    if t.ndim == 2:  # already (N, 4)
        return t
    if t.ndim != 3 or t.shape[-1] != INTRINSICS_DIM:
        raise ValueError(f"expected (N, V, 4) tensor, got shape {t.shape}")
    return t[:, view, :]
