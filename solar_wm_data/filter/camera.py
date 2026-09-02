"""Camera-specific filter quantities.

Scalar formula definitions plus `scale_cov`. The gates themselves are applied
in `filter/stage.py`, vectorised over frames, because they are per-frame gates:
a clip is rejected on its worst frame rather than on collapsed medians.

Applied uniformly across all datasets. Given frame resolution (W, H) and
intrinsics (fx, fy, cx, cy):

* horizontal/vertical field of view  theta = 2*arctan(dim / (2*f)), must lie in
  [25 deg, 120 deg];
* focal divergence  |fx - fy| / ((fx + fy) / 2), must be <= 0.20;
* metric-scale coefficient of variation  std(s_t) / (mean(s_t) + eps) over the
  per-frame scale factors, must be <= 2.0.

All pure CPU math, deterministic, no external deps beyond numpy.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_EPS = 1e-8


def fov_degrees(width: float, height: float, fx: float, fy: float) -> tuple[float, float]:
    """Horizontal and vertical field of view in degrees."""
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    fov_y = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return fov_x, fov_y


def focal_divergence(fx: float, fy: float) -> float:
    """Symmetric normalized focal mismatch |fx-fy| / ((fx+fy)/2)."""
    denom = (fx + fy) / 2.0
    return abs(fx - fy) / (denom + _EPS)


def scale_cov(scale_factors: Sequence[float]) -> float:
    """Coefficient of variation of per-frame metric scale factors.

    Returns ``inf`` for an empty sequence so the gate REJECTS clips with no
    metric scale recovered, rather than silently passing them (CoV 0 <= max).
    A genuinely missing scale means metric reconstruction failed.
    """
    s = np.asarray(list(scale_factors), dtype=np.float64)
    if s.size == 0:
        return float("inf")
    return float(np.std(s) / (np.mean(s) + _EPS))
