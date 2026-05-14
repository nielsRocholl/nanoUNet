"""Propagation-offset for simulated longitudinal COG error."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def apply_propagation_offset(
    centroid_zyx: Tuple[int, int, int],
    patch_shape: Tuple[int, int, int],
    sigma_per_axis: Tuple[float, float, float],
    max_vox: float,
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    cz, cy, cx = centroid_zyx
    dz, dy, dx = rng.normal(0, sigma_per_axis)
    mag = float(np.sqrt(dz * dz + dy * dy + dx * dx))
    if mag > max_vox and mag > 1e-8:
        s = max_vox / mag
        dz, dy, dx = dz * s, dy * s, dx * s
    pz = int(round(cz + dz))
    py = int(round(cy + dy))
    px = int(round(cx + dx))
    d, h, w = patch_shape
    return (max(0, min(d - 1, pz)), max(0, min(h - 1, py)), max(0, min(w - 1, px)))
