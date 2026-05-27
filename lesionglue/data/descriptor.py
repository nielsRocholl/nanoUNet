"""Yerebakan L0 (1372-D) and full 5-level hierarchy (6860-D) HU descriptors."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from tracking.data.features import L0_DIM, YEREBAKAN_LEVELS

SCALES_L0_MM = (8.0, 20.0, 48.0, 128.0)
GRID = 7


def _offsets_for_scales(scales: tuple[float, ...]) -> np.ndarray:
    out = []
    for d in scales:
        for i in range(GRID):
            for j in range(GRID):
                for k in range(GRID):
                    out.append(((i - 3) * d, (j - 3) * d, (k - 3) * d))
    return np.asarray(out, dtype=np.float64)


OFFSETS_L0 = _offsets_for_scales(SCALES_L0_MM)
assert OFFSETS_L0.shape == (L0_DIM, 3)


def _sample(vol: np.ndarray, affine: np.ndarray, center_ijk: np.ndarray, offsets_mm: np.ndarray) -> np.ndarray:
    R, t = affine[:3, :3], affine[:3, 3]
    c = np.asarray(center_ijk, dtype=np.float64).reshape(3)
    samps_w = offsets_mm + (R @ c + t)
    inv = np.linalg.inv(affine)
    ijk = (inv[:3, :3] @ samps_w.T).T + inv[:3, 3]
    return map_coordinates(vol, ijk.T, order=1, mode="constant", cval=0.0).astype(np.float32)


def descriptor_l0(vol: np.ndarray, affine: np.ndarray, center_ijk: np.ndarray) -> np.ndarray:
    return _sample(vol, affine, center_ijk, OFFSETS_L0)


def descriptor_yerebakan(vol: np.ndarray, affine: np.ndarray, center_ijk: np.ndarray) -> np.ndarray:
    parts = []
    for level in range(YEREBAKAN_LEVELS):
        sc = tuple(s / (2.0**level) for s in SCALES_L0_MM)
        parts.append(_sample(vol, affine, center_ijk, _offsets_for_scales(sc)))
    out = np.concatenate(parts)
    assert out.shape == (L0_DIM * YEREBAKAN_LEVELS,)
    return out
