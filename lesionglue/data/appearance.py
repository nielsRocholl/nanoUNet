"""L0 Yerebakan descriptor (1372-D) + mask-derived log-volume, mean HU, sphericity."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

DESC_DIM = 1372
SCALES_MM = (8.0, 20.0, 48.0, 128.0)
GRID = 7


def _build_offsets() -> np.ndarray:
    out = []
    for d in SCALES_MM:
        for i in range(GRID):
            for j in range(GRID):
                for k in range(GRID):
                    out.append(((i - 3) * d, (j - 3) * d, (k - 3) * d))
    arr = np.asarray(out, dtype=np.float64)
    assert arr.shape == (DESC_DIM, 3)
    return arr


OFFSETS_MM = _build_offsets()


def descriptor_l0(vol: np.ndarray, affine: np.ndarray, center_ijk: np.ndarray) -> np.ndarray:
    R, t = affine[:3, :3], affine[:3, 3]
    c = np.asarray(center_ijk, dtype=np.float64).reshape(3)
    samps_w = OFFSETS_MM + (R @ c + t)
    inv = np.linalg.inv(affine)
    ijk = (inv[:3, :3] @ samps_w.T).T + inv[:3, 3]
    return map_coordinates(vol, ijk.T, order=1, mode="constant", cval=0.0).astype(np.float32)


def mask_stats(mask: np.ndarray, lesion_id: int, spacing: np.ndarray, ct: np.ndarray) -> tuple[float, float, float]:
    bin3 = mask == lesion_id
    if not bin3.any():
        raise ValueError(f"empty mask for lesion {lesion_id}")
    dz, dy, dx = spacing.astype(np.float64)
    n = int(bin3.sum())
    vol = float(n * dz * dy * dx)
    hu = ct[bin3].astype(np.float64)
    sa = (
        np.sum(bin3[1:] != bin3[:-1]) * dy * dx
        + np.sum(bin3[:, 1:] != bin3[:, :-1]) * dz * dx
        + np.sum(bin3[:, :, 1:] != bin3[:, :, :-1]) * dz * dy
    )
    sph = (36.0 * np.pi * vol * vol) ** (1.0 / 3.0) / max(sa, 1e-6)
    return float(np.log1p(vol)), float(hu.mean()), float(sph)
