"""L0 Yerebakan descriptor (1372-D) + pose-invariant mask radiomics (14 scalars)."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class MaskFeats:
    log_volume: float
    mean_hu: float
    sphericity: float
    hu_std: float
    hu_p10: float
    hu_p50: float
    hu_p90: float
    hu_min: float
    hu_max: float
    bbox_e0_mm: float
    bbox_e1_mm: float
    bbox_e2_mm: float
    pca_l1_mm: float
    pca_l2_mm: float


def mask_stats(mask: np.ndarray, lesion_id: int, spacing: np.ndarray, ct: np.ndarray) -> MaskFeats:
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
    pts = np.argwhere(bin3).astype(np.float64)
    mn, mx = pts.min(0), pts.max(0)
    ext_raw = (mx - mn + 1.0) * np.array([dz, dy, dx], dtype=np.float64)
    exts = sorted([float(ext_raw[0]), float(ext_raw[1]), float(ext_raw[2])], reverse=True)
    p1 = p2 = 0.0
    if len(pts) >= 5:
        ctr = pts.mean(axis=0)
        coords = (pts - ctr) * np.array([dz, dy, dx], dtype=np.float64)
        c = np.cov(coords.T)
        w = np.linalg.eigh(c)[0]
        w = np.sort(w)[::-1]
        p1 = float(np.sqrt(max(float(w[0]), 0.0)))
        p2 = float(np.sqrt(max(float(w[1]), 0.0)))
    return MaskFeats(
        log_volume=float(np.log1p(vol)),
        mean_hu=float(hu.mean()),
        sphericity=float(sph),
        hu_std=float(hu.std()),
        hu_p10=float(np.percentile(hu, 10)),
        hu_p50=float(np.percentile(hu, 50)),
        hu_p90=float(np.percentile(hu, 90)),
        hu_min=float(hu.min()),
        hu_max=float(hu.max()),
        bbox_e0_mm=exts[0],
        bbox_e1_mm=exts[1],
        bbox_e2_mm=exts[2],
        pca_l1_mm=p1,
        pca_l2_mm=p2,
    )
