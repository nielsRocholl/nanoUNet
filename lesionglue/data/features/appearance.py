"""Pose-invariant mask radiomics (14 scalars per lesion).

Bbox from scipy.ndimage.find_objects, 1-voxel pad for surface area — same
values as a full-volume `mask == id` scan, ~70× faster on 50-lesion CTs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


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


def label_objects(mask: np.ndarray):
    return ndi.find_objects(mask.astype(np.int32, copy=False))


def _sl(objs, lid: int, shape: tuple[int, ...], pad: int = 1):
    if lid < 1 or lid > len(objs) or objs[lid - 1] is None:
        raise ValueError(
            f"empty mask for lesion {lid}\n"
            "Expected every lesion_id in the patient's meta CSV to be a label present in its mask NIfTI.\n"
            f"Fix: make the meta CSV lesion_id column and the mask labels agree (label {lid} is missing from the mask)"
        )
    sl = objs[lid - 1]
    if pad:
        sl = tuple(slice(max(0, s.start - pad), min(d, s.stop + pad)) for s, d in zip(sl, shape))
    return sl


def mask_stats(mask: np.ndarray, lesion_id: int, spacing: np.ndarray, ct: np.ndarray, *, objects=None) -> MaskFeats:
    objs = objects if objects is not None else label_objects(mask)
    sl = _sl(objs, lesion_id, mask.shape)
    bin3 = mask[sl] == lesion_id
    if not bin3.any():
        # nanochat-style: allow E1 (internal invariant: sl comes from find_objects for this label, so it is never empty)
        raise ValueError(f"empty mask for lesion {lesion_id}")
    dz, dy, dx = spacing.astype(np.float64)
    n = int(bin3.sum())
    vol = float(n * dz * dy * dx)
    hu = ct[sl][bin3].astype(np.float64)
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
        w = np.sort(np.linalg.eigh(np.cov(coords.T))[0])[::-1]
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


def mask_stats_all(mask: np.ndarray, ids: list[int], spacing: np.ndarray, ct: np.ndarray) -> dict[int, MaskFeats]:
    objs = label_objects(mask)
    return {lid: mask_stats(mask, lid, spacing, ct, objects=objs) for lid in ids}


def centroids(mask: np.ndarray, ids: list[int]) -> dict[int, np.ndarray]:
    objs = label_objects(mask)
    out = {}
    for lid in ids:
        sl = _sl(objs, lid, mask.shape, pad=0)
        pts = np.argwhere(mask[sl] == lid)
        assert pts.size, f"empty mask label {lid}"
        off = np.array([s.start for s in sl], dtype=np.float64)
        out[lid] = pts.mean(axis=0).astype(np.float64) + off + 0.5
    return out
