"""Patch bbox geometry: where in the volume a training patch is cut from.

Kept apart from click sampling (sampling.py) -- one decides WHERE the patch is, the other decides
WHAT prompt goes in it."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


def _lbs_ubs(
    patch_size: np.ndarray, shape: np.ndarray, need_to_pad: np.ndarray
) -> tuple[list[int], list[int]]:
    need = need_to_pad.copy()
    dim = len(shape)
    for d in range(dim):
        if need[d] + shape[d] < patch_size[d]:
            need[d] = patch_size[d] - shape[d]
    lbs_ = [-need[i] // 2 for i in range(dim)]
    ubs_ = [shape[i] + need[i] // 2 + need[i] % 2 - patch_size[i] for i in range(dim)]
    return lbs_, ubs_


def _sample_bbox(
    shape: np.ndarray,
    centroids_global: List[Tuple[int, int, int]],
    weights,
    fg_patch_prob: float,
    patch_size: np.ndarray,
    need_to_pad: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], Tuple[int, int, int] | None]:
    lbs_, ubs_ = _lbs_ubs(patch_size, shape, need_to_pad)
    dim = len(shape)
    force_fg = rng.random() < fg_patch_prob
    anchor: Tuple[int, int, int] | None = None
    if force_fg and centroids_global:
        if weights is None:
            j = int(rng.integers(len(centroids_global)))
        else:
            j = int(rng.choice(len(centroids_global), p=weights))
        c = centroids_global[j]
        anchor = c
        bbox_lbs: List[int] = []
        for i in range(dim):
            v = int(c[i])
            lo = max(lbs_[i], v - patch_size[i] + 1)
            hi = min(v, ubs_[i])
            if lo > hi:
                bbox_lbs.append(max(lbs_[i], v - patch_size[i] // 2))
            else:
                bbox_lbs.append(int(rng.integers(lo, hi + 1)))
    else:
        bbox_lbs = [int(rng.integers(lbs_[i], ubs_[i] + 1)) for i in range(dim)]
    bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(dim)]
    return bbox_lbs, bbox_ubs, anchor


def crop_patch(data, seg, bbox) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], tuple]:
    data_crop = np.asarray(crop_and_pad_nd(data, bbox, 0))
    seg_crop = np.asarray(crop_and_pad_nd(seg, bbox, -1))
    bbox_lbs = [b[0] for b in bbox]
    bbox_ubs = [b[1] for b in bbox]
    patch_shape = tuple(int(bbox_ubs[k] - bbox_lbs[k]) for k in range(3))
    pslc = tuple(slice(bbox_lbs[k], bbox_ubs[k]) for k in range(3))
    return data_crop, seg_crop, patch_shape, pslc
