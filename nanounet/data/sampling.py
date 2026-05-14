"""Per-click prompt sampling: jitter authored centroids, optional false-positive clicks (gated by probability), encode positives."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from scipy.ndimage import distance_transform_edt

from nanounet.config import RoiPromptConfig
from nanounet.prompt.centroids import filter_centroids_in_patch
from nanounet.prompt.encoding import encode_points_to_heatmap_pair
from nanounet.prompt.propagation import apply_propagation_offset


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
    fg_patch_prob: float,
    patch_size: np.ndarray,
    need_to_pad: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int]]:
    lbs_, ubs_ = _lbs_ubs(patch_size, shape, need_to_pad)
    dim = len(shape)
    force_fg = rng.random() < fg_patch_prob
    if force_fg and centroids_global:
        c = centroids_global[int(rng.integers(len(centroids_global)))]
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
    return bbox_lbs, bbox_ubs


def _sample_false_pos(
    seg_crop: np.ndarray, n: int, min_dist_vox: int, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    """Random background voxels for positive-channel clicks ≥ min_dist away from foreground."""
    if n <= 0:
        return []
    s = np.asarray(seg_crop)
    if s.ndim == 4:
        s = s[0]
    pos = s > 0
    if not pos.any():
        allowed = ~pos
    else:
        allowed = distance_transform_edt(~pos) > float(min_dist_vox)
    coords = np.argwhere(allowed)
    if len(coords) == 0:
        return []
    k = min(n, len(coords))
    idx = rng.choice(len(coords), k, replace=False)
    return [tuple(int(v) for v in coords[i]) for i in idx]


def build_patch(
    data,
    seg,
    properties: dict,
    cfg: RoiPromptConfig,
    patch_size: np.ndarray,
    final_patch_size: np.ndarray,
    annotated_classes_key,
    force_zero_prompt: bool,
    rng: np.random.Generator,
) -> dict:
    _ = annotated_classes_key  # callers still pass annotated key from plans
    if "centroids_zyx" not in properties:
        raise KeyError("centroids_zyx required; no seg-derived fallback (R12)")
    raw_c = properties["centroids_zyx"]
    if raw_c is None:
        raise KeyError("centroids_zyx required; no seg-derived fallback (R12)")
    cts_global = [tuple(int(x) for x in c) for c in raw_c]
    need_to_pad = (patch_size - final_patch_size).astype(int)
    shape = np.array(data.shape[1:])
    bbox_lbs, bbox_ubs = _sample_bbox(
        shape,
        cts_global,
        cfg.sampling.fg_patch_prob,
        patch_size,
        need_to_pad,
        rng,
    )
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    data_crop = np.asarray(crop_and_pad_nd(data, bbox, 0))
    seg_crop = np.asarray(crop_and_pad_nd(seg, bbox, -1))
    patch_shape = tuple(int(bbox_ubs[k] - bbox_lbs[k]) for k in range(3))
    pslc = (
        slice(bbox_lbs[0], bbox_ubs[0]),
        slice(bbox_lbs[1], bbox_ubs[1]),
        slice(bbox_lbs[2], bbox_ubs[2]),
    )

    pp: List[Tuple[int, int, int]] = []
    pn: List[Tuple[int, int, int]] = []
    if not force_zero_prompt:
        inch = filter_centroids_in_patch(cts_global, pslc)
        cm = cfg.sampling.click_modes
        if cm.drop == 0.0:
            kept = list(inch)
        else:
            kept = [p for p in inch if rng.random() < cm.pos]
        prop = cfg.sampling.propagated
        rg2 = np.random.default_rng(int(rng.integers(0, 2**31)))
        pp = [apply_propagation_offset(p, patch_shape, prop.sigma_per_axis, prop.max_vox, rg2) for p in kept]
        lo_fp, hi_fp = cfg.sampling.n_false_pos
        if hi_fp <= 0 or rng.random() >= cfg.sampling.false_pos_probability:
            n_fp = 0
        else:
            n_fp = int(rng.integers(lo_fp, hi_fp + 1))
        if n_fp > 0:
            pp = pp + _sample_false_pos(seg_crop, n_fp, cfg.sampling.false_pos_min_dist_vox, rng)

    pr = cfg.prompt
    hm = encode_points_to_heatmap_pair(
        pp, pn, patch_shape, pr.point_radius_vox, pr.encoding, None, pr.prompt_intensity_scale
    )
    x = np.concatenate([data_crop, hm.numpy()], axis=0)
    return {"image": x.astype(np.float32), "segmentation": seg_crop.astype(np.int16)}
