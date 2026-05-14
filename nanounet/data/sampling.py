"""Four training modes: pos, pos+spur, pos_no_prompt, neg — bbox + point lists."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nanounet.config import RoiPromptConfig
from nanounet.prompt.centroids import centroids_from_seg, filter_centroids_in_patch
from nanounet.prompt.encoding import encode_points_to_heatmap_pair
from nanounet.prompt.propagation import apply_propagation_offset

MODE_POS, MODE_SPUR, MODE_NO_PROMPT, MODE_NEG = 0, 1, 2, 3


def _sample_spurious(seg: np.ndarray, n: int, rng: np.random.Generator) -> List[Tuple[int, int, int]]:
    s = np.asarray(seg)
    if s.ndim == 4:
        s = s[0]
    coords = np.argwhere(s == 0)
    if len(coords) == 0:
        return []
    n = min(n, len(coords))
    idx = rng.choice(len(coords), n, replace=False)
    return [tuple(int(c) for c in coords[i]) for i in idx]


def _lesion_voxels(seg: np.ndarray) -> np.ndarray:
    s = np.asarray(seg)
    if s.ndim == 4:
        s = s[0]
    return np.argwhere(s > 0)


def _has_lesion(seg: np.ndarray) -> bool:
    s = np.asarray(seg)
    if s.ndim == 4:
        s = s[0]
    return bool(np.any(s > 0))


def mode_bbox(
    shape: np.ndarray,
    class_locations: Union[dict, None],
    cfg: RoiPromptConfig,
    patch_size: np.ndarray,
    need_to_pad: np.ndarray,
    annotated_classes_key,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], int]:
    need = need_to_pad.copy()
    dim = len(shape)
    for d in range(dim):
        if need[d] + shape[d] < patch_size[d]:
            need[d] = patch_size[d] - shape[d]
    lbs_ = [-need[i] // 2 for i in range(dim)]
    ubs_ = [shape[i] + need[i] // 2 + need[i] % 2 - patch_size[i] for i in range(dim)]

    probs = np.array(cfg.sampling.mode_probs)
    mode = int(rng.choice(4, p=probs))
    eligible = []
    if class_locations:
        eligible = [k for k in class_locations if k != annotated_classes_key and len(class_locations[k]) > 0]
    has_fg = len(eligible) > 0
    if mode == MODE_NEG and not has_fg:
        mode = MODE_POS
    elif mode != MODE_NEG and not has_fg:
        mode = MODE_NEG
    force_fg = mode != MODE_NEG

    if not force_fg:
        bbox_lbs = [int(rng.integers(lbs_[i], ubs_[i] + 1)) for i in range(dim)]
    else:
        assert class_locations is not None
        eligible_f = [k for k in class_locations if k != annotated_classes_key and len(class_locations[k]) > 0]
        t2 = [k == annotated_classes_key if isinstance(k, tuple) else False for k in eligible_f]
        if any(t2) and len(eligible_f) > 1:
            eligible_f.pop(np.where(t2)[0][0])
        selected_class = eligible_f[int(rng.choice(len(eligible_f)))] if eligible_f else None
        if selected_class is not None:
            voxels = class_locations[selected_class]
            sel_v = voxels[int(rng.choice(len(voxels)))]
            bbox_lbs = []
            for i in range(dim):
                v = int(sel_v[i + 1])
                lo = max(lbs_[i], v - patch_size[i] + 1)
                hi = min(v, ubs_[i])
                if lo > hi:
                    bbox_lbs.append(max(lbs_[i], v - patch_size[i] // 2))
                else:
                    bbox_lbs.append(int(rng.integers(lo, hi + 1)))
        else:
            bbox_lbs = [int(rng.integers(lbs_[i], ubs_[i] + 1)) for i in range(dim)]
    bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(dim)]
    return bbox_lbs, bbox_ubs, mode


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
    need_to_pad = (patch_size - final_patch_size).astype(int)
    shape = np.array(data.shape[1:])
    cl = properties.get("class_locations")
    bbox_lbs, bbox_ubs, mode = mode_bbox(shape, cl, cfg, patch_size, need_to_pad, annotated_classes_key, rng)
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    data_crop = np.asarray(crop_and_pad_nd(data, bbox, 0))
    seg_crop = np.asarray(crop_and_pad_nd(seg, bbox, -1))
    patch_shape = tuple(bbox_ubs[k] - bbox_lbs[k] for k in range(3))
    slz = slice(bbox_lbs[0], bbox_ubs[0])
    sly = slice(bbox_lbs[1], bbox_ubs[1])
    slx = slice(bbox_lbs[2], bbox_ubs[2])
    pslc = (slz, sly, slx)

    if mode == MODE_NEG and _has_lesion(seg_crop):
        mode = MODE_POS
    elif mode != MODE_NEG and not _has_lesion(seg_crop):
        mode = MODE_NEG

    pr = cfg.prompt
    if force_zero_prompt:
        pp, pn = [], []
    elif mode == MODE_NO_PROMPT:
        pp, pn = [], []
    elif mode == MODE_NEG:
        n0 = int(rng.integers(cfg.sampling.n_neg[0], cfg.sampling.n_neg[1] + 1))
        pn = _sample_spurious(seg_crop, n0, rng)
        pp = []
    else:
        raw_c = properties.get("centroids_zyx")
        if raw_c is None:
            cts = centroids_from_seg(seg_crop)
        else:
            cts = [tuple(int(x) for x in c) for c in raw_c]
        pp = filter_centroids_in_patch(cts, pslc)
        if not pp:
            lv = _lesion_voxels(seg_crop)
            if len(lv) > 0:
                ii = int(rng.integers(len(lv)))
                pp = [tuple(int(x) for x in lv[ii])]
        prop = cfg.sampling.propagated
        rg2 = np.random.default_rng(int(rng.integers(0, 2**31)))
        pp = [apply_propagation_offset(p, patch_shape, prop.sigma_per_axis, prop.max_vox, rg2) for p in pp]
        if mode == MODE_SPUR:
            ns = int(rng.integers(cfg.sampling.n_spur[0], cfg.sampling.n_spur[1] + 1))
            pp = pp + _sample_spurious(seg_crop, ns, rng)
        pn = []

    hm = encode_points_to_heatmap_pair(
        pp, pn, patch_shape, pr.point_radius_vox, pr.encoding, None, pr.prompt_intensity_scale
    )
    x = np.concatenate([data_crop, hm.numpy()], axis=0)
    return {"image": x.astype(np.float32), "segmentation": seg_crop.astype(np.int16), "mode": int(mode)}
