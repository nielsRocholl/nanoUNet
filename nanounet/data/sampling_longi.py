"""Two-stream patch: FU 3ch as build_patch + BL 3ch VOI co-located on the paired
baseline centroid. BL prompt = ONE exact click at the true baseline center (no jitter, no
spurious). Null baseline duplicates FU (identity DWB)."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import Callable, Iterator, Tuple

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nanounet.config import RoiPromptConfig
from nanounet.data.sampling import _sample_bbox, crop_patch, prompt_channels
from nanounet.prompt.encoding import encode_points_to_heatmap_pair


def _weights_from_props(properties: dict, cts_global: list) -> np.ndarray | None:
    w = properties.get("centroid_weights")
    if w is None:
        return None
    assert len(w) == len(cts_global), (len(w), len(cts_global))
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return w / s if s > 0 else None


def _lookup_pair(
    anchor: Tuple[int, int, int] | None,
    cts: list[Tuple[int, int, int]],
    pairs: list | None,
) -> list[int] | None:
    if anchor is None or pairs is None:
        return None
    try:
        idx = cts.index(anchor)
    except ValueError:
        return None
    return pairs[idx]


def build_patch_longi(
    fu_data,
    fu_seg,
    fu_prop: dict,
    open_bl: Callable[[str], AbstractContextManager[tuple]],
    cfg: RoiPromptConfig,
    patch_size: np.ndarray,
    final_patch_size: np.ndarray,
    force_zero_prompt: bool,
    rng: np.random.Generator,
) -> dict:
    cts = [tuple(map(int, c)) for c in fu_prop["centroids_zyx"]]
    weights = _weights_from_props(fu_prop, cts)
    need_to_pad = (patch_size - final_patch_size).astype(int)
    shape = np.array(fu_data.shape[1:])
    bbox_lbs, bbox_ubs, anchor = _sample_bbox(
        shape,
        cts,
        weights,
        cfg.sampling.fg_patch_prob,
        patch_size,
        need_to_pad,
        rng,
    )
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    fu_crop, seg_crop, pshape, pslc = crop_patch(fu_data, fu_seg, bbox)
    fu_hm = prompt_channels(seg_crop, cts, pslc, pshape, cfg, force_zero_prompt, rng)
    fu_stream = np.concatenate([fu_crop, fu_hm], axis=0)

    bl_id = fu_prop.get("baseline_case_id")
    pairs = fu_prop.get("pairs_zyx")
    c_bl = _lookup_pair(anchor, cts, pairs)
    if bl_id is None or c_bl is None or force_zero_prompt:
        bl_stream = fu_stream
    else:
        off = np.array(anchor) - np.array(bbox_lbs)
        bl_lbs = [int(round(c_bl[k])) - int(off[k]) for k in range(3)]
        bl_bbox = [[bl_lbs[k], bl_lbs[k] + int(patch_size[k])] for k in range(3)]
        with open_bl(bl_id) as (bl_data, _):
            bl_crop = np.asarray(crop_and_pad_nd(bl_data, bl_bbox, 0))
        c_bl_local = tuple(int(off[k]) for k in range(3))
        pr = cfg.prompt
        bl_hm = encode_points_to_heatmap_pair(
            [c_bl_local],
            [],
            tuple(int(s) for s in pshape),
            pr.point_radius_vox,
            pr.encoding,
            None,
            pr.prompt_intensity_scale,
        ).numpy()
        bl_stream = np.concatenate([bl_crop, bl_hm], axis=0)

    x = np.concatenate([fu_stream, bl_stream], axis=0)
    return {"image": x.astype(np.float32), "segmentation": seg_crop.astype(np.int16)}


@contextmanager
def bl_case_opener(ds, bl_id: str) -> Iterator[tuple]:
    with ds.open_case(bl_id, need_seg=False) as (data, _seg, _seg_prev, _props):
        yield data, None
