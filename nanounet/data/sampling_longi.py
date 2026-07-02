"""Two-stream patch from a 2-channel case: ch0 FU_CT, ch1 warped-BL_CT (voxel-aligned by a shared
preprocessing crop). FU stream = build_patch's prompt (all in-patch centroids, jittered). BL stream
= same-bbox crop of ch1 + ALL in-patch warped clicks (positives only, no jitter, no spurious). Null
baseline (has_baseline false, force_zero_prompt, or the ablation switch) duplicates FU -> identity DWB."""

from __future__ import annotations

import numpy as np

from nanounet.config import RoiPromptConfig
from nanounet.data.sampling import _sample_bbox, crop_patch, prompt_channels
from nanounet.prompt.centroids import filter_centroids_in_patch
from nanounet.prompt.encoding import encode_points_to_heatmap_pair


def _weights_from_props(properties: dict, cts_global: list) -> np.ndarray | None:
    w = properties.get("centroid_weights")
    if w is None:
        return None
    assert len(w) == len(cts_global), (len(w), len(cts_global))
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return w / s if s > 0 else None


def build_patch_longi(
    data,
    seg,
    prop: dict,
    cfg: RoiPromptConfig,
    patch_size: np.ndarray,
    final_patch_size: np.ndarray,
    force_zero_prompt: bool,
    force_null_baseline: bool,
    rng: np.random.Generator,
) -> dict:
    assert data.shape[0] == 2, data.shape  # ch0 FU_CT, ch1 warped BL_CT
    cts = [tuple(map(int, c)) for c in prop["centroids_zyx"]]
    weights = _weights_from_props(prop, cts)
    need_to_pad = (patch_size - final_patch_size).astype(int)
    shape = np.array(data.shape[1:])
    bbox_lbs, bbox_ubs, _anchor = _sample_bbox(
        shape, cts, weights, cfg.sampling.fg_patch_prob, patch_size, need_to_pad, rng
    )
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    both_crop, seg_crop, pshape, pslc = crop_patch(data, seg, bbox)  # both_crop: (2, *pshape)
    fu_hm = prompt_channels(seg_crop, cts, pslc, pshape, cfg, force_zero_prompt, rng)
    fu_stream = np.concatenate([both_crop[0:1], fu_hm], axis=0)

    has_bl = prop.get("has_baseline", False)
    if force_zero_prompt or force_null_baseline or not has_bl:
        bl_stream = fu_stream  # duplicate FU -> DWB(x_FU - x_FU)=0 -> identity (single-timepoint)
    else:
        clicks = [tuple(map(int, c)) for c in prop["bl_clicks_zyx"]]
        bl_local = filter_centroids_in_patch(clicks, pslc)  # ALL in-patch warped clicks, local coords
        pr = cfg.prompt
        bl_hm = encode_points_to_heatmap_pair(
            bl_local, [], tuple(int(s) for s in pshape),
            pr.point_radius_vox, pr.encoding, None, pr.prompt_intensity_scale,
        ).numpy()
        bl_stream = np.concatenate([both_crop[1:2], bl_hm], axis=0)

    x = np.concatenate([fu_stream, bl_stream], axis=0)  # 6ch: [FU_CT,FU_hm+,FU_hm-,BL_CT,BL_hm+,BL_hm-]
    return {"image": x.astype(np.float32), "segmentation": seg_crop.astype(np.int16)}
