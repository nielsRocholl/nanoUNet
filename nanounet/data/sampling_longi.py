"""Two-stream patch from a 2-channel case: ch0 FU_CT, ch1 warped-BL_CT (voxel-aligned by a shared
preprocessing crop). FU click SELECTION = build_patch's prompt, sourced from prop["fu_clicks_zyx"]
— the union of BL+FU lesion ids (from nanounet_longi_clicks --clicks-fu-dir), not just real FU
lesions. This includes "disappeared" lesions (no FU ground truth at that point), so the model is
prompted there and supervised by seg_crop to predict nothing -> learns disappearance. These are
registration-propagated points (not mask-derived guesses), so we jitter them exactly like build_patch
via cfg.sampling.propagated -- fu_clicks_zyx has no 1:1 lesion correspondence, so volume_vox is
resolved by nearest-neighbour match against centroids_zyx (see _volume_for_click), falling back to a
pooled size-bin draw. BL click selection = ALL in-patch warped clicks (positives only, no jitter, no
spurious). Null baseline (has_baseline false, force_zero_prompt, or the ablation switch) duplicates
FU -> identity DWB; rendering (both heatmaps and the identity-DWB duplication) happens after
augmentation, in patch_iterable.py."""

from __future__ import annotations

import numpy as np

from nanounet.config import RoiPromptConfig
from nanounet.data.patch_bbox import _sample_bbox, crop_patch
from nanounet.data.sampling import draw_false_pos, points_variant
from nanounet.prompt.centroids import filter_centroids_in_patch

_NEAREST_MATCH_MAX_VOX = 20.0


def _volumes_for_clicks(cts: list, prop: dict) -> list[float | None]:
    """volume_vox for each FU click, matched to the nearest centroids_zyx entry within
    _NEAREST_MATCH_MAX_VOX; None (-> pooled draw) if no centroid is that close."""
    centroids = prop.get("centroids_zyx") or []
    volumes = prop.get("volume_vox") or []
    if not centroids or not cts:
        return [None] * len(cts)
    pts = np.asarray(centroids, dtype=np.float64)
    out: list[float | None] = []
    for c in cts:
        d2 = np.sum((pts - np.asarray(c, dtype=np.float64)) ** 2, axis=1)
        j = int(np.argmin(d2))
        out.append(float(volumes[j]) if d2[j] <= _NEAREST_MATCH_MAX_VOX**2 else None)
    return out


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
    prompts_per_patch: int = 1,
    extra_rng: np.random.Generator | None = None,
) -> dict:
    assert data.shape[0] == 2, data.shape  # ch0 FU_CT, ch1 warped BL_CT
    assert "fu_clicks_zyx" in prop, (
        "fu_clicks_zyx missing from case properties; preprocessed data predates the FU-click mapping "
        "step (this is a longi-only field, not the legacy centroids_zyx).\n"
        "Fix: nanounet_longi_clicks -d <id> --plans <plans> --clicks-dir <clicksTr> --clicks-fu-dir <clicksTrFU>"
    )
    cts = [tuple(map(int, c)) for c in prop["fu_clicks_zyx"]]
    # centroid_weights were sized/ordered for the old real-FU-lesion centroids_zyx, not the new
    # BL+FU union (which also includes disappeared-lesion points with no weight entry) -> uniform sampling.
    weights = None
    need_to_pad = (patch_size - final_patch_size).astype(int)
    shape = np.array(data.shape[1:])
    bbox_lbs, bbox_ubs, _anchor = _sample_bbox(
        shape, cts, weights, cfg.sampling.fg_patch_prob, patch_size, need_to_pad, rng
    )
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    both_crop, seg_crop, _pshape, pslc = crop_patch(data, seg, bbox)  # both_crop: (2, *pshape)
    fu_volumes = _volumes_for_clicks(cts, prop)
    has_bl = prop.get("has_baseline", False)
    null_baseline = bool(force_zero_prompt or force_null_baseline or not has_bl)
    if null_baseline:
        bl_fixed = None
    else:  # ALL in-patch warped BL clicks, local coords; already real points, so never jittered
        clicks = [tuple(map(int, c)) for c in prop["bl_clicks_zyx"]]
        bl_fixed = np.asarray(filter_centroids_in_patch(clicks, pslc), dtype=np.float32).reshape(-1, 3)
    # N independent FU click draws over ONE shared crop (see build_patch). A null baseline duplicates
    # that variant's own FU points -> DWB(x_FU - x_FU)=0 -> identity (single-timepoint).
    # See build_patch: ONE decoy per patch, shared by every variant.
    fp = draw_false_pos(seg_crop, cfg, force_zero_prompt, rng)
    variants = []
    for _ in range(prompts_per_patch):
        v = points_variant(seg_crop, cts, pslc, cfg, force_zero_prompt, rng, True, fu_volumes, fp)
        v["bl_points_pos"] = v["points_pos"] if null_baseline else bl_fixed
        variants.append(v)
    if extra_rng is not None:
        # See build_patch: independent RNG stream, same crop, does not perturb the main sequence.
        v = points_variant(seg_crop, cts, pslc, cfg, force_zero_prompt, extra_rng, True, fu_volumes, fp)
        v["bl_points_pos"] = v["points_pos"] if null_baseline else bl_fixed
        variants.append(v)

    return {
        "image": both_crop.astype(np.float32),  # 2ch: [FU_CT, BL_CT]
        "segmentation": seg_crop.astype(np.int16),
        "points_variants": variants,
        "null_baseline": null_baseline,
    }
