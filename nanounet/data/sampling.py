"""Per-click prompt sampling: jitter authored centroids (skippable via select_prompt_points(jitter=
False)), optional false-positive clicks (gated by probability). Returns click COORDINATES only --
rendering to heatmaps happens after augmentation, in patch_iterable.py."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from nanounet.config import RoiPromptConfig
from nanounet.data.error_table import draw_propagated_offset
from nanounet.data.patch_bbox import _sample_bbox, crop_patch
from nanounet.prompt.centroids import filter_centroids_in_patch


def select_prompt_points(
    seg_crop: np.ndarray,
    cts_global: List[Tuple[int, int, int]],
    pslc: tuple,
    cfg: RoiPromptConfig,
    force_zero_prompt: bool,
    rng: np.random.Generator,
    jitter: bool = True,
    volumes_vox: List[float | None] | None = None,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Click SELECTION only -- returns (positive, negative) patch-local point lists; rendering to
    heatmaps happens later (after augmentation), see nanounet/train/patch_iterable.py.

    Order matters: offset applied to the GLOBAL centroid first, THEN filtered into the patch -- a
    displaced click that lands outside the patch is simply not rendered, no clamping to the border.
    jitter=False for points already real (registered/propagated), not mask-derived guesses."""
    pp: List[Tuple[int, int, int]] = []
    pn: List[Tuple[int, int, int]] = []
    if not force_zero_prompt:
        if jitter:
            prop = cfg.sampling.propagated
            rg2 = np.random.default_rng(int(rng.integers(0, 2**31)))
            vols = volumes_vox if volumes_vox is not None else [None] * len(cts_global)
            assert len(vols) == len(cts_global)
            displaced = [draw_propagated_offset(c, v, prop, rg2) for c, v in zip(cts_global, vols)]
        else:
            displaced = list(cts_global)
        inch = filter_centroids_in_patch(displaced, pslc)
        cm = cfg.sampling.click_modes
        kept = inch if cm.drop == 0.0 else [p for p in inch if rng.random() < cm.pos]
        pp = list(kept)
        if rng.random() < cfg.sampling.false_pos_probability:
            pp = pp + _sample_false_pos(seg_crop, rng)
    return pp, pn


# One decoy for the rare "click on empty tissue" case; not a difficulty knob (at deployment every click
# refers to a real lesion, and the genuine negative is the disappeared lesion, already in the data).
# The old 30-50 vox setting tuned hardness, wrongly: 42% of lesions have a neighbour closer than 30.
_FALSE_POS_GUARD_VOX = 5


def points_variant(seg_crop, cts_global, pslc, cfg, force_zero_prompt, rng, jitter, volumes_vox):
    """One prompt draw as float (N,3) arrays, ready to ride through the augmentation chain."""
    pp, pn = select_prompt_points(
        seg_crop, cts_global, pslc, cfg, force_zero_prompt, rng, jitter, volumes_vox
    )
    return {
        "points_pos": np.asarray(pp, dtype=np.float32).reshape(-1, 3),
        "points_neg": np.asarray(pn, dtype=np.float32).reshape(-1, 3),
    }


def _sample_false_pos(seg_crop: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """One random background voxel >= _FALSE_POS_GUARD_VOX from foreground. KD-tree rejection
    sampling over the sparse foreground coords: a full-volume EDT here dominated dataloader CPU
    and starved the GPU (~60% util)."""
    s = np.asarray(seg_crop)
    if s.ndim == 4:
        s = s[0]
    shape = s.shape
    fg = np.argwhere(s > 0)
    if len(fg) == 0:
        return [tuple(int(rng.integers(0, d)) for d in shape)]
    tree = cKDTree(fg)
    for _ in range(8):
        cand = np.stack([rng.integers(0, d, size=72) for d in shape], axis=1)
        dist, _ = tree.query(cand, k=1)
        hit = cand[dist > float(_FALSE_POS_GUARD_VOX)]
        if len(hit):
            return [tuple(int(v) for v in hit[0])]
    return []


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
    prompts_per_patch: int = 1,
    extra_rng: np.random.Generator | None = None,
) -> dict:
    _ = annotated_classes_key
    raw_c = properties.get("centroids_zyx")
    if raw_c is None:
        raise KeyError("centroids_zyx required; no seg-derived fallback (R12)")
    cts_global = [tuple(int(x) for x in c) for c in raw_c]
    raw_v = properties.get("volume_vox")
    if raw_v is None:
        raise KeyError(
            "volume_vox missing from case properties (needed to size-match the empirical "
            "registration-error draw to each lesion). Fix: nanounet_preprocess --sidecars-only"
        )
    volumes_vox = [float(v) for v in raw_v]
    assert len(volumes_vox) == len(cts_global), (len(volumes_vox), len(cts_global))
    w = properties.get("centroid_weights")
    if w is not None:
        assert len(w) == len(cts_global), (len(w), len(cts_global))
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        weights = w / s if s > 0 else None
    else:
        weights = None
    need_to_pad = (patch_size - final_patch_size).astype(int)
    shape = np.array(data.shape[1:])
    bbox_lbs, bbox_ubs, _anchor = _sample_bbox(
        shape, cts_global, weights, cfg.sampling.fg_patch_prob, patch_size, need_to_pad, rng
    )
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    data_crop, seg_crop, _patch_shape, pslc = crop_patch(data, seg, bbox)
    # N independent click draws over ONE shared crop, so a consistency pair differs only in the click.
    variants = [
        points_variant(seg_crop, cts_global, pslc, cfg, force_zero_prompt, rng, True, volumes_vox)
        for _ in range(prompts_per_patch)
    ]
    if extra_rng is not None:
        # Val prompt-agreement diagnostic: 1 more draw, same crop, RNG stream never touches `rng`.
        variants.append(
            points_variant(seg_crop, cts_global, pslc, cfg, force_zero_prompt, extra_rng, True, volumes_vox)
        )
    return {
        "image": data_crop.astype(np.float32),
        "segmentation": seg_crop.astype(np.int16),
        "points_variants": variants,
    }
