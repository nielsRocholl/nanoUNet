"""Per-click prompt sampling: jitter authored centroids (skippable via select_prompt_points(jitter=
False)), optional false-positive clicks (gated by probability). Returns click COORDINATES only --
rendering to heatmaps happens after augmentation, in patch_iterable.py.

When `cfg.sampling.instance_targets` is set, the returned segmentation is click-conditional:
foreground only for lesion instances that kept their click (nanounet/data/instance_target.py). The
kept set is drawn once per patch, before displacement, and shared by every prompt variant."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree

from nanounet.config import RoiPromptConfig
from nanounet.data.error_table import draw_propagated_offset
from nanounet.data.instance_target import kept_clicks, resolve_instance_target
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
    false_pos: List[Tuple[int, int, int]] | None = None,
    kept: list[int] | None = None,
    fallback: dict | None = None,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int]:
    """Click SELECTION only -- (positive, negative, n_false_pos) patch-local point lists; heatmap
    rendering happens later, after augmentation (nanounet/train/patch_iterable.py). Offset applies
    to the GLOBAL centroid first, THEN filtered into the patch -- an out-of-patch displaced click
    is dropped, never clamped. jitter=False is for points already real (registered/propagated).
    `false_pos` (drawn ONCE PER PATCH by the caller) is appended as the LAST `n_false_pos` entries
    of `pp` -- click_inside_flags relies on that order. `kept` (drawn once per patch by build_patch
    for instance_targets) fixes WHICH lesions may click; the dropout below must not re-run it, or
    the kept-masked target would disagree with what got clicked."""
    pp: List[Tuple[int, int, int]] = []
    pn: List[Tuple[int, int, int]] = []
    n_fp = 0
    if not force_zero_prompt:
        if jitter:
            prop = cfg.sampling.propagated
            rg2 = np.random.default_rng(int(rng.integers(0, 2**31)))
            vols = volumes_vox if volumes_vox is not None else [None] * len(cts_global)
            assert len(vols) == len(cts_global)
            displaced = [draw_propagated_offset(c, v, prop, rg2) for c, v in zip(cts_global, vols)]
        else:
            displaced = list(cts_global)
        if kept is None:
            inch = filter_centroids_in_patch(displaced, pslc)
            cm = cfg.sampling.click_modes
            kept_ = inch if cm.drop == 0.0 else [p for p in inch if rng.random() < cm.pos]
        else:
            # Kept set fixed per patch (instance_targets): a kept lesion ALWAYS gets a click -- if
            # displaced out of the patch, fall back to a point on its own tissue (inference does the
            # same, longi_row.py:37-39) rather than drop it and train "no click => background".
            kept_ = kept_clicks(displaced, kept, pslc, fallback)
        pp = list(kept_)
        if false_pos:
            n_fp = len(false_pos)
            pp = pp + list(false_pos)
    return pp, pn, n_fp

def draw_false_pos(seg_crop, cfg: RoiPromptConfig, force_zero_prompt: bool, rng) -> list:
    """One decoy draw PER PATCH (not per variant) -- see select_prompt_points."""
    if force_zero_prompt or rng.random() >= cfg.sampling.false_pos_probability:
        return []
    return _sample_false_pos(seg_crop, rng)

# One decoy for the rare "click on empty tissue" case; not a difficulty knob (at deployment every click
# refers to a real lesion, and the genuine negative is the disappeared lesion, already in the data).
# The old 30-50 vox setting tuned hardness, wrongly: 42% of lesions have a neighbour closer than 30.
_FALSE_POS_GUARD_VOX = 5

def points_variant(
    seg_crop, cts_global, pslc, cfg, force_zero_prompt, rng, jitter, volumes_vox, false_pos=None,
    kept=None, fallback=None,
):
    """One prompt draw as float (N,3) arrays, ready to ride through the augmentation chain."""
    pp, pn, n_fp = select_prompt_points(
        seg_crop, cts_global, pslc, cfg, force_zero_prompt, rng, jitter, volumes_vox, false_pos, kept, fallback
    )
    return {
        "points_pos": np.asarray(pp, dtype=np.float32).reshape(-1, 3),
        "points_neg": np.asarray(pn, dtype=np.float32).reshape(-1, 3),
        "n_false_pos": n_fp,
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
    # Per-centroid sampling weights from the <case>_weights.json sidecar; absent => uniform draw,
    # unless require_weights refuses that fallback (nanounet/config.py SamplingConfig).
    w = properties.get("centroid_weights")
    if w is None and cfg.sampling.require_weights:
        raise KeyError(
            "centroid_weights missing (no <case>_weights.json sidecar) and sampling.require_weights "
            "is true: refusing uniform per-centroid sampling.\n"
            "Fix: nanounet_lesion_weights -d <id> --plans <plans> --meta-dir <dir of <hash>.csv "
            "lesion-type files>   -- or set require_weights to false"
        )
    weights = None
    if w is not None:
        assert len(w) == len(cts_global), (len(w), len(cts_global))
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        weights = w / s if s > 0 else None
    need_to_pad = (patch_size - final_patch_size).astype(int)
    shape = np.array(data.shape[1:])
    bbox_lbs, bbox_ubs, _anchor = _sample_bbox(
        shape, cts_global, weights, cfg.sampling.fg_patch_prob, patch_size, need_to_pad, rng
    )
    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]
    data_crop, seg_crop, _patch_shape, pslc = crop_patch(data, seg, bbox)
    # ONE decoy draw for the whole patch, shared by every variant: a consistency/agreement pair must
    # differ only in lesion-click placement, never in where the false-positive click landed.
    fp = draw_false_pos(seg_crop, cfg, force_zero_prompt, rng)
    seg_out = seg_crop
    kept: list[int] | None = None
    fallback: dict | None = None
    if cfg.sampling.instance_targets:
        raw_bb = properties.get("bboxes_zyx")
        if raw_bb is None:
            raise KeyError(
                "bboxes_zyx missing from case properties (needed to map each cc3d component in the "
                "crop back to its parent lesion). Fix: nanounet_preprocess --sidecars-only"
            )
        bboxes_global = [[int(v) for v in b] for b in raw_bb]
        assert len(bboxes_global) == len(cts_global), (len(bboxes_global), len(cts_global))
        seg_out, kept, fallback = resolve_instance_target(
            seg_crop, cts_global, bboxes_global, pslc, cfg.sampling.click_modes.pos, rng
        )
    # N independent click draws over ONE shared crop, so a consistency pair differs only in the
    # click. `seg_crop` (not `seg_out`) feeds draw_false_pos and click_inside_flags below: decoys
    # avoid real tissue, and the inside/outside vote reflects the real lesion, not the click-target.
    variants = [
        points_variant(seg_crop, cts_global, pslc, cfg, force_zero_prompt, rng, True, volumes_vox, fp, kept, fallback)
        for _ in range(prompts_per_patch)
    ]
    if extra_rng is not None:
        # Val prompt-agreement diagnostic: 1 more draw, same crop, RNG stream never touches `rng`.
        variants.append(
            points_variant(
                seg_crop, cts_global, pslc, cfg, force_zero_prompt, extra_rng, True, volumes_vox, fp, kept, fallback
            )
        )
    return {
        "image": data_crop.astype(np.float32),
        "segmentation": seg_out.astype(np.int16),
        "points_variants": variants,
    }
