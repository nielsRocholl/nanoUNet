"""Click-conditional targets: foreground only for lesion instances that received a click.

The seg on disk is binary, so instance identity is recovered by cc3d on the CROP -- 5.7 ms mean
against a 477 ms build_patch baseline (measured, see HANDOFF_step1_done_step6_next.md), i.e. inside
the IO noise the path already carries. Full-volume connected components would not be.

The kept set is drawn ONCE per patch, before displacement, so every prompt variant of a patch shares
one target: that is what keeps the consistency term measuring click PLACEMENT rather than penalising
two variants for legitimately disagreeing about which lesion they were pointed at."""

from __future__ import annotations

from typing import List, Tuple

import cc3d
import numpy as np


def resolve_instance_target(
    seg_crop: np.ndarray,
    cts_global: List[Tuple[int, int, int]],
    seeds_global: List[Tuple[int, int, int]],
    pslc: tuple,
    keep_prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, list[int]]:
    """cc3d on the crop + kept-set draw + local-coordinate conversion + masking, bundled so
    build_patch (sampling.py) stays under the LOC budget (plan section 4). Returns (seg_out, kept)
    where `kept` indexes into cts_global -- callers pass it on to points_variant so only kept
    lesions are clicked."""
    m = seg_crop[0] if seg_crop.ndim == 4 else seg_crop
    lab, _n = cc3d.connected_components((m > 0).astype(np.uint8), connectivity=26, return_N=True)
    in_patch = [j for j, c in enumerate(cts_global) if _inside(c, pslc)]
    # ONE kept-set draw for the whole patch: every variant must share the target, else the
    # consistency term would penalise two variants for holding different (correct) answers.
    kept = draw_kept(in_patch, keep_prob, rng)
    seg_out = clicked_target(seg_crop, lab, kept, _to_local(seeds_global, pslc), _to_local(cts_global, pslc))
    return seg_out, kept


def _inside(c: Tuple[int, int, int], pslc: tuple) -> bool:
    slz, sly, slx = pslc
    z, y, x = c
    return slz.start <= z < slz.stop and sly.start <= y < sly.stop and slx.start <= x < slx.stop


def _to_local(pts: List[Tuple[int, int, int]], pslc: tuple) -> List[Tuple[int, int, int]]:
    slz, sly, slx = pslc
    return [(z - slz.start, y - sly.start, x - slx.start) for z, y, x in pts]


def draw_kept(in_patch: list[int], keep_prob: float, rng: np.random.Generator) -> list[int]:
    """Which in-patch lesions keep their click. Drawn once per patch, on UNDISPLACED centroids, so
    the choice cannot depend on the random displacement (which differs per variant)."""
    if keep_prob >= 1.0:
        return list(in_patch)
    return [j for j in in_patch if rng.random() < keep_prob]


def clicked_target(
    seg_crop: np.ndarray,
    lab: np.ndarray,
    kept: list[int],
    seeds_local: List[Tuple[int, int, int]],
    cts_local: List[Tuple[int, int, int]],
) -> np.ndarray:
    """seg_crop masked down to the cc3d components of `kept`. Components no kept lesion maps to
    become background -- correct under the project rule: a lesion whose centroid is outside the
    patch never receives a click, so it must not be segmented. -1 padding is preserved.

    Probe order per kept lesion: seed_zyx first, then centroids_zyx -- the plain centroid falls
    outside its own lesion ~12% of the time on concave shapes (see centroids.py), the seed
    (argmax-EDT) is guaranteed interior."""
    shape = lab.shape
    kept_labels: set[int] = set()
    for j in kept:
        v = _probe(lab, shape, seeds_local[j])
        if v == 0:
            v = _probe(lab, shape, cts_local[j])
        if v != 0:
            kept_labels.add(v)
    m = seg_crop[0] if seg_crop.ndim == 4 else seg_crop
    if kept_labels:
        out = np.where(np.isin(lab, list(kept_labels)), 1, 0).astype(m.dtype)
    else:
        out = np.zeros_like(m)
    out[m < 0] = -1
    return out.reshape(seg_crop.shape)


def _probe(lab: np.ndarray, shape: tuple, pt: Tuple[int, int, int]) -> int:
    z, y, x = pt
    d, h, w = shape
    if 0 <= z < d and 0 <= y < h and 0 <= x < w:
        return int(lab[z, y, x])
    return 0
