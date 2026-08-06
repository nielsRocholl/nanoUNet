"""Click-conditional targets: foreground only for lesion instances that received a click.

The seg on disk is binary, so instance identity is recovered by cc3d on the CROP -- 5.7 ms mean
against a 477 ms build_patch baseline (measured), i.e. inside the IO noise the path already carries.
Full-volume connected components would not be.

Two things here exist because TRAINING MUST MATCH INFERENCE, and an earlier version of this file
did not:

1. Membership is by VOXEL OVERLAP, not by whether a lesion's centroid lands in the patch. A large
   lesion spans several patches; in the neighbouring ones its centroid is outside, and the old rule
   made all of it background. That taught "suppress anything touching the patch face" -- and
   nanounet/infer/border_expand.py works by continuing exactly where the prediction touches a patch
   face, so the model was being trained to break its own inference path. Measured cost of the old
   rule: 13.7% of all foreground voxels silently suppressed.

2. Every kept lesion ALWAYS gets a click. If displacement pushes its click out of the patch, the
   click is placed on the lesion's largest in-crop component instead of dropped. Inference does the
   same thing (nanounet/infer/longi_row.py:37-39 falls back to local_prompt_points_for_patch when
   no click lands inside a patch), so dropping it here trained the opposite of deployment.

The kept set is drawn ONCE per patch, before displacement, so every prompt variant shares one
target: that is what keeps the consistency term measuring click PLACEMENT rather than penalising
two variants for disagreeing about which lesion they were pointed at.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cc3d
import numpy as np

from nanounet.prompt.centroids import filter_centroids_in_patch

ZYX = Tuple[int, int, int]


def resolve_instance_target(
    seg_crop: np.ndarray,
    cts_global: List[ZYX],
    bboxes_global: List[List[int]],
    pslc: tuple,
    keep_prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, list[int], Dict[int, ZYX]]:
    """Returns (target, kept, fallback_clicks).

    `kept` indexes into cts_global. `fallback_clicks` maps a kept lesion to a patch-local point on
    its own tissue, used by select_prompt_points when displacement throws that lesion's click out of
    the patch -- so a kept lesion is never silently left unclicked."""
    m = seg_crop[0] if seg_crop.ndim == 4 else seg_crop
    lab, n = cc3d.connected_components((m > 0).astype(np.uint8), connectivity=26, return_N=True)
    off = np.array([s.start for s in pslc])

    comp_lesion, comp_centroid, comp_size = _map_components(lab, n, off, bboxes_global, cts_global)
    in_patch = sorted({j for j in comp_lesion.values() if j is not None})
    kept = draw_kept(in_patch, keep_prob, rng)
    kept_set = set(kept)

    kept_labels = [L for L, j in comp_lesion.items() if j in kept_set]
    if kept_labels:
        out = np.where(np.isin(lab, kept_labels), 1, 0).astype(m.dtype)
    else:
        out = np.zeros_like(m)
    out[m < 0] = -1

    # One fallback click per kept lesion: the centroid of its LARGEST component in this crop.
    best: Dict[int, Tuple[int, ZYX]] = {}
    for L, j in comp_lesion.items():
        if j not in kept_set:
            continue
        if j not in best or comp_size[L] > best[j][0]:
            best[j] = (comp_size[L], comp_centroid[L])
    fallback = {j: v[1] for j, v in best.items()}
    return out.reshape(seg_crop.shape), kept, fallback


def _map_components(lab, n: int, off, bboxes_global, cts_global):
    """Assign every cc3d component in the crop to the global lesion it came from.

    Cropping can split one lesion into several components, and every component is a subset of
    exactly one lesion's voxels (lesions are disjoint), so matching a component's centroid against
    the per-lesion global bboxes in the sidecar recovers the parent. Ambiguity (overlapping
    bboxes) or a concave component whose centroid sits outside its own bbox falls back to the
    nearest lesion centroid -- never to "no lesion", which would silently suppress tissue."""
    comp_lesion: Dict[int, int | None] = {}
    comp_centroid: Dict[int, ZYX] = {}
    comp_size: Dict[int, int] = {}
    if n == 0:
        return comp_lesion, comp_centroid, comp_size
    stats = cc3d.statistics(lab, no_slice_conversion=True)
    cents, counts = stats["centroids"], stats["voxel_counts"]
    for L in range(1, n + 1):
        if counts[L] == 0:
            continue
        c_local = tuple(int(round(v)) for v in cents[L])
        comp_centroid[L] = _clip(c_local, lab.shape)
        comp_size[L] = int(counts[L])
        comp_lesion[L] = _lesion_for_point(np.asarray(c_local) + off, bboxes_global, cts_global)
    return comp_lesion, comp_centroid, comp_size


def _lesion_for_point(p_global, bboxes_global, cts_global) -> int | None:
    hits = [j for j, b in enumerate(bboxes_global) if _in_bbox(p_global, b)]
    if len(hits) == 1:
        return hits[0]
    pool = hits if hits else range(len(cts_global))
    best, bd = None, None
    for j in pool:
        c = cts_global[j]
        d = (p_global[0] - c[0]) ** 2 + (p_global[1] - c[1]) ** 2 + (p_global[2] - c[2]) ** 2
        if bd is None or d < bd:
            best, bd = j, d
    return best


def _in_bbox(p, b) -> bool:
    """`b` is [z0, z1, y0, y1, x0, x1], inclusive -- the format centroids.py writes."""
    return b[0] <= p[0] <= b[1] and b[2] <= p[1] <= b[3] and b[4] <= p[2] <= b[5]


def _clip(p: ZYX, shape) -> ZYX:
    return tuple(int(min(max(v, 0), s - 1)) for v, s in zip(p, shape))


def draw_kept(in_patch: list[int], keep_prob: float, rng: np.random.Generator) -> list[int]:
    """Which in-patch lesions keep their click. Drawn once per patch, on the UNDISPLACED lesion
    set, so the choice cannot depend on the random displacement (which differs per variant)."""
    if keep_prob >= 1.0:
        return list(in_patch)
    return [j for j in in_patch if rng.random() < keep_prob]


def kept_clicks(displaced, kept, pslc, fallback) -> list:
    """Patch-local click for every kept lesion. A kept lesion is NEVER left unclicked: if its
    displaced click landed outside the patch, its fallback point (a voxel on its own tissue in this
    crop) is used instead -- mirroring inference, which clamps a click into any patch that contains
    none (longi_row.py:37-39)."""
    out = []
    for j in kept:
        inp = filter_centroids_in_patch([displaced[j]], pslc)
        if inp:
            out.append(inp[0])
        elif fallback and j in fallback:
            out.append(fallback[j])
    return out
