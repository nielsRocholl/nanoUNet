"""Forward-map meta-CSV cog_fu/cog_bl into preprocessed voxels; pair each FU
centroid to its BL case + BL centroid; QC by median match distance."""

from __future__ import annotations

from collections import Counter

import numpy as np

from nanounet.plan.lesion_types import case_to_csv, cog_to_preprocessed


def patient_bl_cases(all_ids: list[str], hash_: str) -> list[str]:
    out: list[str] = []
    for cid in all_ids:
        if "_BL_img" not in cid:
            continue
        h, _ = case_to_csv(cid)
        if h == hash_:
            out.append(cid)
    return out


def build_fu_baseline(
    fu_id: str,
    fu_props: dict,
    fu_pre_shape: tuple[int, int, int],
    bl_candidates: list[tuple[str, dict, tuple[int, int, int]]],
    lesions_bl_fu: list[tuple[np.ndarray, np.ndarray]],
    tf: list[int],
    axis_order: str,
    max_match_dist: float,
) -> tuple[dict, dict]:
    _ = fu_id
    fu_bbox = fu_props["bbox_used_for_cropping"]
    fu_shape_ac = fu_props["shape_after_cropping_and_before_resampling"]
    mapped_fu = [
        cog_to_preprocessed(cog_fu, tf, fu_bbox, fu_shape_ac, fu_pre_shape, axis_order)
        for _, cog_fu in lesions_bl_fu
    ]
    cogs_fu = np.stack(mapped_fu) if mapped_fu else np.empty((0, 3))

    pairs_zyx: list[list[int] | None] = []
    bl_ids: list[str] = []
    fu_dists: list[float] = []

    for c in fu_props["centroids_zyx"]:
        c_arr = np.asarray(c, dtype=float)
        if len(mapped_fu) == 0:
            pairs_zyx.append(None)
            continue
        d = np.linalg.norm(cogs_fu - c_arr[None, :], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) > max_match_dist:
            pairs_zyx.append(None)
            continue
        cog_bl_raw = lesions_bl_fu[j][0]
        best_id, best_cog, best_bl_dist = None, None, float("inf")
        for bl_id, bl_props, bl_pre_shape in bl_candidates:
            bl_bbox = bl_props["bbox_used_for_cropping"]
            bl_shape_ac = bl_props["shape_after_cropping_and_before_resampling"]
            cog_bl = cog_to_preprocessed(
                cog_bl_raw, tf, bl_bbox, bl_shape_ac, bl_pre_shape, axis_order
            )
            cents = bl_props["centroids_zyx"]
            if not cents:
                continue
            bc = np.stack([np.asarray(x, dtype=float) for x in cents])
            bd = np.linalg.norm(bc - cog_bl[None, :], axis=1)
            k = int(np.argmin(bd))
            if float(bd[k]) <= max_match_dist and float(bd[k]) < best_bl_dist:
                best_bl_dist = float(bd[k])
                best_id = bl_id
                best_cog = [int(round(cog_bl[i])) for i in range(3)]
        if best_id is None:
            pairs_zyx.append(None)
            continue
        pairs_zyx.append(best_cog)
        bl_ids.append(best_id)
        fu_dists.append(float(d[j]))

    baseline_case_id = Counter(bl_ids).most_common(1)[0][0] if bl_ids else None
    n_paired = len(bl_ids)
    stats = {
        "n_centroids": len(fu_props["centroids_zyx"]),
        "n_paired": n_paired,
        "median_match_dist": float(np.median(fu_dists)) if fu_dists else float("inf"),
    }
    return {"baseline_case_id": baseline_case_id, "pairs_zyx": pairs_zyx}, stats
