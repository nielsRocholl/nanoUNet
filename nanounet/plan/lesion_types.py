"""Assign a hard-type sampling weight to each preprocessed centroid by forward-mapping
meta-CSV lesion cogs (original coords) into preprocessed voxel space and nearest-matching.

The CSV cogs live in original-image voxel coords; preprocessed centroids are
transposed+cropped+resampled. We replay that geometry (transpose_forward, crop bbox,
resample scale) so a euclidean nearest-match in preprocessed voxels assigns each
centroid a lesion type, hence a boost weight. Hard types are oversampled because they
are diluted by Lung/Liver in uniform within-case sampling.
"""

from __future__ import annotations

import csv

import numpy as np

# Manual hard-type boost picked by the user; every other type (and unmatched centroids) is 1.0.
HARD_TYPE_BOOST = {"Lymph node": 4.0, "Soft tissue / Skin": 3.0, "Skeleton": 4.0}


def case_to_csv(case_id: str) -> tuple[str, str]:
    """Map a preprocessed case id to (hash, timepoint). timepoint is 'BL' or 'FU'."""
    s = case_id
    prefix = "d013_Longitudinal_CT_"
    assert s.startswith(prefix), s
    s = s[len(prefix):]
    has_bl = "_BL_img" in s
    has_fu = "_FU_img" in s
    # Exactly one timepoint marker must be present, else the id is malformed.
    assert has_bl != has_fu, case_id
    if has_bl:
        return s.split("_BL_img")[0], "BL"
    return s.split("_FU_img")[0], "FU"


def load_lesion_pairs(csv_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse CSV rows where both cog_bl and cog_fu are present."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bl = row["cog_bl"].strip()
            fu = row["cog_fu"].strip()
            if not bl or not fu:
                continue
            cog_bl = np.array([float(x) for x in bl.split()], dtype=float)
            cog_fu = np.array([float(x) for x in fu.split()], dtype=float)
            assert cog_bl.shape == (3,) and cog_fu.shape == (3,), (bl, fu)
            out.append((cog_bl, cog_fu))
    return out


def load_lesions(csv_path: str, timepoint: str) -> list[tuple[np.ndarray, str]]:
    """Parse the per-patient CSV; return (cog_raw, lesion_type) for lesions present at this timepoint."""
    assert timepoint in ("BL", "FU")
    col = "cog_bl" if timepoint == "BL" else "cog_fu"
    out: list[tuple[np.ndarray, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell = row[col].strip()
            if not cell:  # lesion absent at this timepoint
                continue
            cog = np.array([float(x) for x in cell.split()], dtype=float)
            assert cog.shape == (3,), cell
            out.append((cog, row["lesion_type"]))
    return out


def cog_to_preprocessed(
    cog_raw: np.ndarray,
    tf: list[int],
    bbox: list,
    shape_after_crop,
    pre_shape,
    axis_order: str,
) -> np.ndarray:
    """Forward-map one original-space cog into preprocessed voxel coords (z, y, x)."""
    assert axis_order in ("xyz", "zyx")
    p_zyx = cog_raw[::-1] if axis_order == "xyz" else cog_raw
    p_t = np.array([p_zyx[tf[d]] for d in range(3)], dtype=float)
    p_c = np.array([p_t[d] - bbox[d][0] for d in range(3)], dtype=float)
    scale = np.array([pre_shape[d] / shape_after_crop[d] for d in range(3)], dtype=float)
    return p_c * scale


def build_case_weights(
    centroids_zyx,
    lesions: list[tuple[np.ndarray, str]],
    boost: dict,
    max_match_dist: float,
) -> tuple[list[float], dict]:
    """Weight each centroid by its nearest mapped lesion's type (within max_match_dist)."""
    cents = [np.asarray(c, dtype=float) for c in centroids_zyx]
    if not cents:
        return [], {"n_centroids": 0, "n_matched": 0, "median_match_dist": float("inf"), "matched_types": []}

    weights: list[float] = []
    dists: list[float] = []
    matched_types: list[str] = []
    if lesions:
        cogs = np.stack([m for m, _ in lesions])  # (L, 3) mapped to preprocessed coords
        types = [t for _, t in lesions]
    for c in cents:
        if not lesions:
            weights.append(1.0)
            continue
        d = np.linalg.norm(cogs - c[None, :], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= max_match_dist:
            weights.append(boost.get(types[j], 1.0))
            dists.append(float(d[j]))
            matched_types.append(types[j])
        else:
            weights.append(1.0)
    med = float(np.median(dists)) if dists else float("inf")
    stats = {
        "n_centroids": len(cents),
        "n_matched": len(dists),
        "median_match_dist": med,
        "matched_types": matched_types,
    }
    return weights, stats
