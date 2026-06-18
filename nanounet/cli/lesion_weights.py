"""Offline build of per-centroid sampling weights for d013 hard-type oversampling.

Forward-maps meta-CSV lesion cogs into preprocessed voxel space, nearest-matches each
preprocessed centroid to a lesion type, and writes <id>_weights.json next to the case.
Crashes loudly if the overall median match distance exceeds the gate (wrong cog axis order).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nanounet.common import cprint, nano_header, preprocessed_dir
from nanounet.data.blosc2_dataset import Blosc2Folder, case_spatial_shape, load_case_properties
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.lesion_types import (
    HARD_TYPE_BOOST,
    build_case_weights,
    case_to_csv,
    cog_to_preprocessed,
    load_lesions,
)
from nanounet.plan.plans import Plans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--meta-dir", required=True, help="folder of <hash>.csv lesion-type files")
    ap.add_argument("--only-prefix", default="d013_")
    ap.add_argument("--cog-axis-order", choices=("xyz", "zyx"), default="xyz")
    ap.add_argument("--max-match-dist", type=float, default=10.0, help="voxels")
    ap.add_argument("--max-median-dist", type=float, default=8.0, help="sanity gate on overall median")
    args = ap.parse_args()

    ds = convert_id_to_dataset_name(args.dataset_id)
    nano_header(f"nanoUNet lesion-weights  {ds}  prefix {args.only_prefix}", color="green")
    pp = preprocessed_dir()
    plans_path = join(pp, ds, args.plans + ".json")
    pm = Plans(plans_path)
    tf = pm.transpose_forward
    cm = pm.get_configuration("3d_fullres")
    case_dir = join(pp, ds, cm.data_identifier)

    ids = [i for i in Blosc2Folder.get_identifiers(case_dir) if i.startswith(args.only_prefix)]
    assert ids, f"no cases with prefix {args.only_prefix} in {case_dir}"

    n_centroids = 0
    n_matched = 0
    per_case_med: list[float] = []
    type_counts: Counter = Counter()
    for cid in ids:
        props = load_case_properties(case_dir, cid)
        pre_shape = case_spatial_shape(case_dir, cid)
        bbox = props["bbox_used_for_cropping"]
        shape_after_crop = props["shape_after_cropping_and_before_resampling"]

        hash_, tp = case_to_csv(cid)
        csv_path = join(args.meta_dir, hash_ + ".csv")
        assert isfile(csv_path), csv_path  # required input, no fallback (R12)
        lesions = load_lesions(csv_path, tp)
        mapped = [
            (cog_to_preprocessed(cog, tf, bbox, shape_after_crop, pre_shape, args.cog_axis_order), t)
            for cog, t in lesions
        ]

        weights, stats = build_case_weights(props["centroids_zyx"], mapped, HARD_TYPE_BOOST, args.max_match_dist)
        with open(join(case_dir, cid + "_weights.json"), "w", encoding="utf-8") as f:
            json.dump({"centroid_weights": weights}, f)

        n_centroids += stats["n_centroids"]
        n_matched += stats["n_matched"]
        type_counts.update(stats["matched_types"])
        if stats["n_centroids"] > 0:
            per_case_med.append(stats["median_match_dist"])

    overall_med = float(np.median([m for m in per_case_med if np.isfinite(m)])) if per_case_med else float("inf")
    pct = 100.0 * n_matched / n_centroids if n_centroids else 0.0
    cprint(f"cases:            {len(ids)}")
    cprint(f"centroids:        {n_centroids}")
    cprint(f"matched:          {n_matched} ({pct:.1f}%)")
    cprint(f"overall med dist: {overall_med:.2f} vox")
    for t, c in type_counts.most_common():
        cprint(f"  {t}: {c}")
    # Wrong axis order leaves matches far away; crash so the user re-runs with --cog-axis-order zyx.
    assert overall_med <= args.max_median_dist, "cog->preprocessed mapping looks wrong; check --cog-axis-order"
