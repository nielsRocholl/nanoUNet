"""Offline build of per-FU-centroid BL baseline pairing sidecars for two-stream finetune.

Forward-maps meta-CSV cog_bl/cog_fu into preprocessed voxel space, pairs each FU
centroid to its BL case + BL centroid, and writes <id>_baseline.json next to the case.
Crashes loudly if the overall median match distance exceeds the gate (wrong cog axis order).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nanounet.common import cprint, nano_header, nano_progress, preprocessed_dir
from nanounet.data.blosc2_dataset import Blosc2Folder, case_spatial_shape, load_case_properties
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.lesion_types import case_to_csv, load_lesion_pairs
from nanounet.plan.longi_pairs import build_fu_baseline, patient_bl_cases
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
    nano_header(f"nanoUNet longi-pairs  {ds}  prefix {args.only_prefix}", color="green")
    pp = preprocessed_dir()
    pm = Plans(join(pp, ds, args.plans + ".json"))
    tf = pm.transpose_forward
    cm = pm.get_configuration("3d_fullres")
    case_dir = join(pp, ds, cm.data_identifier)

    all_ids = Blosc2Folder.get_identifiers(case_dir)
    fu_ids = [i for i in all_ids if i.startswith(args.only_prefix) and "_FU_img" in i]
    assert fu_ids, f"no FU cases with prefix {args.only_prefix} in {case_dir}"

    n_centroids = 0
    n_paired = 0
    per_case_med: list[float] = []
    with nano_progress(len(fu_ids), "longi-pairs") as advance:
        for fu_id in fu_ids:
            fu_props = load_case_properties(case_dir, fu_id)
            fu_pre_shape = case_spatial_shape(case_dir, fu_id)
            hash_, _ = case_to_csv(fu_id)
            csv_path = join(args.meta_dir, hash_ + ".csv")
            assert isfile(csv_path), csv_path
            pairs_csv = load_lesion_pairs(csv_path)
            bl_cands = []
            for bl_id in patient_bl_cases(all_ids, hash_):
                bl_props = load_case_properties(case_dir, bl_id)
                bl_pre_shape = case_spatial_shape(case_dir, bl_id)
                bl_cands.append((bl_id, bl_props, bl_pre_shape))
            out, stats = build_fu_baseline(
                fu_id,
                fu_props,
                fu_pre_shape,
                bl_cands,
                pairs_csv,
                tf,
                args.cog_axis_order,
                args.max_match_dist,
            )
            with open(join(case_dir, fu_id + "_baseline.json"), "w", encoding="utf-8") as f:
                json.dump(out, f)
            n_centroids += stats["n_centroids"]
            n_paired += stats["n_paired"]
            if stats["n_centroids"] > 0:
                per_case_med.append(stats["median_match_dist"])
            advance(1)

    overall_med = float(np.median([m for m in per_case_med if np.isfinite(m)])) if per_case_med else float("inf")
    pct = 100.0 * n_paired / n_centroids if n_centroids else 0.0
    cprint(f"FU cases:         {len(fu_ids)}")
    cprint(f"centroids:        {n_centroids}")
    cprint(f"paired:           {n_paired} ({pct:.1f}%)")
    cprint(f"overall med dist: {overall_med:.2f} vox")
    assert overall_med <= args.max_median_dist, "cog->preprocessed mapping looks wrong; check --cog-axis-order"
