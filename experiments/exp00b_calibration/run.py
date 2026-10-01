# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
"""exp00b - Calibration of the prompt noise  (paper: Sec. "Calibrated prompt noise" and sec:calibration; plan Sec. 6 exp00b)

QUESTION   How was the shipped registration-error table built, can it be reproduced from the data, and how much of it comes from
           the held-out 60 patients?
WHY        Gives `sec:calibration` a home: which registrations, how many lesions, how the size bins are cut, what the residuals are.
           The table is accepted as shipped (owner decision); this run verifies and documents it and never rewrites it.
DATA       `Longitudinal-CT/derivatives/registration_error_table.json` (backends `original` and `unigradicon`), the meta CSVs
           (`original`: cog_propagated vs cog_fu), `derivatives/unigrad-icon-registration/{train,test}/lesions/*.json`
           (`unigradicon`: bl_click = propagated point, fu_click = true point), FU image headers (native spacing).
METHOD     1. Residual per lesion present at both timepoints (UNCHANGED + MERGING) = propagated point - true FU point, in native voxels,
              x native spacing -> mm, / table spacing (zyx 1.25/0.781/0.789) -> "resampled voxels" (the table's frame).
           2. Size = equivalent-sphere diameter of the FU lesion from `volume_fu` (mm^3), cut with the table's `size_bins_mm`
              (the 6-line binning is replicated here because nanounet's `_bin_index` is private).
           3. Apply the table's own `excluded.case_level_failure_patients`. Its `lesion_level_outliers` are counts only (the lost
              script's rule is not in the file), so they cannot be applied: the recomputed rows that the shipped table lacks are
              listed, not filtered by a rule invented here.
           4. Reproduction check (reported, never tuned): recomputed `n_per_bin` vs the shipped one, and every shipped offset triple is
              looked up among the recomputed rows of the same bin (tolerance 2e-3 vox, the table is rounded to 4 decimals).
           5. Leakage: shipped rows that come from the held-out 60 patients (the segmenter was already trained with the table).
OUTPUT     results.json `reproduction`, `per_bin`, `overall`, `leakage`, `frame_check`, `per_lesion` (every residual, flags
           `excluded_case_level`, `in_shipped_table`, `in_holdout60`), `per_bin` stats per row set (shipped rows / recomputed rows).
COMMAND    python -m experiments.exp00b_calibration.run --tag paper_v1
DEPENDS ON experiments/common.py only.
RUNTIME    About 1 min on CPU (336 header reads).
CAVEATS    The plan named `unigrad-icon-registration/*/meta` for the unigradicon residuals: those CSVs are byte-identical copies of the
           original meta (checked), so the uniGradICON points are taken from `lesions/*.json`. The table's frame spacing (z 1.25 mm) differs
           from the segmenter plan spacing (z 2.5 mm); see `frame_check`.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.ui import cprint, nano_progress
from experiments.common import HOLDOUT_CSV, LONGI_ROOT, abort_if, add_common_args, missing_paths, problem, start_run
from nanounet.data.patch.error_table import volume_vox_to_diam_mm

EXP = "exp00b_calibration"
TOL = 2e-3  # vox; the shipped table is rounded to 4 decimals
CORPUS_PLAN = Path("/nnunet_data/NanoUNet_preprocessed/Dataset900_Merged/nnUNetResEncUNetLPlans_h200_smallpv.json")
KINDS = ("UNCHANGED", "MERGING")


def bin_of(diam_mm: float, bins: list) -> int:
    """Size bin of an equivalent-sphere diameter: lo <= d < hi; above the last lower edge = last bin (same rule as nanounet's private _bin_index)."""
    for i, (lo, hi) in enumerate(bins):
        if lo <= diam_mm < hi:
            return i
    return len(bins) - 1 if diam_mm >= bins[-1][0] else 0


def fu_spacings(root: Path, keys: list[tuple[str, int]], workers: int) -> dict:
    def one(k: tuple[str, int]) -> tuple:
        return nibabel.load(str(root / "inputsTrFU" / f"{k[0]}_{k[1]:02d}.nii.gz")).header.get_zooms()[:3]  # header only; (x, y, z) mm

    with nano_progress(len(keys), "FU image headers") as advance, ThreadPoolExecutor(workers) as ex:
        out = {}
        for k, s in zip(keys, ex.map(one, keys)):
            out[k] = np.array(s)
            advance()
    return out


def residual_rows(root: Path, backend: str, pids: list[str]) -> list[dict]:
    """Per lesion present at both timepoints: native-voxel residual (x, y, z), FU volume (mm^3), ids."""
    rows = []
    for pid in pids:
        meta = pd.read_csv(root / "meta" / f"{pid}.csv")
        if backend == "original":
            for r in meta[meta["topology_class"].isin(KINDS)].itertuples():
                if isinstance(r.cog_propagated, str) and isinstance(r.cog_fu, str):
                    d = np.array([float(x) for x in r.cog_propagated.split()]) - np.array([float(x) for x in r.cog_fu.split()])
                    rows.append({"patient": pid, "lesion_id": int(r.lesion_id), "topology": r.topology_class, "fu_index": int(r.img_id_fu), "vol_fu_mm3": float(r.volume_fu), "delta_native_xyz": d})
        else:
            vol = meta.set_index("lesion_id")["volume_fu"]
            for split in ("train", "test"):
                for f in sorted((root / "derivatives/unigrad-icon-registration" / split / "lesions").glob(f"{pid}_*.json")):
                    for L in json.loads(f.read_text())["lesions"]:
                        if L["topology"] in KINDS and L.get("bl_click") is not None and L.get("fu_click") is not None:
                            d = np.array(L["bl_click"], float) - np.array(L["fu_click"], float)
                            rows.append({"patient": pid, "lesion_id": int(L["lesion_id"]), "topology": L["topology"], "fu_index": int(L["img_id_fu"]), "vol_fu_mm3": float(vol.loc[L["lesion_id"]]), "delta_native_xyz": d})
    return rows


def stats(rows: list[dict]) -> dict:
    mm = np.array([r["delta_mm"] for r in rows])
    v = np.array([[r["dz_vox"], r["dy_vox"], r["dx_vox"]] for r in rows]).reshape(-1, 3)
    q = (lambda p: float(np.percentile(mm, p))) if len(mm) else (lambda p: None)
    return {"n": len(rows), "median_mm": q(50), "p90_mm": q(90), "p95_mm": q(95), "max_mm": float(mm.max()) if len(mm) else None,
            **{f"sd_{a}_vox": float(v[:, i].std()) if len(mm) else None for i, a in enumerate("zyx")}}


def md_table(rows: list[dict], cols: list[str]) -> str:
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, gpu=False)
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="Longitudinal-CT root (meta/, inputsTrFU/, derivatives/)")
    ap.add_argument("--table", type=Path, default=None, help="registration_error_table.json to verify (default: <data-root>/derivatives/registration_error_table.json)")
    ap.add_argument("--holdout-csv", type=Path, default=HOLDOUT_CSV, help="CSV listing the held-out 60 patients (column `patient`)")
    ap.add_argument("--segmenter-plan", type=Path, default=CORPUS_PLAN, help="nnU-Net plans JSON of the segmenter, only read for the frame check (3d_fullres spacing)")
    ap.add_argument("--workers", type=int, default=16, help="threads for the FU header reads")
    args = ap.parse_args()
    table_path = args.table or args.data_root / "derivatives/registration_error_table.json"
    problems = missing_paths({"Longitudinal-CT meta dir": args.data_root / "meta", "registration error table": table_path, "holdout csv": args.holdout_csv, "segmenter plan": args.segmenter_plan,
                              "uniGradICON lesions dir": args.data_root / "derivatives/unigrad-icon-registration/train/lesions"}, "mount /nnunet_data or pass the matching --data-root/--table/--holdout-csv/--segmenter-plan flag")
    if not problems:
        t = json.loads(table_path.read_text())
        if not {"spacing_zyx", "size_bins_mm", "backends", "excluded", "provenance"} <= set(t) or not {"original", "unigradicon"} <= set(t["backends"]):
            problems.append(problem(f"{table_path} is not a registration-error table", "keys spacing_zyx, size_bins_mm, backends{original, unigradicon}, excluded, provenance", "pass the table with --table (default <data-root>/derivatives/registration_error_table.json)"))
    abort_if(problems)
    run = start_run(EXP, ap, args, inputs={"registration error table": table_path, "holdout csv": args.holdout_csv, "segmenter plan": args.segmenter_plan}, paper={"section": "Method > Calibrated prompt noise; sec:calibration", "supports": "how the residual table was measured"})
    table = json.loads(table_path.read_text())
    sp, bins = np.array(table["spacing_zyx"]), table["size_bins_mm"]
    holdout = {ln.strip() for ln in args.holdout_csv.read_text().splitlines()[1:] if ln.strip()}
    pids = sorted(p.stem for p in (args.data_root / "meta").glob("*.csv"))
    if args.limit_patients >= 0:
        pids = pids[: args.limit_patients]
    raw = {b: residual_rows(args.data_root, b, pids) for b in ("original", "unigradicon")}
    spac = fu_spacings(args.data_root, sorted({(r["patient"], r["fu_index"]) for rs in raw.values() for r in rs}), args.workers)
    per_lesion, repro, per_bin, overall, leak = [], [], [], [], []
    for b, rs in raw.items():
        fail, shipped = set(table["excluded"][b]["case_level_failure_patients"]), table["backends"][b]["offsets_zyx"]
        rows = []
        for r in rs:
            d_mm = (r["delta_native_xyz"] * spac[(r["patient"], r["fu_index"])])[::-1]  # (z, y, x) mm
            off = d_mm / sp  # the table's frame (zyx voxels)
            diam = volume_vox_to_diam_mm(r["vol_fu_mm3"] / float(np.prod(sp)), tuple(sp))
            rows.append({"backend": b, "patient": r["patient"], "lesion_id": r["lesion_id"], "topology": r["topology"], "fu_index": r["fu_index"], "size_mm": float(diam), "size_bin": bin_of(diam, bins),
                         "dz_vox": float(off[0]), "dy_vox": float(off[1]), "dx_vox": float(off[2]), "delta_mm": float(np.linalg.norm(d_mm)), "in_holdout60": r["patient"] in holdout,
                         "excluded_case_level": r["patient"] in fail, "in_shipped_table": False})
        for k in range(len(bins)):  # look every shipped triple up among the recomputed, non-excluded rows of the same bin
            mine = [r for r in rows if r["size_bin"] == k and not r["excluded_case_level"]]
            if mine:
                tree = cKDTree(np.array([[r["dz_vox"], r["dy_vox"], r["dx_vox"]] for r in mine]))
                d, i = tree.query(np.array(shipped[k]))
                for j in set(i[d < TOL].tolist()):
                    mine[j]["in_shipped_table"] = True
        per_lesion += rows
        keep = [r for r in rows if not r["excluded_case_level"]]
        for k in range(len(bins)):
            n_re, n_sh = sum(r["size_bin"] == k for r in keep), len(shipped[k])
            n_hit = sum(r["in_shipped_table"] and r["size_bin"] == k for r in keep)
            repro.append({"backend": b, "size_bin": k, "bin_mm": bins[k], "n_recomputed": n_re, "n_shipped_table": n_sh, "n_provenance": table["provenance"]["n_per_bin"][b][k], "diff_recomputed_minus_shipped": n_re - n_sh,
                          "shipped_rows_found_in_recomputed": n_hit, "recomputed_rows_absent_from_table": sum(not r["in_shipped_table"] and r["size_bin"] == k for r in keep)})
            for label, sel in (("shipped_rows", [r for r in keep if r["in_shipped_table"] and r["size_bin"] == k]), ("recomputed_rows", [r for r in keep if r["size_bin"] == k])):
                per_bin.append({"backend": b, "row_set": label, "size_bin": k, "bin_mm": bins[k], **stats(sel)})
        arr = np.array(shipped[0] + [x for k in range(1, len(bins)) for x in shipped[k]]) if shipped else np.zeros((0, 3))
        med_table = float(np.median(np.linalg.norm(arr * sp, axis=1)))
        overall.append({"backend": b, "median_mm_table_offsets": med_table, "n_table_offsets": len(arr), **{f"{lab}_{k}": v for lab, sel in (("recomputed", keep), ("shipped_rows", [r for r in keep if r["in_shipped_table"]])) for k, v in stats(sel).items() if k in ("n", "median_mm", "p95_mm", "max_mm")}})
        sh = [r for r in keep if r["in_shipped_table"]]
        leak.append({"backend": b, "shipped_rows_total": len(sh), "shipped_rows_from_holdout60": sum(r["in_holdout60"] for r in sh), "share": (sum(r["in_holdout60"] for r in sh) / len(sh)) if sh else None,
                     "holdout_patients_contributing": len({r["patient"] for r in sh if r["in_holdout60"]}), "per_bin_from_holdout60": [sum(r["in_holdout60"] and r["size_bin"] == k for r in sh) for k in range(len(bins))]})
    plan_sp = json.loads(args.segmenter_plan.read_text())["configurations"]["3d_fullres"]["spacing"]
    frame = {"table_spacing_zyx": table["spacing_zyx"], "segmenter_plan_spacing_zyx": plan_sp, "z_spacing_ratio_plan_over_table": plan_sp[0] / table["spacing_zyx"][0], "frame": table["frame"],
             "note": "nanounet applies the table offsets to centroids in the preprocessed grid (plan spacing) and computes the size bin with the table spacing; with a z spacing of 2.5 mm against 1.25 mm the z offsets are 2x larger in mm than measured"}
    match = all(r["diff_recomputed_minus_shipped"] == 0 and r["recomputed_rows_absent_from_table"] == 0 for r in repro)
    full = args.limit_patients < 0
    summary = {"table_matches_shipped": bool(match) if full else None, "shipped_rows_reproduced": {b: f"{sum(r['shipped_rows_found_in_recomputed'] for r in repro if r['backend'] == b)}/{sum(r['n_shipped_table'] for r in repro if r['backend'] == b)}" for b in raw},
               "n_per_bin_recomputed": {b: [r["n_recomputed"] for r in repro if r["backend"] == b] for b in raw}, "n_per_bin_shipped": {b: [r["n_shipped_table"] for r in repro if r["backend"] == b] for b in raw},
               "recomputed_rows_absent_from_table": {b: sum(r["recomputed_rows_absent_from_table"] for r in repro if r["backend"] == b) for b in raw},
               "overall_median_mm": {o["backend"]: o["median_mm_table_offsets"] for o in overall}, "holdout60_share_of_table": {x["backend"]: {"rows_from_holdout60": x["shipped_rows_from_holdout60"], "rows_total": x["shipped_rows_total"], "share": x["share"]} for x in leak}, "frame_check": frame}
    notes = ["The shipped table is not modified. Recomputed n_per_bin exceeds the shipped one because the lost script also dropped lesion-level outliers (provenance counts: original 20, unigradicon 119) by a rule not stored in the file; the recomputed rows absent from the table are flagged in per_lesion.in_shipped_table=false and are all large offsets.",
             "unigradicon residuals use derivatives/unigrad-icon-registration/*/lesions/*.json (bl_click - fu_click); the meta/ CSVs there are byte-identical copies of the original meta CSVs.",
             "Bins use the equivalent-sphere diameter of the FU lesion computed from volume_fu (mm^3) with the table's spacing; MERGING lesions use the merged FU lesion's centroid and volume.",
             f"frame_check: table spacing z {table['spacing_zyx'][0]} mm vs segmenter plan spacing z {plan_sp[0]} mm." + (" Smoke run: limited patients, reproduction numbers are not meaningful." if not full else "")]
    cprint("reproduction: " + "; ".join(f"{b}: recomputed {summary['n_per_bin_recomputed'][b]} vs shipped {summary['n_per_bin_shipped'][b]}" for b in raw))
    md = "# exp00b calibration\n\n## Reproduction of n_per_bin\n\n" + md_table(repro, ["backend", "size_bin", "bin_mm", "n_recomputed", "n_shipped_table", "diff_recomputed_minus_shipped", "shipped_rows_found_in_recomputed", "recomputed_rows_absent_from_table"]) \
        + "\n## Per-bin residual statistics\n\n" + md_table([{**x, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()}} for x in per_bin], ["backend", "row_set", "size_bin", "n", "median_mm", "p90_mm", "p95_mm", "max_mm", "sd_z_vox", "sd_y_vox", "sd_x_vox"]) \
        + "\n## Held-out 60 contribution\n\n" + md_table(leak, ["backend", "shipped_rows_total", "shipped_rows_from_holdout60", "share", "holdout_patients_contributing"])
    run.finish(summary, {"reproduction": repro, "per_bin": per_bin, "overall": overall, "leakage": leak, "frame_check": [frame], "per_lesion": per_lesion}, table_md=md, notes=notes, next_cmd=f"cat {run.dir}/table.md",
               definitions={"residual": "propagated point - true FU point, native voxels x native spacing, then / table spacing", "size": "equivalent-sphere diameter of the FU lesion from volume_fu", "match_tolerance_vox": TOL})


if __name__ == "__main__":
    main()
