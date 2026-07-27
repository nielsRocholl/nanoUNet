"""Measure propagated-click registration error against true follow-up lesion centres.
Dataset-specific side script (not part of the core nanoUNet pipeline). Compares two
registration backends -- the original pipeline's `cog_propagated` and the unigradICON
derivative's `bl_click` -- against ground-truth `cog_fu` centroids, converts the offset
into resampled-voxel space (native axis 2 is through-plane and maps to resampled z; axes
1,0 map to y,x), drops lesion- and case-level outliers, bins by lesion size, and writes
an offset table that training samples from to simulate registration noise.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from rich.table import Table
from nanounet.common import cprint, nano_header, nano_progress
SIZE_BINS_MM = [[0, 5], [5, 10], [10, 20], [20, 30], [30, 50], [50, 1e9]]


def parse_pt(cell: object) -> np.ndarray | None:
    if pd.isna(cell):
        return None
    return np.array([float(x) for x in str(cell).split()], dtype=np.float64)
def load_original_records(meta_dir: Path) -> list[dict]:
    records = []
    for f in sorted(meta_dir.glob("*.csv")):
        pid = f.stem
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            fu, prop = parse_pt(row["cog_fu"]), parse_pt(row["cog_propagated"])
            if fu is None or prop is None:
                continue
            records.append(dict(patient_id=pid, img_id_fu=int(row["img_id_fu"]), lesion_id=int(row["lesion_id"]),
                                 cog_fu=fu, propagated=prop, volume_fu=float(row["volume_fu"])))
    return records


def load_unigrad_records(deriv_dir: Path) -> list[dict]:
    records = []
    meta_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for split in ("train", "test"):
        for f in sorted((deriv_dir / split / "lesions").glob("*.json")):
            pid = f.stem.rsplit("_", 1)[0]
            for les in json.loads(f.read_text())["lesions"]:
                # only the clean warp result is a real measurement; any other fill_source['bl']
                # value carries a suffix marking a synthesised or fallback-filled point
                if (les.get("fill_source") or {}).get("bl") != "warped_bl":
                    continue
                fu, prop = les.get("cog_fu"), les.get("bl_click")
                if fu is None or prop is None:
                    continue
                key = (split, pid)
                if key not in meta_cache:
                    meta_cache[key] = pd.read_csv(deriv_dir / split / "meta" / f"{pid}.csv").set_index("lesion_id")
                vol_fu = float(meta_cache[key].loc[les["lesion_id"], "volume_fu"])
                records.append(dict(patient_id=pid, img_id_fu=int(les["img_id_fu"]), lesion_id=int(les["lesion_id"]),
                                     cog_fu=np.array(fu), propagated=np.array(prop), volume_fu=vol_fu))
    return records


def size_bin(diameter_mm: float) -> int:
    for i, (lo, hi) in enumerate(SIZE_BINS_MM):
        if lo <= diameter_mm < hi:
            return i
    return len(SIZE_BINS_MM) - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--longi-root", default="/nnunet_data/Longitudinal-CT")
    ap.add_argument("--out", default="/nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json")
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.25, 0.781, 0.789], metavar=("Z", "Y", "X"))
    ap.add_argument("--max-lesion-offset-vox", type=float, default=100.0)
    ap.add_argument("--max-case-median-vox", type=float, default=20.0)
    ap.add_argument("--min-per-bin", type=int, default=30)
    args = ap.parse_args()
    nano_header("measure_registration_error: " + " ".join(sys.argv[1:]))
    root = Path(args.longi_root)
    meta_dir, targets_dir = root / "meta", root / "targetsTrFU"
    deriv_dir = root / "derivatives" / "unigrad-icon-registration"
    out_path = Path(args.out)
    if not root.is_dir():
        raise FileNotFoundError(f"--longi-root {root} does not exist.\nFix: pass the Longitudinal-CT dataset root.")
    if not meta_dir.is_dir() or not any(meta_dir.glob("*.csv")):
        raise FileNotFoundError(f"No meta CSVs found under {meta_dir}.\nFix: check --longi-root points at Longitudinal-CT.")
    if not targets_dir.is_dir() or not any(targets_dir.glob("*.nii.gz")):
        raise FileNotFoundError(f"No follow-up label volumes under {targets_dir}.\nFix: check the targetsTrFU folder is populated.")
    if not deriv_dir.is_dir():
        raise FileNotFoundError(f"unigradICON derivative not found at {deriv_dir}.\nFix: run the unigradICON registration step first.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orig = load_original_records(meta_dir)
    uni = load_unigrad_records(deriv_dir)
    cprint(f"loaded {len(orig)} original-backend and {len(uni)} unigradICON-backend candidate lesions")

    by_vol: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in orig:
        by_vol[(r["patient_id"], r["img_id_fu"])].append(dict(r, backend="original"))
    for r in uni:
        by_vol[(r["patient_id"], r["img_id_fu"])].append(dict(r, backend="unigradicon"))

    nii_files = sorted(targets_dir.glob("*.nii.gz"))
    zooms = np.array([nib.load(f).header.get_zooms()[:3] for f in nii_files])
    if not (np.median(zooms[:, 2]) > np.median(zooms[:, 0]) and np.median(zooms[:, 2]) > np.median(zooms[:, 1])):
        raise AssertionError(
            f"Expected native axis 2 (through-plane) to have the largest median zoom, got medians {np.median(zooms, axis=0)}.\n"
            f"The offset axis remap (native 2->z, 1->y, 0->x) assumed by this script would be wrong.\n"
            f"Fix: inspect targetsTrFU header zooms and correct the remap in this script before proceeding."
        )

    fu_hits = fu_total = 0
    with nano_progress(len(nii_files), "reading follow-up volumes") as advance:
        for f in nii_files:
            pid, imgid = f.stem.split(".")[0].rsplit("_", 1)
            recs = by_vol.get((pid, int(imgid)), [])
            img = nib.load(f)
            zoom, shape = np.array(img.header.get_zooms()[:3]), img.shape
            data = np.asanyarray(img.dataobj)
            for r in recs:
                offset_mm = (r["propagated"] - r["cog_fu"]) * zoom
                r["offset_vox_zyx"] = np.array([offset_mm[2] / args.spacing[0], offset_mm[1] / args.spacing[1],
                                                 offset_mm[0] / args.spacing[2]])
                r["mag_vox"] = float(np.linalg.norm(r["offset_vox_zyx"]))
                r["mag_mm"] = float(np.linalg.norm(offset_mm))
                r["diameter_mm"] = 2 * (3 * r["volume_fu"] / (4 * np.pi)) ** (1 / 3)
                idx = tuple(np.clip(np.round(r["propagated"]).astype(int), 0, np.array(shape) - 1))
                r["in_lesion"] = bool(data[idx] == r["lesion_id"])
            # ground-truth centre accuracy is backend-independent; check it once per volume
            for _, row in pd.read_csv(meta_dir / f"{pid}.csv").iterrows():
                fu = parse_pt(row["cog_fu"])
                if fu is None or int(row["img_id_fu"]) != int(imgid):
                    continue
                fu_total += 1
                idx = tuple(np.clip(np.round(fu).astype(int), 0, np.array(shape) - 1))
                fu_hits += bool(data[idx] == int(row["lesion_id"]))
            advance()

    results, excluded = {}, {}
    for backend in ("original", "unigradicon"):
        recs = [r for rs in by_vol.values() for r in rs if r["backend"] == backend]
        cprint(f"[{backend}] pooled median offset before outlier removal: "
               f"{np.median([r['mag_mm'] for r in recs]):.2f} mm over {len(recs)} lesions")
        survivors = [r for r in recs if r["mag_vox"] <= args.max_lesion_offset_vox]
        by_patient: dict[str, list[dict]] = defaultdict(list)
        for r in survivors:
            by_patient[r["patient_id"]].append(r)
        failed_patients = sorted(pid for pid, rs in by_patient.items()
                                  if np.median([r["mag_vox"] for r in rs]) > args.max_case_median_vox)
        results[backend] = [r for r in survivors if r["patient_id"] not in failed_patients]
        excluded[backend] = dict(case_level_failure_patients=failed_patients,
                                  lesion_level_outliers=len(recs) - len(survivors))
    bins_out: dict[str, list] = {}
    n_per_bin: dict[str, list[int]] = {}
    table = Table(title="registration error by backend / size bin")
    for col in ("backend", "size bin (mm)", "n", "med vox", "p90 vox", "p95 vox", "med mm", "p90 mm", "p95 mm", "in-lesion %"):
        table.add_column(col)
    for backend in ("original", "unigradicon"):
        binned: list[list[dict]] = [[] for _ in SIZE_BINS_MM]
        for r in results[backend]:
            binned[size_bin(r["diameter_mm"])].append(r)
        bins_out[backend] = [[list(np.round(r["offset_vox_zyx"], 4)) for r in b] for b in binned]
        n_per_bin[backend] = [len(b) for b in binned]
        for (lo, hi), b in zip(SIZE_BINS_MM, binned):
            if not b:
                table.add_row(backend, f"{lo:g}-{hi:g}", "0", "-", "-", "-", "-", "-", "-", "-")
                continue
            vox, mm = np.array([r["mag_vox"] for r in b]), np.array([r["mag_mm"] for r in b])
            table.add_row(backend, f"{lo:g}-{hi:g}", str(len(b)),
                          f"{np.median(vox):.2f}", f"{np.percentile(vox, 90):.2f}", f"{np.percentile(vox, 95):.2f}",
                          f"{np.median(mm):.2f}", f"{np.percentile(mm, 90):.2f}", f"{np.percentile(mm, 95):.2f}",
                          f"{100*np.mean([r['in_lesion'] for r in b]):.1f}")
    cprint(table)
    bad_bins = [(bk, i) for bk in ("original", "unigradicon") for i, c in enumerate(n_per_bin[bk]) if c < args.min_per_bin]
    if bad_bins:
        detail = "; ".join(f"{bk} bin {SIZE_BINS_MM[i]}mm has {n_per_bin[bk][i]}" for bk, i in bad_bins)
        raise ValueError(f"Fewer than --min-per-bin={args.min_per_bin} offsets in: {detail}.\n"
                          f"Fix: lower --min-per-bin, merge size bins upstream, or gather more registered cases before rerunning.")

    payload = dict(frame="resampled_voxels_zyx", spacing_zyx=args.spacing, size_bins_mm=SIZE_BINS_MM,
                   backends={bk: dict(offsets_zyx=bins_out[bk]) for bk in ("original", "unigradicon")},
                   excluded=excluded,
                   provenance=dict(generated_utc=datetime.now(timezone.utc).isoformat(), command=" ".join(sys.argv),
                                    longi_root=str(root), n_per_bin=n_per_bin))
    out_path.write_text(json.dumps(payload, indent=2))

    cprint("\nexcluded (lesion-level outliers / case-level failure patients): " + ", ".join(
        f"{bk}: {excluded[bk]['lesion_level_outliers']} / {len(excluded[bk]['case_level_failure_patients'])}"
        for bk in ("original", "unigradicon")))
    cprint(f"true follow-up centre inside own lesion: {fu_hits}/{fu_total} = {100*fu_hits/fu_total:.1f}%")
    cprint(f"wrote {out_path}")
    cprint("next: use this table when sampling propagation offsets during training / preprocessing.")


if __name__ == "__main__":
    main()