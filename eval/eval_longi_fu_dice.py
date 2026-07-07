#!/usr/bin/env python3
"""Dice between longitudinal-model FU predictions and FU ground truth.

Reports:
  - per-case whole-volume (micro) Dice
  - per-lesion Dice (local bbox+margin crop around each GT lesion instance,
    so one lesion's score isn't polluted by unrelated foreground elsewhere
    in the volume)
  - macro Dice = mean per-lesion Dice (every lesion counts once, regardless
    of size, so large lesions can't drown out small ones)
  - the same, stratified by lesion_type (from the per-patient meta CSV)

Ground-truth segmentations are instance-labeled (voxel value == lesion_id);
predictions are binary (bg/fg). Lesion identity + type come from
meta/<patient>.csv, filtered to the rows whose img_id_fu matches this
stem's image-id suffix and whose cog_fu is non-empty (i.e. actually visible
in this FU scan).

Usage:
    python3 eval_longi_fu_dice.py \
        --pred-dir  /nnunet_data/.../predictions/longi_dwb_best134 \
        --gt-dir    /nnunet_data/.../targetsTrFU \
        --meta-dir  /nnunet_data/.../meta \
        [--margin 10] [--out-csv per_lesion_dice.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import SimpleITK as sitk


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    denom = float(a.sum() + b.sum())
    if denom == 0.0:
        return 1.0
    return 2.0 * inter / denom


def load_lesions_for_stem(meta_dir: str, pid: str, img_id: str) -> dict[int, str]:
    """lesion_id -> lesion_type for lesions visible in this FU image."""
    path = os.path.join(meta_dir, f"{pid}.csv")
    if not os.path.isfile(path):
        return {}
    out: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_id_fu = row.get("img_id_fu", "").strip()
            if not img_id_fu or int(img_id_fu) != int(img_id):
                continue
            if not row.get("cog_fu", "").strip():
                continue  # not present in this FU scan
            out[int(row["lesion_id"])] = row.get("lesion_type", "unknown").strip() or "unknown"
    return out


def per_lesion_dice(pred: np.ndarray, gt_instances: np.ndarray, lesion_id: int, margin: int) -> float | None:
    """Local bbox+margin crop around one GT instance. Prediction voxels that fall on a
    *different* GT lesion instance are excluded from this lesion's false-positive count —
    otherwise a correctly-segmented neighbour (common for tightly clustered small lesions,
    e.g. skin/soft-tissue nodules) would wrongly deflate this lesion's score."""
    mask = gt_instances == lesion_id
    if not mask.any():
        return None
    zs, ys, xs = np.where(mask)
    z0, z1 = max(zs.min() - margin, 0), min(zs.max() + margin + 1, gt_instances.shape[0])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, gt_instances.shape[1])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, gt_instances.shape[2])
    gt_crop = mask[z0:z1, y0:y1, x0:x1]
    gt_instances_crop = gt_instances[z0:z1, y0:y1, x0:x1]
    other_lesion_crop = (gt_instances_crop != 0) & (gt_instances_crop != lesion_id)
    pred_crop = (pred[z0:z1, y0:y1, x0:x1] > 0) & ~other_lesion_crop
    return dice(pred_crop, gt_crop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--margin", type=int, default=10, help="voxel margin around each lesion's bbox")
    ap.add_argument("--out-csv", default=None, help="optional path to dump the per-lesion rows")
    args = ap.parse_args()

    gt_files = sorted(f for f in os.listdir(args.gt_dir) if f.endswith(".nii.gz"))
    if not gt_files:
        sys.exit(f"no GT files found in {args.gt_dir}")

    case_dice: list[tuple[str, float]] = []
    lesion_rows: list[tuple[str, int, str, float]] = []  # stem, lesion_id, type, dice
    n_missing_pred = 0
    n_missing_lesion_voxels = 0

    for fname in gt_files:
        stem = fname[: -len(".nii.gz")]
        pred_path = os.path.join(args.pred_dir, fname)
        if not os.path.isfile(pred_path):
            print(f"[skip] no prediction for {stem}", file=sys.stderr)
            n_missing_pred += 1
            continue
        pid, img_id = stem.rsplit("_", 1)

        gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.gt_dir, fname)))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        if gt.shape != pred.shape:
            print(f"[skip] shape mismatch for {stem}: gt {gt.shape} vs pred {pred.shape}", file=sys.stderr)
            continue

        case_dice.append((stem, dice(pred > 0, gt > 0)))

        lesions = load_lesions_for_stem(args.meta_dir, pid, img_id)
        for lesion_id, lesion_type in lesions.items():
            d = per_lesion_dice(pred, gt, lesion_id, args.margin)
            if d is None:
                n_missing_lesion_voxels += 1
                print(f"[warn] {stem}: lesion {lesion_id} ({lesion_type}) has no GT voxels, skipped", file=sys.stderr)
                continue
            lesion_rows.append((stem, lesion_id, lesion_type, d))

    if not lesion_rows:
        sys.exit("no lesions scored — check --meta-dir / stem naming")

    n_cases = len(case_dice)
    mean_case_dice = sum(d for _, d in case_dice) / n_cases if n_cases else float("nan")
    macro_dice = sum(d for *_, d in lesion_rows) / len(lesion_rows)

    by_type: dict[str, list[float]] = defaultdict(list)
    for _, _, ltype, d in lesion_rows:
        by_type[ltype].append(d)

    print(f"\ncases scored:            {n_cases}  (missing predictions: {n_missing_pred})")
    print(f"lesions scored:           {len(lesion_rows)}  (missing GT voxels: {n_missing_lesion_voxels})")
    print(f"\nmean per-case whole-volume Dice (micro): {mean_case_dice:.4f}")
    print(f"macro (per-lesion) Dice, all types:       {macro_dice:.4f}")

    print(f"\n{'lesion_type':<20} {'n':>5}  {'mean_dice':>9}")
    for ltype, ds in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"{ltype:<20} {len(ds):>5}  {sum(ds) / len(ds):>9.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "lesion_id", "lesion_type", "dice"])
            w.writerows(lesion_rows)
        print(f"\nper-lesion rows written to {args.out_csv}")


if __name__ == "__main__":
    main()
