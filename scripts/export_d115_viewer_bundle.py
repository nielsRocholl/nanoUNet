#!/usr/bin/env python3
"""Build Dataset115 viewer bundle: native inputs/targets + warped preds in registered layout.

After nanounet_predict_preprocessed, copies native FU/BL scans and union-click JSONs into
inputsTsFU/inputsTsBL, copies GT into targetsTs*, and warps preprocessed preds back to
scanner space into predsTsFU. Output mirrors the registered-dataset folder convention.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, load_json, load_pickle, maybe_mkdir_p

from nanounet.common import config_table, cprint, nano_header, nano_progress
from nanounet.infer.export import export_preprocessed_seg_to_native
from nanounet.plan.plans import Plans

DEFAULT_DATASET = "/nnunet_data/nnUNet_raw/Dataset115_longi_test"
DEFAULT_PRED = "/nnunet_data/NanoUNet_preprocessed/Dataset115_longi_test/preds"
DEFAULT_PREP = "/nnunet_data/NanoUNet_preprocessed/Dataset115_longi_test/nnUNetPlans_3d_fullres"
DEFAULT_MODEL = (
    "/nnunet_data/NanoUNet_results/nanounet/"
    "Dataset114_longi_nnUNetResEncUNetLPlans_h200_smallpv_f0_finetune_dwb"
)
DEFAULT_REGISTERED = "/nnunet_data/unprocessed-universal-lesion-segmentation-registered-unigradicon"

OUT_DIRS = ("inputsTsFU", "inputsTsBL", "targetsTsFU", "targetsTsBL", "predsTsFU")


def _require_dir(path: str, hint: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Required directory not found: {path}\n"
            f"Expected {hint}.\n"
            f"Fix: pass the correct --flag or create the path first."
        )


def _stems(labels_dir: str, ending: str) -> list[str]:
    return sorted(f[: -len(ending)] for f in os.listdir(labels_dir) if f.endswith(ending))


def _copy(src: str, dst: str, overwrite: bool) -> None:
    if not overwrite and os.path.isfile(dst):
        return
    shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-raw", default=DEFAULT_DATASET)
    ap.add_argument("--pred-dir", default=DEFAULT_PRED)
    ap.add_argument("--preprocessed-dir", default=DEFAULT_PREP)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--registered-root", default=DEFAULT_REGISTERED)
    ap.add_argument("--out", default=None, help="default: <dataset-raw>/viewer_export")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    nano_header("Dataset115 viewer export", color="green")
    out = args.out or join(args.dataset_raw, "viewer_export")
    _require_dir(args.dataset_raw, "nnUNet_raw/Dataset115_longi_test")
    _require_dir(args.pred_dir, "preprocessed preds from nanounet_predict_preprocessed")
    _require_dir(args.preprocessed_dir, "preprocessed data_identifier folder")
    if not os.path.isfile(join(args.model_dir, "plans.json")):
        raise FileNotFoundError(
            f"No plans.json in model dir: {args.model_dir}\n"
            f"Expected a training run directory.\n"
            f"Fix: pass --model-dir <NANOUNET_RESULTS>/nanounet/<run_dir>"
        )
    _require_dir(join(args.registered_root, "targetsTrBL"), "registered targetsTrBL")

    pl = Plans(join(args.model_dir, "plans.json"))
    cm = pl.get_configuration("3d_fullres")
    dj = load_json(join(args.dataset_raw, "dataset.json"))
    end = dj["file_ending"]
    stems = _stems(join(args.dataset_raw, "labelsTr"), end)
    if not stems:
        raise FileNotFoundError(
            f"No labels in {join(args.dataset_raw, 'labelsTr')}.\n"
            f"Fix: point --dataset-raw at a valid nnUNet_raw longi dataset."
        )

    for d in OUT_DIRS:
        maybe_mkdir_p(join(out, d))

    missing_pred, exported_pred, copied = [], 0, 0
    config_table(
        [("dataset_raw", args.dataset_raw, "cli/default"), ("pred_dir", args.pred_dir, "cli/default"),
         ("out", out, "cli/default"), ("cases", len(stems), "labelsTr"),
         ("overwrite", args.overwrite, "cli")],
        title="viewer export",
    )

    with nano_progress(len(stems), "export cases") as advance:
        for stem in stems:
            fu_img = join(args.dataset_raw, "imagesTr", f"{stem}_0000{end}")
            bl_img = join(args.dataset_raw, "imagesTr", f"{stem}_0001{end}")
            fu_clk = join(args.dataset_raw, "clicksTrFU", f"{stem}.json")
            bl_clk = join(args.dataset_raw, "clicksTr", f"{stem}.json")
            fu_gt = join(args.dataset_raw, "labelsTr", f"{stem}{end}")
            bl_gt = join(args.registered_root, "targetsTrBL", f"{stem}{end}")
            pred_pp = join(args.pred_dir, f"{stem}{end}")
            props_pkl = join(args.preprocessed_dir, f"{stem}.pkl")

            for path in (fu_img, bl_img, fu_clk, bl_clk, fu_gt, bl_gt):
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Missing source file: {path}\n"
                        f"Expected complete Dataset115 raw + registered BL targets.\n"
                        f"Fix: verify --dataset-raw and --registered-root."
                    )

            _copy(fu_img, join(out, "inputsTsFU", f"{stem}{end}"), args.overwrite)
            _copy(fu_clk, join(out, "inputsTsFU", f"{stem}.json"), args.overwrite)
            _copy(bl_img, join(out, "inputsTsBL", f"{stem}{end}"), args.overwrite)
            _copy(bl_clk, join(out, "inputsTsBL", f"{stem}.json"), args.overwrite)
            _copy(fu_gt, join(out, "targetsTsFU", f"{stem}{end}"), args.overwrite)
            _copy(bl_gt, join(out, "targetsTsBL", f"{stem}{end}"), args.overwrite)
            copied += 5

            pred_out = join(out, "predsTsFU", f"{stem}{end}")
            if not os.path.isfile(pred_pp):
                missing_pred.append(stem)
            elif args.overwrite or not os.path.isfile(pred_out):
                seg = sitk.GetArrayFromImage(sitk.ReadImage(pred_pp))
                props = load_pickle(props_pkl)
                export_preprocessed_seg_to_native(seg, props, cm, pl, dj, pred_out)
                exported_pred += 1
            advance()

    cprint(
        f"[green]done — {len(stems)} case(s), {copied} native files copied, "
        f"{exported_pred} pred(s) warped → {out}[/green]"
    )
    cprint("[dim]viewer: load inputsTsFU + targetsTsFU + predsTsFU (same native grid)[/dim]")
    if missing_pred:
        cprint(f"[yellow]{len(missing_pred)} case(s) missing preds (inference incomplete): {missing_pred[:5]}{'...' if len(missing_pred) > 5 else ''}[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
