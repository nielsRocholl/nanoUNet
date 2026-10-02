"""Foundation mode planner: nnFoundationCNN topology + z-only resampling, no planner search.

Replaces `run_plan` (nanounet_preprocess default; `--no-foundation` restores it). The topology and
192^3 patch are fixed by the pretrained checkpoint (D9), so the plan is written from constants plus
the fingerprint's spacings/shapes. Only the thickest axis is resampled to `z_target` mm; the other
two keep each case's native spacing and grid, so the REAL spacing is per case
(`properties["spacing_after_resampling"]`); `spacing` in the plan is a nominal median for display.
"""

from __future__ import annotations

import glob
import os
import shutil

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json

from core.ui import cprint
from nanounet.common import preprocessed_dir, raw_dir
from nanounet.data.store.io import reader_writer_class_from_dataset
from nanounet.data.volume.resampling import compute_new_shape, resample_data_or_seg_to_shape
from nanounet.model.foundation import ARCH_KWARGS, KW_REQUIRES_IMPORT, NET_CLASS, PATCH_SIZE
from nanounet.model.network import estimate_conv_feature_map_size
from nanounet.plan.dataset.ids import convert_id_to_dataset_name, get_filenames_of_train_images_and_targets
from nanounet.plan.resenc.planner import _save_plans
from nanounet.plan.resenc.planner_resenc import MIN_BATCH, PRESETS, REF_BS_3D

THICK_AXIS_RATIO = 1.25  # A3: a thick axis must exceed 1.25x axis 0 to be picked over axis 0
PRESET = "nnUNetPlannerResEncL"


def default_plans_name(z_target: float) -> str:
    return f"nnFoundationCNN_z{z_target:.1f}".replace(".", "p")


def zonly_target_spacing(sp, z_target: float, ratio: float = THICK_AXIS_RATIO) -> tuple[list[float], int]:
    """The one z-only rule (preprocessing, plan, docs): resample the thickest axis to z_target."""
    ax = int(np.argmax(sp))
    ax = ax if sp[ax] > ratio * sp[0] else 0
    t = [float(x) for x in sp]
    t[ax] = float(z_target)
    return t, ax


def case_target_spacing(cm, o_sp) -> tuple[list[float], int]:
    """(target spacing, resampled axis) for one case in plan axis order. Median mode (every non-foundation
    plan) returns the plan's spacing and axis -1 (all axes); z_only mode returns the per-case thick-axis rule."""
    c = cm.configuration
    if c.get("spacing_mode", "median") == "z_only":
        return zonly_target_spacing(o_sp, c["z_target_mm"], c["thick_axis_ratio"])
    t = list(cm.spacing)
    return (t if len(t) == len(o_sp) else [o_sp[0], *t]), -1


def export_spacing(cm, props: dict, sp_t: list[float]) -> list[float]:
    """Spacing of the plan-space prediction grid, for resampling a prediction back to the native grid."""
    if cm.configuration.get("spacing_mode", "median") != "z_only":
        return cm.spacing if len(cm.spacing) == len(sp_t) else [sp_t[0], *cm.spacing]
    if "spacing_after_resampling" not in props:
        raise SystemExit(
            "This case's properties have no 'spacing_after_resampling', which z-only plans need to map predictions back to the native grid.\n"
            "Expected the sidecar written by a z-only preprocess (every case records the spacing it was resampled to).\n"
            "Fix: re-run nanounet_preprocess for this plan (the case was preprocessed before spacing_after_resampling existed)"
        )
    return props["spacing_after_resampling"]


def _batch_size(gpu_mem_gb: float, n_in: int, n_out: int) -> int:
    preset = PRESETS[PRESET]
    est = estimate_conv_feature_map_size(tuple(PATCH_SIZE), n_in, n_out, NET_CLASS, ARCH_KWARGS, KW_REQUIRES_IMPORT)
    ref = preset.reference_val_3d * (gpu_mem_gb / preset.reference_val_corresp_gb)
    bs = max(round(ref / est * REF_BS_3D), MIN_BATCH)
    return bs - bs % 2  # A4: --prompts-per-patch 2 needs an even batch


def _check_identifier(pf: str, plans_name: str, ident: str) -> None:
    for f in glob.glob(join(pf, "*Plans*.json")) + glob.glob(join(pf, "nnFoundation*.json")):
        if os.path.basename(f) == plans_name + ".json":
            continue
        for name, c in load_json(f)["configurations"].items():
            if c["data_identifier"] == ident:
                raise SystemExit(
                    f"data_identifier {ident!r} is already used by {os.path.basename(f)} ({name}).\n"
                    f"Expected a unique folder per plans variant: preprocessing wipes the data folder unless --resume.\n"
                    f"Fix: nanounet_preprocess -d <id> --plans-name <new name>"
                )


def run_foundation_plan(dataset_id: int, plans_name: str | None, z_target: float, gpu_mem_gb: float | None, info: dict) -> str:
    dn = convert_id_to_dataset_name(dataset_id)
    rf, pf = join(raw_dir(), dn), join(preprocessed_dir(), dn)
    if not isfile(join(pf, "dataset_fingerprint.json")):
        raise SystemExit(
            f"No dataset fingerprint at {pf}.\n"
            f"The foundation plan reads spacings and shapes from dataset_fingerprint.json.\n"
            f"Fix: nanounet_preprocess -d {dataset_id}   (without --skip-fingerprint; see nanounet/docs/steps/preprocess.md)"
        )
    ident = plans_name or default_plans_name(z_target)
    data_ident = f"{ident}_3d_fullres"
    _check_identifier(pf, ident, data_ident)
    dj, fp = load_json(join(rf, "dataset.json")), load_json(join(pf, "dataset_fingerprint.json"))
    n_in = len(dj["channel_names"] if "channel_names" in dj else dj["modality"])
    vram = float(gpu_mem_gb if gpu_mem_gb is not None else PRESETS[PRESET].default_vram_gb)
    sps, shs = fp["spacings"], fp["shapes_after_crop"]
    targets = [zonly_target_spacing(sp, z_target)[0] for sp in sps]
    med_shape = np.median(np.stack([compute_new_shape(sh, sp, t) for sh, sp, t in zip(shs, sps, targets)]), 0)
    ds = get_filenames_of_train_images_and_targets(rf, dj)
    rw = reader_writer_class_from_dataset(dj, ds[next(iter(ds))]["images"][0], verbose=False).__name__
    dst = join(pf, "dataset.json")
    if not isfile(dst):
        shutil.copyfile(join(rf, "dataset.json"), dst)
    mfn = resample_data_or_seg_to_shape.__name__
    cfg = {
        "data_identifier": data_ident,
        "preprocessor_name": "DefaultPreprocessor",
        "batch_size": _batch_size(vram, n_in, len(dj["labels"])),
        "patch_size": list(PATCH_SIZE),
        "median_image_size_in_voxels": [float(x) for x in med_shape],
        "spacing": [float(x) for x in np.median(targets, 0)],
        "spacing_mode": "z_only",
        "z_target_mm": float(z_target),
        "thick_axis_ratio": THICK_AXIS_RATIO,
        "normalization_schemes": ["ZScoreNormalization"] * n_in,
        "use_mask_for_norm": [False] * n_in,
        "resampling_fn_data": mfn,
        "resampling_fn_seg": mfn,
        "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 1, "force_separate_z": None},  # A2
        "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
        "resampling_fn_probabilities": mfn,
        "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
        "architecture": {"network_class_name": NET_CLASS, "arch_kwargs": ARCH_KWARGS, "_kw_requires_import": KW_REQUIRES_IMPORT},
        "batch_dice": False,
    }
    plans = {
        "dataset_name": dn,
        "plans_name": ident,
        "original_median_spacing_after_transp": [float(x) for x in np.median(sps, 0)],
        "original_median_shape_after_transp": [int(round(x)) for x in np.median(np.stack(shs), 0)],
        "image_reader_writer": rw,
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {"3d_fullres": cfg},
        "experiment_planner_used": "nnFoundationPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": fp["foreground_intensity_properties_per_channel"],
        "pretrain_info": info,
    }
    _save_plans(pf, ident, plans)
    cprint(f"[bold green]✓ wrote {ident}.json[/bold green]  (batch {cfg['batch_size']}, patch {PATCH_SIZE}, z {z_target} mm)")
    return ident
