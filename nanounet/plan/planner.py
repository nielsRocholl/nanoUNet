"""``dataset_fingerprint.json`` + ResEnc preset → ``<plans>.json`` with only ``3d_fullres``."""

from __future__ import annotations

import shutil
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, maybe_mkdir_p, save_json

from nanounet.common import ANISO_THRESHOLD, cprint, preprocessed_dir, raw_dir
from nanounet.data.io import reader_writer_class_from_dataset
from nanounet.data.normalization import normalization_class_for_channel
from nanounet.data.resampling import compute_new_shape
from nanounet.plan.dataset_id import convert_id_to_dataset_name, get_filenames_of_train_images_and_targets
from nanounet.plan.json_export import recursive_fix_for_json_export
from nanounet.plan.planner_resenc import PRESETS, resenc_3d_fullres_plan


def _maybe_copy_splits(raw_folder: str, pre_folder: str) -> None:
    s, t = join(raw_folder, "splits_final.json"), join(pre_folder, "splits_final.json")
    if not isfile(s):
        return
    if not isfile(t):
        shutil.copyfile(s, t)
        return
    a, b = load_json(s), load_json(t)
    for i in range(len(a)):
        assert set(a[i]["train"]) == set(b[i]["train"])
        assert set(a[i]["val"]) == set(b[i]["val"])


def _transpose(suppress: bool, target_spacing: np.ndarray):
    if suppress:
        return [0, 1, 2], [0, 1, 2]
    ax = int(np.argmax(target_spacing))
    rest = [i for i in range(3) if i != ax]
    tf = [ax] + rest
    tb = [int(np.argwhere(np.array(tf) == i)[0][0]) for i in range(3)]
    return tf, tb


def _fullres_spacing(fp: dict, overwrite: Optional[list[float]], aniso: float) -> np.ndarray:
    if overwrite is not None:
        return np.array(overwrite, dtype=float)
    sp = np.vstack(fp["spacings"])
    sizes = fp["shapes_after_crop"]
    target = np.percentile(sp, 50, axis=0)
    tgt_sz = np.percentile(np.vstack(sizes), 50, axis=0)
    w_ax = int(np.argmax(target))
    other = [i for i in range(len(target)) if i != w_ax]
    oth_sp = [target[i] for i in other]
    oth_sz = [tgt_sz[i] for i in other]
    if target[w_ax] > aniso * max(oth_sp) and tgt_sz[w_ax] * aniso < min(oth_sz):
        t = float(np.percentile(sp[:, w_ax], 10))
        if t < max(oth_sp):
            t = max(max(oth_sp), t) + 1e-5
        target[w_ax] = t
    return target


def _norm_schemes(dj: dict, fp: dict) -> tuple[list[str], list[bool]]:
    modalities = dj["channel_names"] if "channel_names" in dj else dj["modality"]
    classes = [normalization_class_for_channel(m) for m in modalities.values()]
    names = [c.__name__ for c in classes]
    if fp["median_relative_size_after_cropping"] < 0.75:
        mask = [bool(c.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true) for c in classes]
    else:
        mask = [False] * len(classes)
    return names, mask


def _save_plans(pre_folder: str, plans_name: str, plans: dict) -> None:
    recursive_fix_for_json_export(plans)
    out = join(pre_folder, plans_name + ".json")
    if isfile(out):
        old = load_json(out)
        prev = {k: v for k, v in old["configurations"].items() if k not in plans["configurations"]}
        plans["configurations"].update(prev)
    maybe_mkdir_p(pre_folder)
    save_json(plans, out, sort_keys=False)


def run_plan(
    dataset_id: int,
    planner_class_name: str,
    gpu_mem_gb: Optional[float],
    max_patch: Optional[Tuple[int, int, int]],
    plans_name_override: Optional[str],
    overwrite_target_spacing: Optional[Tuple[float, float, float]] = None,
    suppress_transpose: bool = False,
    preprocessor_name: str = "DefaultPreprocessor",
    verbose: bool = True,
    patch_edge: int = 256,
) -> str:
    dn = convert_id_to_dataset_name(dataset_id)
    rf = join(raw_dir(), dn)
    pf = join(preprocessed_dir(), dn)
    if not isfile(join(pf, "dataset_fingerprint.json")):
        raise RuntimeError("fingerprint missing; run fingerprint step first")
    dj = load_json(join(rf, "dataset.json"))
    fp = load_json(join(pf, "dataset_fingerprint.json"))
    preset = PRESETS.get(planner_class_name)
    if preset is None:
        raise RuntimeError(f"unknown planner {planner_class_name!r}; use one of {sorted(PRESETS)}")
    ident = plans_name_override or preset.plans_identifier
    if plans_name_override:
        preset = replace(preset, plans_identifier=ident)
    vram = float(gpu_mem_gb if gpu_mem_gb is not None else preset.default_vram_gb)
    _maybe_copy_splits(rf, pf)
    ows = list(overwrite_target_spacing) if overwrite_target_spacing is not None else None
    ts = _fullres_spacing(fp, ows, ANISO_THRESHOLD)
    tf, tb = _transpose(suppress_transpose, ts)
    ts_t = ts[tf]
    new_shapes = [compute_new_shape(tuple(sh), sp, ts) for sp, sh in zip(fp["spacings"], fp["shapes_after_crop"])]
    med = np.median(np.stack(new_shapes, 0), 0)
    med_t = med[tf]
    if med_t[0] == 1:
        raise RuntimeError("2D-only dataset not supported")
    approx_nvox = float(np.prod(med_t, dtype=np.float64) * dj["numTraining"])
    norm_n, norm_m = _norm_schemes(dj, fp)
    cache: dict = {}
    plan_fr = resenc_3d_fullres_plan(
        ts_t,
        med_t,
        "nnUNetPlans_3d_fullres",
        approx_nvox,
        dj,
        norm_n,
        norm_m,
        preprocessor_name,
        vram,
        preset,
        max_patch,
        cache,
        patch_edge,
    )
    ds = get_filenames_of_train_images_and_targets(rf, dj)
    ex = ds[next(iter(ds.keys()))]["images"][0]
    image_rw = reader_writer_class_from_dataset(dj, ex, verbose=verbose).__name__
    med_sp = np.median(fp["spacings"], 0)[tf]
    med_sh = np.median(np.stack(fp["shapes_after_crop"], 0), 0)[tf]
    shutil.copyfile(join(rf, "dataset.json"), join(pf, "dataset.json"))
    plans = {
        "dataset_name": dn,
        "plans_name": ident,
        "original_median_spacing_after_transp": [float(x) for x in med_sp],
        "original_median_shape_after_transp": [int(round(x)) for x in med_sh],
        "image_reader_writer": image_rw,
        "transpose_forward": [int(x) for x in tf],
        "transpose_backward": [int(x) for x in tb],
        "configurations": {"3d_fullres": plan_fr},
        "experiment_planner_used": planner_class_name,
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": fp["foreground_intensity_properties_per_channel"],
    }
    _save_plans(pf, ident, plans)
    if verbose:
        cprint(f"[bold green]✓ wrote {ident}.json[/bold green]")
    return ident
