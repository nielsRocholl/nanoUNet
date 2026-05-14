"""Single-case raw → cropped normalized blosc2 bundle (transpose → nonzero crop → norm → resample)."""

from __future__ import annotations

from typing import List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json as bg_load_json

from nanounet.common import cprint
from nanounet.data.crop import crop_to_nonzero
from nanounet.data.io import reader_writer_class_from_dataset
from nanounet.data.normalization import normalization_class_from_plan_name
from nanounet.data.resampling import compute_new_shape
from nanounet.plan.case_pp_sample import sample_foreground_locations
from nanounet.plan.labels import labels_from_dataset_json
from nanounet.plan.plans import Config3d, Plans


def _load_dj(dataset_json: dict | str) -> dict:
    return dataset_json if isinstance(dataset_json, dict) else bg_load_json(dataset_json)


def run_case_npy(
    data: np.ndarray,
    seg: np.ndarray | None,
    properties: dict,
    plans: Plans,
    cm: Config3d,
    dataset_json: dict,
    verbose: bool = False,
):
    data = data.astype(np.float32)
    has_seg = seg is not None
    if seg is not None:
        assert data.shape[1:] == seg.shape[1:]
        seg = np.copy(seg)
    tf = plans.transpose_forward
    data = data.transpose([0, *[i + 1 for i in tf]])
    if seg is not None:
        seg = seg.transpose([0, *[i + 1 for i in tf]])
    o_sp = [properties["spacing"][i] for i in tf]
    properties["shape_before_cropping"] = data.shape[1:]
    data, seg, bbox = crop_to_nonzero(data, seg)
    properties["bbox_used_for_cropping"] = bbox
    properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]
    t_sp = list(cm.spacing)
    if len(t_sp) < len(data.shape[1:]):
        t_sp = [o_sp[0]] + t_sp
    new_sh = tuple(int(x) for x in compute_new_shape(data.shape[1:], o_sp, t_sp))
    fi = plans.foreground_intensity_properties_per_channel
    for c in range(data.shape[0]):
        cls = normalization_class_from_plan_name(cm.normalization_schemes[c])
        ip = fi[str(c)] if str(c) in fi else fi[int(c)]
        nrm = cls(use_mask_for_norm=cm.use_mask_for_norm[c], intensityproperties=ip)
        data[c] = nrm.run(data[c], seg[0])
    data = np.asarray(cm.resampling_fn_data(data, new_sh, o_sp, t_sp))
    seg = np.asarray(cm.resampling_fn_seg(seg, new_sh, o_sp, t_sp))
    if verbose:
        cprint(
            f"[dim]pp {data.shape} {seg.shape if seg is not None else None} {new_sh} {o_sp} {t_sp}[/dim]"
        )
    if has_seg:
        lm = labels_from_dataset_json(dataset_json)
        coll = list(lm.foreground_labels)
        if lm.has_ignore_label:
            coll.append([-1] + lm.all_labels)
        properties["class_locations"] = sample_foreground_locations(seg, coll, verbose=verbose)
    if seg is not None:
        seg = seg.astype(np.int16 if np.max(seg) > 127 else np.int8)
    return data, seg, properties


def run_case(image_files: List[str], seg_file: str | None, plans: Plans, cm: Config3d, dataset_json: dict | str, verbose: bool = False):
    dj = _load_dj(dataset_json)
    rw_cls = reader_writer_class_from_dataset(dj, image_files[0], verbose=verbose)
    rw = rw_cls()
    data, p = rw.read_images(image_files)
    seg = rw.read_seg(seg_file)[0] if seg_file else None
    return run_case_npy(data, seg, p, plans, cm, dj, verbose=verbose)


def run_case_save(
    out_trunc: str,
    image_files: List[str],
    seg_file: str,
    plans: Plans,
    cm: Config3d,
    dataset_json: dict | str,
    verbose: bool = False,
):
    from nanounet.data.blosc2_dataset import Blosc2Folder

    dj = _load_dj(dataset_json)
    data, seg, props = run_case(image_files, seg_file, plans, cm, dj, verbose=verbose)
    ps = tuple(cm.patch_size)
    bd, cd = Blosc2Folder.comp_blosc2_params(data.shape, ps, data.itemsize)
    bs, cs = Blosc2Folder.comp_blosc2_params(seg.shape, ps, seg.itemsize)
    Blosc2Folder.save_case(
        data.astype(np.float32, copy=False),
        seg.astype(np.int16, copy=False),
        props,
        out_trunc,
        chunks=cd,
        blocks=bd,
        chunks_seg=cs,
        blocks_seg=bs,
    )
