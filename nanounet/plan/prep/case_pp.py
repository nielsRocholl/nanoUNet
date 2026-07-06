"""Single-case raw → cropped normalized blosc2 bundle; uniform foreground sampling for oversampling."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json as bg_load_json

from nanounet.common import cprint
from nanounet.data.crop import crop_to_nonzero
from nanounet.data.io import reader_writer_class_from_dataset
from nanounet.data.normalization import normalization_class_from_plan_name
from nanounet.data.resampling import compute_new_shape
from nanounet.plan.labels import labels_from_dataset_json
from nanounet.plan.plans import Config3d, Plans


def sample_foreground_locations(
    seg: np.ndarray,
    classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
    seed: int = 1234,
    verbose: bool = False,
    min_num_samples: int = 10000,
    min_percent_coverage: float = 0.01,
) -> dict:
    rnd = np.random.RandomState(seed)
    req_labs = set()
    for c in classes_or_regions:
        if isinstance(c, (tuple, list)):
            req_labs.update(int(x) for x in c)
        else:
            req_labs.add(int(c))
    req_arr = np.fromiter(req_labs, dtype=np.int32)
    vm = np.isin(seg, req_arr)
    coords = np.argwhere(vm)
    seg_sel = seg[vm]
    del vm
    n = seg_sel.size
    out: dict = {}
    if n == 0:
        for c in classes_or_regions:
            k = tuple(c) if isinstance(c, (tuple, list)) else int(c)
            out[k] = []
        return out
    order = np.argsort(seg_sel, kind="stable")
    lab_sorted = seg_sel[order]
    coords_sorted = coords[order]
    chg = np.flatnonzero(lab_sorted[1:] != lab_sorted[:-1]) + 1
    starts = np.r_[0, chg]
    ends = np.r_[chg, n]
    labels_present = lab_sorted[starts]
    l2r = {int(l): (int(s), int(e)) for l, s, e in zip(labels_present, starts, ends)}
    present = set(l2r.keys())
    for c in classes_or_regions:
        is_reg = isinstance(c, (tuple, list))
        labs = tuple(int(x) for x in c) if is_reg else (int(c),)
        k = labs if is_reg else labs[0]
        if not any(lab in present for lab in labs):
            out[k] = []
            continue
        ranges, counts = [], []
        for lab in labs:
            r = l2r.get(lab)
            if r is None:
                continue
            s, e = r
            cnt = e - s
            if cnt > 0:
                ranges.append((s, e))
                counts.append(cnt)
        if not counts:
            out[k] = []
            continue
        total = int(np.sum(counts))
        tgt = min(min_num_samples, total)
        tgt = max(tgt, int(np.ceil(total * min_percent_coverage)))
        offsets = rnd.choice(total, tgt, replace=False)
        cum = np.cumsum(counts)
        which = np.searchsorted(cum, offsets, side="right")
        prev = np.concatenate(([0], cum[:-1]))
        in_range = offsets - prev[which]
        starts_for = np.fromiter((ranges[i][0] for i in which), dtype=np.int64, count=which.size)
        picked = starts_for + in_range.astype(np.int64)
        out[k] = coords_sorted[picked]
        if verbose:
            cprint(f"[dim]{c} {tgt}[/dim]")
    return out


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
