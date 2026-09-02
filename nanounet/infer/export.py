"""pp-space uint8 argmax → per-tile native paste, SimpleITK seg write."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

from nanounet.data.io import reader_writer_class_from_dataset
from nanounet.infer.patch_export import tiles_to_native_seg
from nanounet.plan.plans import Config3d, Plans


def save_preprocessed_seg(seg: np.ndarray, spacing: tuple[float, ...], out_path: str) -> None:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(seg.astype(np.uint8))
    img.SetSpacing(tuple(float(s) for s in spacing))
    sitk.WriteImage(img, out_path)


def export_preprocessed_seg_to_native(
    seg_pp: np.ndarray,
    props: dict,
    cm: Config3d,
    plans: Plans,
    dataset_json: dict,
    output_path: str,
    num_threads: int = 8,
) -> None:
    """Preprocessed-space binary seg → native scanner grid (inverse of save_preprocessed_seg)."""
    o = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    sp_t = [props["spacing"][i] for i in plans.transpose_forward]
    sh = props["shape_after_cropping_and_before_resampling"]
    cur_sp = cm.spacing if len(cm.spacing) == len(sh) else [sp_t[0], *cm.spacing]
    tgt_sp = [props["spacing"][i] for i in plans.transpose_forward]
    x = seg_pp[None].astype(np.float32)
    x = np.asarray(cm.resampling_fn_seg(x, sh, cur_sp, tgt_sp))
    seg = x[0]
    dtype = np.uint8 if seg.max() < 255 else np.uint16
    full = np.zeros(props["shape_before_cropping"], dtype=dtype)
    full = insert_crop_into_image(full, seg.astype(dtype), props["bbox_used_for_cropping"])
    full = full.transpose(tuple(plans.transpose_backward))
    torch.set_num_threads(o)
    rw = reader_writer_class_from_dataset(dataset_json, None, verbose=False)()
    rw.write_seg(full, output_path, props)


def native_seg_from_logits(
    seg_pp: Union[np.ndarray, torch.Tensor],
    props: dict,
    cm: Config3d,
    plans: Plans,
    tiles: list[tuple[slice, slice, slice]],
) -> np.ndarray:
    if isinstance(seg_pp, torch.Tensor):
        seg_pp = np.asarray(seg_pp.detach().cpu())
    if not tiles:
        assert not np.any(seg_pp > 0), "FG voxels but no tiles (predict_case_logits must return tiles)"
    crops = [(seg_pp[sl], sl) for sl in tiles]
    return tiles_to_native_seg(crops, plans, cm, props, tuple(int(x) for x in seg_pp.shape))


def export_prediction_from_logits(
    logits: Union[np.ndarray, torch.Tensor],
    props: dict,
    cm: Config3d,
    plans: Plans,
    dataset_json: dict,
    output_trunc: str,
    tiles: list[tuple[slice, slice, slice]],
    save_probabilities: bool = False,
):
    if save_probabilities:
        raise NotImplementedError("save_probabilities")
    native = native_seg_from_logits(logits, props, cm, plans, tiles)
    rw = reader_writer_class_from_dataset(dataset_json, None, verbose=False)()
    rw.write_seg(native, output_trunc + dataset_json["file_ending"], props)
