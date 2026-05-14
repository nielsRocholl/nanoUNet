"""Logits in preprocessed space → resample, uncrop, untranspose, SimpleITK seg write."""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

from nanounet.data.io import reader_writer_class_from_dataset
from nanounet.plan.labels import Labels, labels_from_dataset_json
from nanounet.plan.plans import Config3d, Plans


def convert_logits_to_seg_shape(
    logits: torch.Tensor | np.ndarray,
    plans: Plans,
    cm: Config3d,
    lm: Labels,
    props: dict,
    num_threads: int = 8,
) -> np.ndarray:
    o = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    if isinstance(logits, np.ndarray):
        x = torch.from_numpy(logits)
    else:
        x = logits
    sp_t = [props["spacing"][i] for i in plans.transpose_forward]
    sh = props["shape_after_cropping_and_before_resampling"]
    cur_sp = cm.spacing if len(cm.spacing) == len(sh) else [sp_t[0], *cm.spacing]
    tgt_sp = [props["spacing"][i] for i in plans.transpose_forward]
    x = cm.resampling_fn_probabilities(x, sh, cur_sp, tgt_sp)
    seg = lm.convert_logits_to_segmentation(x)
    del x
    full = np.zeros(props["shape_before_cropping"], dtype=np.uint8 if len(lm.foreground_labels) < 255 else np.uint16)
    full = insert_crop_into_image(full, seg, props["bbox_used_for_cropping"])
    full = full.transpose(tuple(plans.transpose_backward))
    torch.set_num_threads(o)
    return full


def export_prediction_from_logits(
    logits: Union[np.ndarray, torch.Tensor],
    props: dict,
    cm: Config3d,
    plans: Plans,
    dataset_json: dict,
    output_trunc: str,
    save_probabilities: bool = False,
):
    lm = labels_from_dataset_json(dataset_json)
    if save_probabilities:
        raise NotImplementedError("save_probabilities")
    seg = convert_logits_to_seg_shape(logits, plans, cm, lm, props)
    rw_cls = reader_writer_class_from_dataset(dataset_json, None, verbose=False)
    rw = rw_cls()
    rw.write_seg(seg, output_trunc + dataset_json["file_ending"], props)
