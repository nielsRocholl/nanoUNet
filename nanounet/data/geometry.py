"""Nonzero crop, transpose, resample delegate, insert crop (acvl_utils)."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image


def nonzero_slices_3d(mask: np.ndarray) -> Tuple[slice, slice, slice]:
    if mask.ndim != 3:
        raise ValueError("mask must be 3d")
    z, y, x = np.where(mask)
    if z.size == 0:
        raise ValueError("empty mask")
    return slice(int(z.min()), int(z.max()) + 1), slice(int(y.min()), int(y.max()) + 1), slice(int(x.min()), int(x.max()) + 1)


def transpose_forward_np(x: np.ndarray, axes: Sequence[int]) -> np.ndarray:
    return x.transpose(tuple(axes))


def transpose_backward_np(x: np.ndarray, forward_axes: Sequence[int]) -> np.ndarray:
    inv = [0] * len(forward_axes)
    for i, j in enumerate(forward_axes):
        inv[j] = i
    return x.transpose(tuple(inv))


def resample_tensor(
    resampling_fn,
    array: Union[np.ndarray, torch.Tensor],
    current_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
):
    return resampling_fn(array, current_shape, current_spacing, new_spacing)


def insert_crop(image: np.ndarray, crop: np.ndarray, bbox_ll: List[List[int]]) -> np.ndarray:
    return insert_crop_into_image(image, crop, bbox_ll)
