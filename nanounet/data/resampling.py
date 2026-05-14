"""3D resampling: scipy map_coordinates + batchgenerators seg resize. nnU-Net-default_resampling port."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from nanounet.common import ANISO_THRESHOLD


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    return (np.max(spacing) / np.min(spacing)) > anisotropy_threshold


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
    return axis


def compute_new_shape(
    old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    assert len(old_spacing) == len(old_shape) == len(new_spacing)
    return np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])


def determine_do_sep_z_and_axis(
    force_separate_z: bool | None,
    current_spacing,
    new_spacing,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
) -> tuple[bool, int | None]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        axis = get_lowres_axis(current_spacing) if force_separate_z else None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z, axis = True, get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z, axis = True, get_lowres_axis(new_spacing)
        else:
            do_separate_z, axis = False, None
    if axis is not None:
        if len(axis) == 3:
            do_separate_z, axis = False, None
        elif len(axis) == 2:
            do_separate_z, axis = False, None
        else:
            axis = int(axis[0])
    return do_separate_z, axis


def resample_data_or_seg(
    data: np.ndarray,
    new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    axis: int | None = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
    dtype_out=None,
):
    assert data.ndim == 4
    assert len(new_shape) == data.ndim - 1
    if is_seg:
        resize_fn, kwargs = resize_segmentation, OrderedDict()
    else:
        resize_fn, kwargs = resize, {"mode": "edge", "anti_aliasing": False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        if do_separate_z:
            assert axis is not None
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]
            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape
                    row_scale, col_scale, dim_scale = float(orig_rows) / rows, float(orig_cols) / cols, float(orig_dim) / dim
                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5
                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode="nearest")[None]
                    else:
                        for cl in np.sort(pd.unique(reshaped_here.ravel())):
                            reshaped_final[c][np.round(map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z, mode="nearest")) > 0.5] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final
    return data


def resample_data_or_seg_to_spacing(
    data: np.ndarray,
    current_spacing,
    new_spacing,
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: bool | None = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
):
    do_sep, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold)
    assert data.ndim == 4
    new_shape = compute_new_shape(np.array(data.shape[1:]), current_spacing, new_spacing)
    return resample_data_or_seg(data, new_shape, is_seg, axis, order, do_sep, order_z)


def resample_data_or_seg_to_shape(
    data: torch.Tensor | np.ndarray,
    new_shape,
    current_spacing,
    new_spacing,
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: bool | None = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
):
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    do_sep, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold)
    assert data.ndim == 4
    return resample_data_or_seg(data, new_shape, is_seg, axis, order, do_sep, order_z)
