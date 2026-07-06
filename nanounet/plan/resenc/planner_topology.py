"""Pool strides + conv kernel sizes + patch pad so sizes divide by ``2**num_pool`` (nnU-Net topology)."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def _shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)


def _pad_shape(shape, must_be_divisible_by):
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)
    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]
    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    return np.array(new_shp).astype(int)


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    dim = len(spacing)
    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))
    pool_op_kernel_sizes = [[1] * len(spacing)]
    conv_kernel_sizes = []
    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim
    while True:
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break
        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]
        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break
        for d in range(dim):
            if kernel_size[d] != 3 and current_spacing[d] / min(current_spacing) < 2:
                kernel_size[d] = 3
        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]
        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1
        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))
    must_be_divisible_by = _shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = _pad_shape(patch_size, must_be_divisible_by)

    def _to_tuple(lst):
        return tuple(_to_tuple(i) if isinstance(i, list) else i for i in lst)

    conv_kernel_sizes.append([3] * dim)
    return num_pool_per_axis, _to_tuple(pool_op_kernel_sizes), _to_tuple(conv_kernel_sizes), tuple(patch_size), must_be_divisible_by
