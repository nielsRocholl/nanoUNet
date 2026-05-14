"""Patch size inflation for spatial augment (batchgenerators rotate_coords_3d)."""

from __future__ import annotations

import numpy as np
from batchgenerators.augmentations.utils import rotate_coords_3d


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2.0 * np.pi, rot_x)
    rot_y = min(90 / 360 * 2.0 * np.pi, rot_y)
    rot_z = min(90 / 360 * 2.0 * np.pi, rot_z)
    coords = np.array(final_patch_size)
    assert len(coords) == 3
    final_shape = np.copy(coords)
    final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
    final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
    final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)
