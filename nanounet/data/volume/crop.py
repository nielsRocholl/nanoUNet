"""Nonzero bounding-box crop; seg voxels outside mask set to ``nonzero_label`` (default -1)."""

from __future__ import annotations

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, get_bbox_from_mask
from scipy.ndimage import binary_fill_holes


def _nonzero_mask(data: np.ndarray) -> np.ndarray:
    assert data.ndim in (3, 4)
    m = data[0] != 0
    for c in range(1, data.shape[0]):
        m |= data[c] != 0
    return binary_fill_holes(m)


def crop_to_nonzero(data: np.ndarray, seg: np.ndarray | None = None, nonzero_label: int = -1):
    nonzero_mask = _nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    slicer = (slice(None),) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox
