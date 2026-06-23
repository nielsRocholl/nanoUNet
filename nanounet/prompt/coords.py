"""World/voxel points → preprocessed (z,y,x); JSON reader for native-voxel clickpoints."""

from __future__ import annotations

import json
import os
from typing import List, Tuple, Union

import numpy as np
import SimpleITK as sitk


def _bbox_lo(bbox, axis: int) -> int:
    b = bbox[axis]
    if isinstance(b, slice):
        return int(b.start or 0)
    if isinstance(b, (tuple, list)):
        return int(b[0])
    return int(b)


def points_to_centers_zyx(
    points: List,
    points_space: str,
    properties: dict,
    preprocessed_shape: Tuple[int, ...],
    spacing: Tuple[float, ...],
    transpose_forward: Union[Tuple[int, ...], List[int], None] = None,
    *,
    voxel_coordinate_frame: str = "full",
) -> List[Tuple[int, int, int]]:
    if voxel_coordinate_frame not in ("full", "preprocessed"):
        raise ValueError(f"voxel_coordinate_frame must be 'full' or 'preprocessed', got {voxel_coordinate_frame!r}")
    if points_space not in ("voxel", "world"):
        raise ValueError(f"points_space must be 'voxel' or 'world', got {points_space!r}")
    out: List[Tuple[int, int, int]] = []
    for pt in points:
        if len(pt) != 3:
            raise ValueError(f"Point must have 3 coords, got {len(pt)}")
        if points_space == "voxel":
            z, y, x = float(pt[0]), float(pt[1]), float(pt[2])
            bbox = properties.get("bbox_used_for_cropping")
            shape_ac = properties.get("shape_after_cropping_and_before_resampling")
            map_full = voxel_coordinate_frame == "full" and bbox is not None and shape_ac is not None
            if map_full:
                if transpose_forward is not None:
                    arr = np.array([z, y, x], dtype=np.float64)
                    z, y, x = (
                        float(arr[transpose_forward[0]]),
                        float(arr[transpose_forward[1]]),
                        float(arr[transpose_forward[2]]),
                    )
                zi, yi, xi = int(np.round(z)), int(np.round(y)), int(np.round(x))
                x = max(0, min(shape_ac[2] - 1, xi - _bbox_lo(bbox, 2)))
                y = max(0, min(shape_ac[1] - 1, yi - _bbox_lo(bbox, 1)))
                z = max(0, min(shape_ac[0] - 1, zi - _bbox_lo(bbox, 0)))
                factor = [preprocessed_shape[i] / shape_ac[i] for i in range(3)]
                x = int(np.round(x * factor[2]))
                y = int(np.round(y * factor[1]))
                z = int(np.round(z * factor[0]))
            else:
                z, y, x = int(np.round(z)), int(np.round(y)), int(np.round(x))
        else:
            x_phys, y_phys, z_phys = float(pt[0]), float(pt[1]), float(pt[2])
            st = properties.get("sitk_stuff")
            if not st:
                raise KeyError("points_space='world' requires properties['sitk_stuff']")
            orig, sp = st["origin"], st["spacing"]
            d = st.get("direction")
            if d is not None and len(tuple(d)) == 9:
                ref = sitk.Image((1, 1, 1), sitk.sitkUInt8)
                ref.SetOrigin(orig)
                ref.SetSpacing(sp)
                ref.SetDirection(d)
                ix, iy, iz = ref.TransformPhysicalPointToContinuousIndex((x_phys, y_phys, z_phys))
                z, y, x = float(iz), float(iy), float(ix)
            else:
                vox_x = (x_phys - orig[0]) / sp[0]
                vox_y = (y_phys - orig[1]) / sp[1]
                vox_z = (z_phys - orig[2]) / sp[2]
                z, y, x = vox_z, vox_y, vox_x
            if transpose_forward is not None:
                arr = np.array([z, y, x])
                z, y, x = arr[transpose_forward[0]], arr[transpose_forward[1]], arr[transpose_forward[2]]
            bbox = properties["bbox_used_for_cropping"]
            shape_ac = properties["shape_after_cropping_and_before_resampling"]
            x = max(0, min(shape_ac[2] - 1, x - _bbox_lo(bbox, 2)))
            y = max(0, min(shape_ac[1] - 1, y - _bbox_lo(bbox, 1)))
            z = max(0, min(shape_ac[0] - 1, z - _bbox_lo(bbox, 0)))
            factor = [preprocessed_shape[i] / shape_ac[i] for i in range(3)]
            x = int(np.round(x * factor[2]))
            y = int(np.round(y * factor[1]))
            z = int(np.round(z * factor[0]))
        z = int(np.clip(z, 0, preprocessed_shape[0] - 1))
        y = int(np.clip(y, 0, preprocessed_shape[1] - 1))
        x = int(np.clip(x, 0, preprocessed_shape[2] - 1))
        out.append((z, y, x))
    return out


def load_points_xyz(json_path: str) -> list[tuple[float, float, float]]:
    """Read 'Points of interest' JSON -> native-voxel (x, y, z) list.

    Format (Longitudinal_CT_v2/inputsTrFU/*.json):
      {"points": [{"name": "1", "point": [x, y, z]}, ...], "type": "Multiple points", ...}
    Empty 'points' -> []  (valid zero-lesion case, handled downstream).
    Missing file or 'points' not a list -> raise (R12: no silent fallback).
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("points")
    if pts is None:
        raise KeyError(f"'points' missing in {json_path}")
    if not isinstance(pts, list):
        raise TypeError(f"'points' must be a list in {json_path}")
    out: list[tuple[float, float, float]] = []
    for item in pts:
        p = item["point"]
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out
