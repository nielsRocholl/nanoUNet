"""Padded click coords from scanner JSON (x,y,z) or caller-supplied preprocessed (z,y,x)."""

from __future__ import annotations

from nanounet.infer.roi_slices import map_points_zyx_unpadded_to_padded
from nanounet.prompt.coords import points_to_centers_zyx

ZYX = tuple[int, int, int]


def resolve_pts_pad(
    *,
    points_xyz: list[tuple[float, float, float]],
    points_zyx_unpadded: list[ZYX] | None,
    props: dict,
    unpadded_shape: tuple[int, ...],
    spacing: tuple[float, ...],
    transpose_forward: tuple[int, ...] | list[int],
    slicer_revert: tuple,
) -> list[ZYX]:
    if points_zyx_unpadded is not None:
        if points_xyz:
            raise ValueError("pass points_xyz or points_zyx_unpadded, not both")
        if not points_zyx_unpadded:
            return []
        return map_points_zyx_unpadded_to_padded(points_zyx_unpadded, slicer_revert)
    if not points_xyz:
        return []
    pts_zyx = [(z, y, x) for x, y, z in points_xyz]
    pre = points_to_centers_zyx(
        pts_zyx,
        "voxel",
        props,
        unpadded_shape,
        spacing,
        transpose_forward,
        voxel_coordinate_frame="full",
    )
    return map_points_zyx_unpadded_to_padded(pre, slicer_revert)
