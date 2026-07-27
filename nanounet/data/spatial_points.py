"""SpatialTransform/MirrorTransform subclasses that also carry click coordinates through the
augmentation chain, so prompt heatmaps can be rendered AFTER the chain at the final patch size
instead of being warped as extra image channels (grid_sample then only ever sees CT channels).

Points are float32 (N,3) patch-local z,y,x; they may land outside the output patch after the
transform, and that is expected -- callers (encode_points_to_heatmap) already skip balls that
fall entirely outside, so we never clip here.

Affine path mirrors SpatialTransform._apply_to_image (batchgeneratorsv2 spatial.py):
  grid = _create_centered_identity_grid2(patch_size)   # o - (ps-1)/2 at output index o
  grid = grid @ affine                                 # plain right-multiply, no transpose
  grid += center_location_in_pixels - shape_in/2
  then _convert divides by shape_in/2 and grid_sample(align_corners=False) unnormalises to
  continuous pixel = g_my + (shape_in-1)/2. Composite forward map:
    p = (o - (ps-1)/2) @ affine + center - 0.5
  Invert for forward point tracking (elastic_offsets must be None):
    o = (p - center + 0.5) @ inv(affine) + (ps-1)/2
When affine is None, _apply_to_image uses discrete crop_tensor(); we replicate its integer
indexing (floor + floor-division) exactly.
"""

from __future__ import annotations

import numpy as np
import torch
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform


def _invert_spatial(points: np.ndarray, patch_size, params: dict) -> np.ndarray:
    if params["elastic_offsets"] is not None:
        raise NotImplementedError(
            "spatial_points: point tracking does not support elastic deformation "
            "(get_parameters returned elastic_offsets is not None). train_transforms sets "
            "p_elastic_deform=0, so this should never fire; if you changed that, points must be "
            "rendered before the chain again instead of after."
        )
    center = np.asarray(params["center_location_in_pixels"], dtype=np.float64)
    ps = np.asarray(patch_size, dtype=np.float64)
    affine = params["affine"]
    if affine is None:
        # _apply_to_image took the discrete crop_tensor() fast path (no rotation/scaling this
        # sample): input_voxel = output_idx + floor(center) - patch_size // 2.
        c = np.floor(center).astype(np.int64)
        shift = c - (ps.astype(np.int64) // 2)
        return (points - shift).astype(np.float32)
    # Inverse of p = (o - (ps-1)/2) @ affine + center - 0.5  (see module docstring).
    out = (points - center + 0.5) @ np.linalg.inv(affine) + (ps - 1.0) / 2.0
    return out.astype(np.float32)


class SpatialPointsTransform(SpatialTransform):
    """SpatialTransform that also moves a (N,3) float `keypoints` array through the same sampled
    affine/crop as the image (see module docstring for the math). Same constructor as
    SpatialTransform."""

    def _apply_to_keypoints(self, keypoints: torch.Tensor, **params) -> torch.Tensor:
        if len(self.patch_size) != 3:
            raise NotImplementedError(
                "spatial_points: do_dummy_2d_data_aug (2D SpatialTransform over a folded z-axis) "
                "is not supported by point tracking. This patch's aspect ratio triggered the "
                "dummy-2D path; render heatmaps before the chain for this patch_size instead."
            )
        if keypoints.numel() == 0:
            return keypoints
        pts = keypoints.numpy().astype(np.float64)
        out = _invert_spatial(pts, self.patch_size, params)
        return torch.from_numpy(out)


class MirrorPointsTransform(MirrorTransform):
    """MirrorTransform that also flips a (N,3) float `keypoints` array on the same sampled axes.
    Unlike SpatialTransform, MirrorTransform never crops or interpolates, so the only extra state
    needed is the (fixed) spatial shape of the tensor at the point this transform runs in the
    chain -- pass the final patch_size explicitly."""

    def __init__(self, allowed_axes, patch_size):
        super().__init__(allowed_axes)
        self.patch_size = tuple(int(s) for s in patch_size)

    def _apply_to_keypoints(self, keypoints: torch.Tensor, **params) -> torch.Tensor:
        if keypoints.numel() == 0 or len(params["axes"]) == 0:
            return keypoints
        out = keypoints.clone()
        for a in params["axes"]:
            out[:, a] = self.patch_size[a] - 1 - out[:, a]
        return out
