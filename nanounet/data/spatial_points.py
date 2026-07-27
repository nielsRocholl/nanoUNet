"""SpatialTransform/MirrorTransform subclasses that also carry click coordinates through the
augmentation chain, so prompt heatmaps can be rendered AFTER the chain at the final patch size
instead of being warped as extra image channels (grid_sample then only ever sees CT channels).

Points are float32 (N,3) patch-local z,y,x; they may land outside the output patch after the
transform, and that is expected -- callers (encode_points_to_heatmap) already skip balls that
fall entirely outside, so we never clip here.

The affine math mirrors SpatialTransform._apply_to_image exactly (see the installed
batchgeneratorsv2/transforms/spatial/spatial.py). Tracing the grid built there through
_convert_my_grid_to_grid_sample_grid (which divides by shape/2) and pytorch's align_corners=False
convention (continuous pixel x = g*size/2 + (size-1)/2 for normalized coord g) shows grid_sample
maps OUTPUT positions to INPUT positions via
    input_voxel(o) = (o - (patch_size-1)/2) @ affine.T + center_location_in_pixels - 0.5
(the -0.5 comes from new_center using shape/2 while the final pixel conversion uses (shape-1)/2).
The `.T` is not optional: spatial.py builds `affine` to act on COLUMN vectors (x' = affine @ x),
but the grid it warps stores points as ROW vectors, so spatial.py itself right-multiplies by
`affine.T` when it warps the grid ("grid stores spatial vectors as row vectors ... affine is built
to act on column vectors ... we therefore multiply by affine.T", spatial.py, `_apply_to_image`,
citing their issue #24). A point tracker that also uses row vectors must follow the same
convention or it silently transforms by affine's transpose instead of affine itself -- for a pure
scaling matrix (symmetric) this is invisible, for any real rotation it is not, which is why a
missing `.T` here previously slipped past review (see /nnunet_data/prompt_sensitivity/final_plan
verification notes: error grew with rotation magnitude and vanished for scale-only draws, the
exact fingerprint of A vs A.T). Tracking a point forward through the same sampled transform needs
the INVERSE of the (correct) forward map:
    o = (p - center_location_in_pixels + 0.5) @ inv(affine).T + (patch_size-1)/2
(inv(affine).T == inv(affine.T), so this is just "invert affine, then apply with the same
row-vector convention as spatial.py's forward map" -- no separate derivation needed.)
When affine is None, _apply_to_image takes the discrete crop_tensor() fast path instead of
grid_sample (no interpolation, no rotation, so no row/column-vector question arises, and no -0.5
term); we replicate its integer indexing (floor + floor-division) exactly so the CT channels and
the points agree to sub-voxel precision.
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
    inv_affine = np.linalg.inv(affine)
    # affine acts on column vectors; points here are row vectors, so apply it (and its inverse)
    # transposed -- see module docstring and spatial.py's own affine.T comment (issue #24).
    out = (points - center + 0.5) @ inv_affine.T + (ps - 1.0) / 2.0
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
