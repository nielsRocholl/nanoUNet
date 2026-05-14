"""Padding offset math for ROI tiles: slice helpers, bg logits vector, merge normalization."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from nanounet.plan.labels import Labels


def map_points_zyx_unpadded_to_padded(points: List[Tuple[int, int, int]], slicer_revert: tuple) -> List[Tuple[int, int, int]]:
    dz, dy, dx = slicer_revert[1].start, slicer_revert[2].start, slicer_revert[3].start
    return [(z + dz, y + dy, x + dx) for z, y, x in points]


def centered_spatial_slices_at_point(
    pz: int, py: int, px: int, patch_size: Tuple[int, int, int], padded_shape: Tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    starts = []
    for p, ps, dim in zip((pz, py, px), patch_size, padded_shape):
        s = p - ps // 2
        s = max(0, min(s, dim - ps))
        starts.append(s)
    return tuple(slice(starts[i], starts[i] + patch_size[i]) for i in range(3))


def spatial_slices_to_tuple(sz: slice, sy: slice, sx: slice) -> Tuple[int, int, int, int, int, int]:
    return (sz.start, sz.stop, sy.start, sy.stop, sx.start, sx.stop)


def shift_spatial_slices(
    sz: slice,
    sy: slice,
    sx: slice,
    axis: int,
    is_low_face: bool,
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    z0, y0, x0 = sz.start, sy.start, sx.start
    half = [patch_size[i] // 2 for i in range(3)]
    if axis == 0:
        z0 = z0 - half[0] if is_low_face else z0 + half[0]
    elif axis == 1:
        y0 = y0 - half[1] if is_low_face else y0 + half[1]
    else:
        x0 = x0 - half[2] if is_low_face else x0 + half[2]
    z0 = max(0, min(z0, padded_shape[0] - patch_size[0]))
    y0 = max(0, min(y0, padded_shape[1] - patch_size[1]))
    x0 = max(0, min(x0, padded_shape[2] - patch_size[2]))
    return slice(z0, z0 + patch_size[0]), slice(y0, y0 + patch_size[1]), slice(x0, x0 + patch_size[2])


def local_prompt_points_for_patch(
    seed_padded: Tuple[int, int, int] | None,
    sz: slice,
    sy: slice,
    sx: slice,
    patch_size: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    if seed_padded is None:
        return []
    pz, py, px = seed_padded
    if sz.start <= pz < sz.stop and sy.start <= py < sy.stop and sx.start <= px < sx.stop:
        return [(pz - sz.start, py - sy.start, px - sx.start)]
    return [(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)]


def background_logits_vector(lm: Labels, num_heads: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    v = torch.full((num_heads,), -10.0, device=device, dtype=dtype)
    v[0] = 10.0
    return v


def safe_divide_merged_logits(logits: torch.Tensor, n_pred: torch.Tensor, bg: torch.Tensor, eps: float = 1e-6) -> None:
    valid = n_pred > eps
    inv = torch.clamp(n_pred, min=eps)
    scaled = logits / inv
    bg_e = bg.view(-1, 1, 1, 1).expand_as(logits)
    ve = valid.unsqueeze(0).expand_as(logits)
    logits.copy_(torch.where(ve, scaled, bg_e))


@torch.inference_mode()
def patch_fg_mask_from_logits(patch_logits: torch.Tensor, lm: Labels) -> np.ndarray:
    seg = patch_logits.argmax(0)
    fg = torch.zeros_like(seg, dtype=torch.bool)
    for fl in lm.foreground_labels:
        fg |= seg == int(fl)
    return fg.cpu().numpy()
