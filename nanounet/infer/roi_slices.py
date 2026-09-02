"""Padding offset math for ROI tiles: slice helpers, bg logits, face-FG expand clicks."""

from __future__ import annotations

from typing import List, Tuple

import torch

from nanounet.plan.labels import Labels
from nanounet.prompt.cluster import cluster_points_for_patch_size, spatial_slices_covering_points

ZYX = Tuple[int, int, int]


def map_points_zyx_unpadded_to_padded(points: List[ZYX], slicer_revert: tuple) -> List[ZYX]:
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


def background_logits_vector(lm: Labels, num_heads: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    v = torch.full((num_heads,), -10.0, device=device, dtype=dtype)
    v[0] = 10.0
    return v


def _fg_mask(logits: torch.Tensor, lm: Labels) -> torch.Tensor:
    seg = logits.argmax(0)
    fg = torch.zeros(seg.shape, dtype=torch.bool, device=seg.device)
    for fl in lm.foreground_labels:
        fg |= seg == int(fl)
    return fg


def fg_face_touch(logits: torch.Tensor, lm: Labels) -> Tuple[bool, ...]:
    fg = _fg_mask(logits, lm)
    t = torch.stack([
        fg[0].any(), fg[-1].any(),
        fg[:, 0].any(), fg[:, -1].any(),
        fg[:, :, 0].any(), fg[:, :, -1].any(),
    ])
    return tuple(bool(x) for x in t.tolist())


def face_fg_click_global(
    logits: torch.Tensor, sl: Tuple[slice, slice, slice], face_i: int, lm: Labels
) -> ZYX | None:
    """Centroid of FG on parent face `face_i`, in padded global coords. None if the face is empty."""
    fg = _fg_mask(logits, lm)
    loc = _face_centroid_local(fg, face_i)
    if loc is None:
        return None
    return (sl[0].start + loc[0], sl[1].start + loc[1], sl[2].start + loc[2])


def _face_centroid_local(fg: torch.Tensor, face_i: int) -> ZYX | None:
    d0, d1, d2 = fg.shape
    if face_i == 0:
        slc, a0 = fg[0], 0
    elif face_i == 1:
        slc, a0 = fg[-1], d0 - 1
    elif face_i == 2:
        slc, a0 = fg[:, 0], 0
    elif face_i == 3:
        slc, a0 = fg[:, -1], d1 - 1
    elif face_i == 4:
        slc, a0 = fg[:, :, 0], 0
    else:
        slc, a0 = fg[:, :, -1], d2 - 1
    if not slc.any():
        return None
    idx = torch.nonzero(slc, as_tuple=False).float().mean(0)
    b, c = int(idx[0].round()), int(idx[1].round())
    if face_i <= 1:
        return (a0, b, c)
    if face_i <= 3:
        return (b, a0, c)
    return (b, c, a0)


def extra_click_in_tile(gxyz: ZYX, sl: Tuple[slice, slice, slice], patch_size: Tuple[int, int, int]) -> ZYX:
    """Parent-face click → child-tile local, clamped into the patch."""
    lz = min(max(gxyz[0] - sl[0].start, 0), patch_size[0] - 1)
    ly = min(max(gxyz[1] - sl[1].start, 0), patch_size[1] - 1)
    lx = min(max(gxyz[2] - sl[2].start, 0), patch_size[2] - 1)
    return (lz, ly, lx)


def seed_slices_for_points(
    pts_pad: List[ZYX],
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
    cluster_margin_frac: float,
    mode: str,
) -> Tuple[List[List[ZYX]], List[Tuple[slice, slice, slice]]]:
    """Seed-tile placement for `mode`. Extracted so preprocessing can compute the same
    tile layout as predict_case_logits before any forward pass runs (click-AABB preprocess)."""
    assert mode in ("clustered", "centered")
    if mode == "clustered":
        seeds_pts = cluster_points_for_patch_size(pts_pad, patch_size, cluster_margin_frac)
        seed_slices = [spatial_slices_covering_points(cl, patch_size, padded_shape) for cl in seeds_pts]
        return seeds_pts, seed_slices
    seen: set = set()
    seeds_pts, seed_slices = [], []
    for p in pts_pad:
        sl = centered_spatial_slices_at_point(p[0], p[1], p[2], patch_size, padded_shape)
        key = (sl[0].start, sl[1].start, sl[2].start, p)
        if key in seen:
            continue
        seen.add(key)
        seeds_pts.append([p])
        seed_slices.append(sl)
    return seeds_pts, seed_slices


def grow_canvas(logits_acc, margin_buf, origin, sl, bg_vec, neg, acc_dtype):
    """Expand one cluster canvas so it covers tile `sl`. Copy existing logits/margin."""
    nh = logits_acc.shape[0]
    cur_hi = tuple(origin[a] + logits_acc.shape[a + 1] for a in range(3))
    lo = tuple(min(origin[a], sl[a].start) for a in range(3))
    hi = tuple(max(cur_hi[a], sl[a].stop) for a in range(3))
    if lo == origin and hi == cur_hi:
        return logits_acc, margin_buf, origin
    nsh = tuple(hi[a] - lo[a] for a in range(3))
    nacc = bg_vec.view(-1, 1, 1, 1).to(acc_dtype).expand(nh, *nsh).contiguous()
    nm = torch.full(nsh, neg, dtype=acc_dtype, device=logits_acc.device)
    off = tuple(origin[a] - lo[a] for a in range(3))
    z, y, x = logits_acc.shape[1:]
    nacc[:, off[0]:off[0] + z, off[1]:off[1] + y, off[2]:off[2] + x] = logits_acc
    nm[off[0]:off[0] + z, off[1]:off[1] + y, off[2]:off[2] + x] = margin_buf
    return nacc, nm, lo
