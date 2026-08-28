"""Batched prompt-ROI logits: cluster seeds, face-grid expand, GPU argmax → CPU uint8."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from torch import autocast

from nanounet.infer.longi_row import encode_inference_row
from nanounet.infer.points_pad import resolve_pts_pad
from nanounet.infer.roi_slices import (
    background_logits_vector,
    canvas_bbox_for_seeds,
    extra_click_in_tile,
    face_fg_click_global,
    fg_face_touch,
    map_points_zyx_unpadded_to_padded,
    seed_slices_for_points,
)
from nanounet.infer.patch_export import patch_unpadded_overlap
from nanounet.infer.tta import max_cat, predict_batch_with_tta
from nanounet.prompt.cluster import cell_slices, face_neighbours, grid_stride
from nanounet.prompt.coords import points_to_centers_zyx

MAX_BORDER_EXTRA = 16
ACC_DTYPE_ENV = "NANOUNET_SINGLE_PATCH_ACCUM_DTYPE"


def _accum_dtype(dev: torch.device) -> torch.dtype:
    if dev.type == "cpu":
        return torch.float32
    r = (os.environ.get(ACC_DTYPE_ENV) or "").lower()
    return torch.float16 if r in ("half", "float16", "fp16") else torch.float32


@torch.inference_mode()
def predict_case_logits(
    *,
    net,
    lm,
    cfg,
    pl,
    cm,
    dev: torch.device,
    pad: torch.Tensor,
    slicer_revert: tuple,
    props: dict,
    points_xyz: list[tuple[float, float, float]],
    encode_prompt: bool,
    use_tta: bool,
    border_expand: bool,
    max_border_expand_extra: int,
    batch_size: int,
    use_amp: bool,
    cluster_margin_frac: float = 0.1,
    mode: str = "clustered",
    is_longi: bool = False,
    bl_present: bool = False,
    bl_points_xyz: list | None = None,
    points_zyx_unpadded: list[tuple[int, int, int]] | None = None,
    on_forward: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, list[tuple[slice, slice, slice]]]:
    assert mode in ("clustered", "centered")
    patch_size = tuple(cm.patch_size)
    padded_shape = tuple(pad.shape[1:])
    unpadded_shape = tuple(s.stop - s.start for s in slicer_revert[1:])
    nh = lm.num_segmentation_heads
    n_img = pad.shape[0] // 2 if (is_longi and bl_present) else pad.shape[0]
    n_stream = n_img + 2
    row_ch = 2 * n_stream if is_longi else n_stream
    acc_dtype = _accum_dtype(dev)
    bg_vec = background_logits_vector(lm, nh, dev, acc_dtype)
    amp_on = use_amp and dev.type == "cuda"
    spacing = tuple(cm.spacing)
    tf = pl.transpose_forward

    pts_pad = resolve_pts_pad(
        points_xyz=points_xyz, points_zyx_unpadded=points_zyx_unpadded, props=props,
        unpadded_shape=unpadded_shape, spacing=spacing, transpose_forward=tf,
        slicer_revert=slicer_revert,
    )
    if not pts_pad:
        return torch.zeros(unpadded_shape, dtype=torch.uint8), []

    bl_pts_pad = None
    if is_longi and bl_present and bl_points_xyz:
        bl_zyx = [(z, y, x) for x, y, z in bl_points_xyz]
        bl_pre = points_to_centers_zyx(
            bl_zyx, "voxel", props, unpadded_shape, spacing, tf, voxel_coordinate_frame="full",
        )
        bl_pts_pad = map_points_zyx_unpadded_to_padded(bl_pre, slicer_revert)

    seeds_pts, seed_slices = seed_slices_for_points(pts_pad, patch_size, padded_shape, cluster_margin_frac, mode)

    stride = grid_stride(patch_size, cfg.inference.tile_step_size)
    pending: list = []
    visited: set = set()
    origins: list = []
    extras_done = [0] * len(seeds_pts)
    for ci, sl in enumerate(seed_slices):
        origin = (sl[0].start, sl[1].start, sl[2].start)
        origins.append(origin)
        pending.append((ci, (0, 0, 0), sl, ()))
        visited.add((ci, (0, 0, 0)))

    canvas_sl = canvas_bbox_for_seeds(seed_slices, stride, max_border_expand_extra, border_expand, padded_shape)
    canvas_shape = tuple(s.stop - s.start for s in canvas_sl)
    canvas_origin = tuple(s.start for s in canvas_sl)

    neg = torch.finfo(acc_dtype).min
    margin_buf = torch.full(canvas_shape, neg, dtype=acc_dtype, device=dev)
    logits_acc = bg_vec.view(-1, 1, 1, 1).to(acc_dtype).expand(nh, *canvas_shape).contiguous()
    fwd_done = 0
    written: list = []
    enc_kw = dict(
        is_longi=is_longi, bl_present=bl_present, bl_pts_pad=bl_pts_pad,
    )
    with autocast(dev.type, enabled=amp_on):
        batch_size = min(batch_size, max_cat(net, torch.empty((1, row_ch, *patch_size), device=dev), dev))

    while pending:
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        rows = []
        for ci, _ijk, sl, extra in batch:
            row = torch.empty((row_ch, *patch_size), device=dev, dtype=torch.float32)
            encode_inference_row(
                row, pad, sl[0], sl[1], sl[2], n_img, seeds_pts[ci], encode_prompt,
                cfg, patch_size, dev, extra_clicks=extra, **enc_kw,
            )
            rows.append(row)
        with autocast(dev.type, enabled=amp_on):
            out = predict_batch_with_tta(net, torch.stack(rows), use_tta)
        for j, (ci, ijk, sl, _extra) in enumerate(batch):
            raw = out[j].float()
            sz, sy, sx = sl
            csz = slice(sz.start - canvas_origin[0], sz.stop - canvas_origin[0])
            csy = slice(sy.start - canvas_origin[1], sy.stop - canvas_origin[1])
            csx = slice(sx.start - canvas_origin[2], sx.stop - canvas_origin[2])
            m = (raw[1:].amax(0) - raw[0]).to(acc_dtype)
            sub_m = margin_buf[csz, csy, csx]
            keep = m > sub_m
            logits_acc[:, csz, csy, csx] = torch.where(
                keep.unsqueeze(0), raw.to(acc_dtype), logits_acc[:, csz, csy, csx]
            )
            margin_buf[csz, csy, csx] = torch.where(keep, m, sub_m)
            written.append(sl)
            fwd_done += 1
            if on_forward is not None:
                on_forward(fwd_done, fwd_done + len(pending))
            if not border_expand or extras_done[ci] >= max_border_expand_extra:
                continue
            for nijk, face_i in face_neighbours(ijk, fg_face_touch(raw, lm)):
                if extras_done[ci] >= max_border_expand_extra:
                    break
                key = (ci, nijk)
                if key in visited:
                    continue
                nsl = cell_slices(origins[ci], nijk, stride, patch_size, padded_shape)
                if (nsl[0].start, nsl[1].start, nsl[2].start) == (sz.start, sy.start, sx.start):
                    continue
                gclick = face_fg_click_global(raw, sl, face_i, lm)
                if gclick is None:
                    continue
                visited.add(key)
                extra_c = (extra_click_in_tile(gclick, nsl, patch_size),)
                pending.append((ci, nijk, nsl, extra_c))
                extras_done[ci] += 1

    tiles: list[tuple[slice, slice, slice]] = []
    seen_u: set = set()
    for sl in written:
        ov = patch_unpadded_overlap(sl[0], sl[1], sl[2], slicer_revert)
        if ov is None:
            continue
        u = ov[0]
        key = (u[0].start, u[0].stop, u[1].start, u[1].stop, u[2].start, u[2].stop)
        if key in seen_u:
            continue
        seen_u.add(key)
        tiles.append(u)
    # argmax on device (GPU ~1.5s vs ~50s CPU C-first); D2H is uint8 labels, not logits.
    # Canvas may be smaller than the unpadded crop — everywhere outside it is background by
    # construction (bg_vec argmax == 0), so start from zeros and only fill the overlap.
    seg = torch.zeros(unpadded_shape, dtype=torch.uint8)
    ov = patch_unpadded_overlap(canvas_sl[0], canvas_sl[1], canvas_sl[2], slicer_revert)
    if ov is not None:
        (uz, uy, ux), (cz, cy, cx) = ov
        seg[uz, uy, ux] = logits_acc[:, cz, cy, cx].float().argmax(0).to(torch.uint8).cpu()
    return seg, tiles
