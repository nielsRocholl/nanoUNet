"""Batched single-case logits: cluster seeds mini-batched, per-cluster border BFS."""

from __future__ import annotations

import os
from collections import deque

import torch
from torch import autocast

from nanounet.infer.border_expand import plan_border_expansion_centers_from_logits
from nanounet.infer.gaussian import gaussian_tile
from nanounet.infer.longi_row import bl_partner_for_cluster, encode_inference_row
from nanounet.infer.roi_slices import (
    background_logits_vector,
    centered_spatial_slices_at_point,
    map_points_zyx_unpadded_to_padded,
    safe_divide_merged_logits,
    spatial_slices_to_tuple,
)
from nanounet.infer.tta import predict_batch_with_tta
from nanounet.prompt.cluster import cluster_points_for_patch_size, spatial_slices_covering_points
from nanounet.prompt.coords import points_to_centers_zyx

ACC_DTYPE_ENV = "NANOUNET_SINGLE_PATCH_ACCUM_DTYPE"


def _acc_dtype(dev: torch.device) -> torch.dtype:
    if dev.type == "cpu":
        return torch.float32
    r = (os.environ.get(ACC_DTYPE_ENV) or "").lower()
    return torch.float16 if r in ("half", "float16", "fp16") else torch.float32


def _map_bl_pts_pad(bl_points_xyz, bl_props, bl_slicer_revert, cm, pl):
    bl_unpadded = tuple(s.stop - s.start for s in bl_slicer_revert[1:])
    bl_pts_pad = [None] * len(bl_points_xyz)
    for i, pt in enumerate(bl_points_xyz):
        if pt is None:
            continue
        x, y, z = pt
        pre_i = points_to_centers_zyx(
            [(z, y, x)], "voxel", bl_props, bl_unpadded, tuple(cm.spacing), pl.transpose_forward,
            voxel_coordinate_frame="full",
        )
        bl_pts_pad[i] = map_points_zyx_unpadded_to_padded(pre_i, bl_slicer_revert)[0]
    return bl_pts_pad


@torch.inference_mode()
def predict_case_logits(
    *,
    net,
    lm,
    cfg,
    pl,
    cm,
    dj: dict,
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
    merge: str = "max",
    pad_bl: torch.Tensor | None = None,
    bl_points_xyz: list | None = None,
    bl_props: dict | None = None,
    bl_slicer_revert: tuple | None = None,
) -> torch.Tensor:
    assert mode in ("clustered", "centered")
    assert merge in ("max", "average")
    patch_size = tuple(cm.patch_size)
    padded_shape = tuple(pad.shape[1:])
    unpadded_shape = tuple(s.stop - s.start for s in slicer_revert[1:])
    nh = lm.num_segmentation_heads
    n_img = pad.shape[0]
    n_stream = n_img + 2
    row_ch = 2 * n_stream if pad_bl is not None else n_stream
    acc_dtype = _acc_dtype(dev)
    gaussian = gaussian_tile(patch_size, dev, torch.float32)
    gaussian_acc = gaussian.to(acc_dtype)
    bg_vec = background_logits_vector(lm, nh, dev, acc_dtype)
    amp_on = use_amp and dev.type == "cuda"

    if not points_xyz:
        return bg_vec.view(-1, 1, 1, 1).expand(nh, *unpadded_shape).contiguous().float().cpu()

    pts_zyx = [(z, y, x) for x, y, z in points_xyz]
    pre = points_to_centers_zyx(
        pts_zyx, "voxel", props, unpadded_shape, tuple(cm.spacing), pl.transpose_forward,
        voxel_coordinate_frame="full",
    )
    pts_pad = map_points_zyx_unpadded_to_padded(pre, slicer_revert)
    bl_pts_pad = None
    if pad_bl is not None and bl_points_xyz is not None and bl_props is not None and bl_slicer_revert is not None:
        bl_pts_pad = _map_bl_pts_pad(bl_points_xyz, bl_props, bl_slicer_revert, cm, pl)

    if mode == "clustered":
        seeds_pts = cluster_points_for_patch_size(pts_pad, patch_size, cluster_margin_frac)
        seed_slices = [spatial_slices_covering_points(cl, patch_size, padded_shape) for cl in seeds_pts]
    else:
        seen: set = set()
        seeds_pts, seed_slices = [], []
        for p in pts_pad:
            sl = centered_spatial_slices_at_point(p[0], p[1], p[2], patch_size, padded_shape)
            key = (spatial_slices_to_tuple(*sl), p)
            if key in seen:
                continue
            seen.add(key)
            seeds_pts.append([p])
            seed_slices.append(sl)

    rows, bl_meta = [], []
    for cl, (sz, sy, sx) in zip(seeds_pts, seed_slices):
        row = torch.empty((row_ch, *patch_size), device=dev, dtype=torch.float32)
        bp, anchor = bl_partner_for_cluster(cl, pts_pad, bl_pts_pad)
        encode_inference_row(row, pad, sz, sy, sx, n_img, cl, encode_prompt, cfg, patch_size, dev,
                             pad_bl, bp, anchor if bp is not None else None)
        rows.append(row)
        bl_meta.append((bp, anchor))

    seed_raw: list[torch.Tensor] = []
    for i in range(0, len(rows), batch_size):
        with autocast(dev.type, enabled=amp_on):
            out = predict_batch_with_tta(net, torch.stack(rows[i : i + batch_size]), use_tta)
        for j in range(out.shape[0]):
            seed_raw.append(out[j].float())

    use_max = merge == "max"
    if use_max:
        neg = torch.finfo(acc_dtype).min
        margin_buf = torch.full(padded_shape, neg, dtype=acc_dtype, device=dev)
        logits_acc = bg_vec.view(-1, 1, 1, 1).to(acc_dtype).expand(nh, *padded_shape).contiguous()
    else:
        logits_acc = torch.zeros((nh, *padded_shape), dtype=acc_dtype, device=dev)
        n_pred = torch.zeros(padded_shape, dtype=acc_dtype, device=dev)

    def accumulate(raw: torch.Tensor, sz: slice, sy: slice, sx: slice) -> None:
        if use_max:
            m = (raw[1:].amax(0) - raw[0]).to(acc_dtype)
            sub_m = margin_buf[sz, sy, sx]
            keep = m > sub_m
            logits_acc[:, sz, sy, sx] = torch.where(keep.unsqueeze(0), raw.to(acc_dtype), logits_acc[:, sz, sy, sx])
            margin_buf[sz, sy, sx] = torch.where(keep, m, sub_m)
        else:
            logits_acc[:, sz, sy, sx] += (raw * gaussian).to(acc_dtype)
            n_pred[sz, sy, sx] += gaussian_acc

    for i, (sz, sy, sx) in enumerate(seed_slices):
        accumulate(seed_raw[i], sz, sy, sx)

    if border_expand:
        for i, cl in enumerate(seeds_pts):
            sz, sy, sx = seed_slices[i]
            raw, seed_key = seed_raw[i], spatial_slices_to_tuple(*seed_slices[i])
            bp, anchor = bl_meta[i]
            vis = {seed_key}
            q = deque(plan_border_expansion_centers_from_logits(
                raw, lm, sz, sy, sx, patch_size, padded_shape, seed_key, max_border_expand_extra, skip_keys=vis))
            done = 0
            while q and done < max_border_expand_extra:
                pe, pye, pxe = q.popleft()
                sze, sye, sxe = centered_spatial_slices_at_point(pe, pye, pxe, patch_size, padded_shape)
                ke = spatial_slices_to_tuple(sze, sye, sxe)
                if ke in vis:
                    continue
                vis.add(ke)
                workon = torch.empty((1, row_ch, *patch_size), device=dev, dtype=torch.float32)
                encode_inference_row(workon[0], pad, sze, sye, sxe, n_img, cl, encode_prompt, cfg, patch_size, dev,
                                     pad_bl, bp, anchor if bp is not None else None)
                with autocast(dev.type, enabled=amp_on):
                    raw_e = predict_batch_with_tta(net, workon, use_tta)[0].float()
                accumulate(raw_e, sze, sye, sxe)
                done += 1
                bud = max_border_expand_extra - done
                if bud > 0:
                    for c in plan_border_expansion_centers_from_logits(
                        raw_e, lm, sze, sye, sxe, patch_size, padded_shape, seed_key, bud, skip_keys=vis):
                        q.append(c)

    if not use_max:
        safe_divide_merged_logits(logits_acc, n_pred, bg_vec)
    return logits_acc[(slice(None), *slicer_revert[1:])].float().cpu()
