"""Batched single-case logits: cluster seeds mini-batched, per-cluster border BFS."""

from __future__ import annotations

from collections import deque

import torch
from torch import autocast

from nanounet.infer.border_expand import plan_border_expansion_centers_from_logits
from nanounet.infer.gaussian import accum_dtype, gaussian_tile
from nanounet.infer.longi_row import encode_inference_row
from nanounet.infer.points_pad import resolve_pts_pad
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
    is_longi: bool = False,
    bl_present: bool = False,
    bl_points_xyz: list | None = None,
    points_zyx_unpadded: list[tuple[int, int, int]] | None = None,
    on_forward: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    assert mode in ("clustered", "centered")
    assert merge in ("max", "average")
    patch_size = tuple(cm.patch_size)
    padded_shape = tuple(pad.shape[1:])
    unpadded_shape = tuple(s.stop - s.start for s in slicer_revert[1:])
    nh = lm.num_segmentation_heads
    n_img = pad.shape[0] // 2 if (is_longi and bl_present) else pad.shape[0]
    n_stream = n_img + 2
    row_ch = 2 * n_stream if is_longi else n_stream
    acc_dtype = accum_dtype(dev)
    gaussian = gaussian_tile(patch_size, dev, torch.float32)
    gaussian_acc = gaussian.to(acc_dtype)
    bg_vec = background_logits_vector(lm, nh, dev, acc_dtype)
    amp_on = use_amp and dev.type == "cuda"
    spacing = tuple(cm.spacing)
    tf = pl.transpose_forward

    pts_pad = resolve_pts_pad(
        points_xyz=points_xyz,
        points_zyx_unpadded=points_zyx_unpadded,
        props=props,
        unpadded_shape=unpadded_shape,
        spacing=spacing,
        transpose_forward=tf,
        slicer_revert=slicer_revert,
    )
    if not pts_pad:
        return bg_vec.view(-1, 1, 1, 1).expand(nh, *unpadded_shape).contiguous().float().cpu()

    bl_pts_pad = None
    if is_longi and bl_present and bl_points_xyz:
        bl_zyx = [(z, y, x) for x, y, z in bl_points_xyz]
        bl_pre = points_to_centers_zyx(
            bl_zyx, "voxel", props, unpadded_shape, spacing, tf, voxel_coordinate_frame="full",
        )
        bl_pts_pad = map_points_zyx_unpadded_to_padded(bl_pre, slicer_revert)

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

    rows = []
    for cl, (sz, sy, sx) in zip(seeds_pts, seed_slices):
        row = torch.empty((row_ch, *patch_size), device=dev, dtype=torch.float32)
        encode_inference_row(
            row, pad, sz, sy, sx, n_img, cl, encode_prompt, cfg, patch_size, dev,
            is_longi=is_longi, bl_present=bl_present, bl_pts_pad=bl_pts_pad,
        )
        rows.append(row)

    fwd_done, fwd_tot = 0, len(rows)

    def _tick() -> None:
        nonlocal fwd_done
        fwd_done += 1
        if on_forward is not None:
            on_forward(fwd_done, fwd_tot)

    seed_raw: list[torch.Tensor] = []
    for i in range(0, len(rows), batch_size):
        with autocast(dev.type, enabled=amp_on):
            out = predict_batch_with_tta(net, torch.stack(rows[i : i + batch_size]), use_tta)
        for j in range(out.shape[0]):
            seed_raw.append(out[j].float())
            _tick()

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
                fwd_tot += 1
                workon = torch.empty((1, row_ch, *patch_size), device=dev, dtype=torch.float32)
                encode_inference_row(
                    workon[0], pad, sze, sye, sxe, n_img, cl, encode_prompt, cfg, patch_size, dev,
                    is_longi=is_longi, bl_present=bl_present, bl_pts_pad=bl_pts_pad,
                )
                with autocast(dev.type, enabled=amp_on):
                    raw_e = predict_batch_with_tta(net, workon, use_tta)[0].float()
                accumulate(raw_e, sze, sye, sxe)
                _tick()
                done += 1
                bud = max_border_expand_extra - done
                if bud > 0:
                    for c in plan_border_expansion_centers_from_logits(
                        raw_e, lm, sze, sye, sxe, patch_size, padded_shape, seed_key, bud, skip_keys=vis):
                        q.append(c)

    if not use_max:
        safe_divide_merged_logits(logits_acc, n_pred, bg_vec)
    return logits_acc[(slice(None), *slicer_revert[1:])].float().cpu()
