"""Batched single-case logits: cluster seeds mini-batched, per-cluster border BFS."""

from __future__ import annotations

import os
from collections import deque

import torch
from torch import autocast

from nanounet.infer.border_expand import plan_border_expansion_centers_from_logits
from nanounet.infer.gaussian import gaussian_tile
from nanounet.infer.roi_slices import (
    background_logits_vector,
    centered_spatial_slices_at_point,
    local_prompt_points_for_patch,
    map_points_zyx_unpadded_to_padded,
    safe_divide_merged_logits,
    spatial_slices_to_tuple,
)
from nanounet.infer.tta import predict_batch_with_tta
from nanounet.prompt.cluster import (
    cluster_points_for_patch_size,
    cluster_prompts_patch_local,
    spatial_slices_covering_points,
)
from nanounet.prompt.coords import points_to_centers_zyx
from nanounet.prompt.encoding import encode_points_to_heatmap_pair

ACC_DTYPE_ENV = "NANOUNET_SINGLE_PATCH_ACCUM_DTYPE"


def _acc_dtype(dev: torch.device) -> torch.dtype:
    if dev.type == "cpu":
        return torch.float32
    r = (os.environ.get(ACC_DTYPE_ENV) or "").lower()
    return torch.float16 if r in ("half", "float16", "fp16") else torch.float32


def _encode_row(
    row: torch.Tensor,
    pad: torch.Tensor,
    sz: slice,
    sy: slice,
    sx: slice,
    n_img: int,
    cluster: list[tuple[int, int, int]],
    encode_prompt: bool,
    cfg,
    patch_size: tuple[int, int, int],
    dev: torch.device,
) -> None:
    row[:n_img] = pad[:, sz, sy, sx]
    if not encode_prompt:
        row[n_img:].zero_()
        return
    loc = cluster_prompts_patch_local(cluster, sz, sy, sx)
    if not loc:
        loc = local_prompt_points_for_patch(cluster[0], sz, sy, sx, patch_size)
    pr = encode_points_to_heatmap_pair(
        loc,
        [],
        patch_size,
        cfg.prompt.point_radius_vox,
        cfg.prompt.encoding,
        device=dev,
        intensity_scale=cfg.prompt.prompt_intensity_scale,
    )
    row[n_img:] = pr.float()


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
) -> torch.Tensor:
    assert mode in ("clustered", "centered")
    patch_size = tuple(cm.patch_size)
    padded_shape = tuple(pad.shape[1:])
    unpadded_shape = tuple(s.stop - s.start for s in slicer_revert[1:])
    nh = lm.num_segmentation_heads
    n_img = pad.shape[0]
    acc_dtype = _acc_dtype(dev)
    gaussian = gaussian_tile(patch_size, dev, torch.float32)
    gaussian_acc = gaussian.to(acc_dtype)
    bg_vec = background_logits_vector(lm, nh, dev, acc_dtype)
    use_mirroring = use_tta
    amp_on = use_amp and dev.type == "cuda"

    if not points_xyz:
        return bg_vec.view(-1, 1, 1, 1).expand(nh, *unpadded_shape).contiguous().float().cpu()

    pts_zyx = [(z, y, x) for x, y, z in points_xyz]
    pre = points_to_centers_zyx(
        pts_zyx,
        "voxel",
        props,
        unpadded_shape,
        tuple(cm.spacing),
        pl.transpose_forward,
        voxel_coordinate_frame="full",
    )
    pts_pad = map_points_zyx_unpadded_to_padded(pre, slicer_revert)
    if mode == "clustered":
        seeds_pts = cluster_points_for_patch_size(pts_pad, patch_size, cluster_margin_frac)
        seed_slices = [
            spatial_slices_covering_points(cl, patch_size, padded_shape) for cl in seeds_pts
        ]
    else:  # centered: one patch per click, prompt = that single click only
        seen: set = set()
        seeds_pts, seed_slices = [], []
        for p in pts_pad:
            sl = centered_spatial_slices_at_point(p[0], p[1], p[2], patch_size, padded_shape)
            key = (spatial_slices_to_tuple(*sl), p)
            if key in seen:  # exact-duplicate input clicks only
                continue
            seen.add(key)
            seeds_pts.append([p])
            seed_slices.append(sl)

    rows: list[torch.Tensor] = []
    for cl, (sz, sy, sx) in zip(seeds_pts, seed_slices):
        row = torch.empty((n_img + 2, *patch_size), device=dev, dtype=torch.float32)
        _encode_row(row, pad, sz, sy, sx, n_img, cl, encode_prompt, cfg, patch_size, dev)
        rows.append(row)
    seed_raw: list[torch.Tensor] = []
    for i in range(0, len(rows), batch_size):
        chunk = torch.stack(rows[i : i + batch_size])
        with autocast(dev.type, enabled=amp_on):
            out = predict_batch_with_tta(net, chunk, use_mirroring)
        out = out.float()
        for j in range(out.shape[0]):
            seed_raw.append(out[j])

    logits_acc = torch.zeros((nh, *padded_shape), dtype=acc_dtype, device=dev)
    n_pred = torch.zeros(padded_shape, dtype=acc_dtype, device=dev)
    for i, (sz, sy, sx) in enumerate(seed_slices):
        logits_acc[:, sz, sy, sx] += (seed_raw[i] * gaussian).to(acc_dtype)
        n_pred[sz, sy, sx] += gaussian_acc

    if border_expand:
        for i, cl in enumerate(seeds_pts):
            sz, sy, sx = seed_slices[i]
            raw = seed_raw[i]
            seed_key = spatial_slices_to_tuple(sz, sy, sx)
            vis = {seed_key}
            q = deque(
                plan_border_expansion_centers_from_logits(
                    raw, lm, sz, sy, sx, patch_size, padded_shape, seed_key, max_border_expand_extra, skip_keys=vis
                )
            )
            done = 0
            while q and done < max_border_expand_extra:
                pe, pye, pxe = q.popleft()
                sze, sye, sxe = centered_spatial_slices_at_point(pe, pye, pxe, patch_size, padded_shape)
                ke = spatial_slices_to_tuple(sze, sye, sxe)
                if ke in vis:
                    continue
                vis.add(ke)
                workon = torch.empty((1, n_img + 2, *patch_size), device=dev, dtype=torch.float32)
                _encode_row(workon[0], pad, sze, sye, sxe, n_img, cl, encode_prompt, cfg, patch_size, dev)
                with autocast(dev.type, enabled=amp_on):
                    raw_e = predict_batch_with_tta(net, workon, use_mirroring)[0].float()
                logits_acc[:, sze, sye, sxe] += (raw_e * gaussian).to(acc_dtype)
                n_pred[sze, sye, sxe] += gaussian_acc
                done += 1
                bud = max_border_expand_extra - done
                if bud > 0:
                    for c in plan_border_expansion_centers_from_logits(
                        raw_e, lm, sze, sye, sxe, patch_size, padded_shape, seed_key, bud, skip_keys=vis
                    ):
                        q.append(c)

    safe_divide_merged_logits(logits_acc, n_pred, bg_vec)
    logits_acc = logits_acc[(slice(None), *slicer_revert[1:])]
    return logits_acc.float().cpu()
