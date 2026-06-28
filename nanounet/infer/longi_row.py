"""Two-stream inference row encoding: FU half + co-located BL half (exact BL click)."""

from __future__ import annotations

import torch

from nanounet.infer.roi_slices import (
    colocated_spatial_slices,
    local_prompt_points_for_patch,
)
from nanounet.prompt.cluster import cluster_prompts_patch_local
from nanounet.prompt.encoding import encode_points_to_heatmap_pair


def bl_partner_for_cluster(
    cl: list[tuple[int, int, int]],
    pts_pad: list[tuple[int, int, int]],
    bl_pts_pad: list | None,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int]]:
    if bl_pts_pad is None:
        return None, cl[0]
    for p in cl:
        try:
            i = pts_pad.index(p)
        except ValueError:
            continue
        bp = bl_pts_pad[i]
        if bp is not None:
            return bp, p
    return None, cl[0]


def encode_inference_row(
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
    pad_bl: torch.Tensor | None = None,
    bl_partner: tuple[int, int, int] | None = None,
    bl_anchor: tuple[int, int, int] | None = None,
) -> None:
    n_stream = n_img + 2
    row[:n_img] = pad[:, sz, sy, sx]
    if not encode_prompt:
        row[n_img:n_stream].zero_()
    else:
        loc = cluster_prompts_patch_local(cluster, sz, sy, sx)
        if not loc:
            loc = local_prompt_points_for_patch(cluster[0], sz, sy, sx, patch_size)
        pr = encode_points_to_heatmap_pair(
            loc, [], patch_size, cfg.prompt.point_radius_vox, cfg.prompt.encoding,
            device=dev, intensity_scale=cfg.prompt.prompt_intensity_scale,
        )
        row[n_img:n_stream] = pr.float()
    if pad_bl is None:
        return
    fu_half = row[:n_stream]
    if bl_partner is None or bl_anchor is None:
        row[n_stream:] = fu_half
        return
    fu_local = (bl_anchor[0] - sz.start, bl_anchor[1] - sy.start, bl_anchor[2] - sx.start)
    bz, by, bx = colocated_spatial_slices(bl_partner, fu_local, patch_size, tuple(pad_bl.shape[1:]))
    row[n_stream : n_stream + n_img] = pad_bl[:, bz, by, bx]
    bl_pr = encode_points_to_heatmap_pair(
        [fu_local], [], patch_size, cfg.prompt.point_radius_vox, cfg.prompt.encoding,
        device=dev, intensity_scale=cfg.prompt.prompt_intensity_scale,
    )
    row[n_stream + n_img :] = bl_pr.float()
