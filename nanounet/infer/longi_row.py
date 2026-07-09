"""Two-stream inference row on a JOINT 2-channel volume (ch0 FU CT, ch1 BL CT sharing one
preprocessing crop, so FU/BL are voxel-aligned — same grid as training's build_patch_longi).
BL stream = same-bbox crop of ch1 + all in-patch BL clicks. Null baseline (no BL image, or prompts
disabled) duplicates the FU stream -> DWB(x_FU - x_FU)=0 -> identity (single-timepoint fallback)."""

from __future__ import annotations

import torch

from nanounet.infer.roi_slices import local_prompt_points_for_patch
from nanounet.prompt.cluster import cluster_prompts_patch_local
from nanounet.prompt.encoding import encode_points_to_heatmap_pair


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
    *,
    is_longi: bool = False,
    bl_present: bool = False,
    bl_pts_pad: list[tuple[int, int, int]] | None = None,
) -> None:
    n_stream = n_img + 2
    row[:n_img] = pad[:n_img, sz, sy, sx]
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
    if not is_longi:
        return
    # Null baseline: duplicate FU -> identity DWB (matches training force_zero_prompt / not has_bl).
    if not bl_present or not encode_prompt:
        row[n_stream:] = row[:n_stream]
        return
    # Real baseline: same bbox crops ch1 because the joint 2-ch crop keeps FU/BL voxel-aligned.
    row[n_stream : n_stream + n_img] = pad[n_img : 2 * n_img, sz, sy, sx]
    bl_local = cluster_prompts_patch_local(bl_pts_pad, sz, sy, sx) if bl_pts_pad else []
    bl_pr = encode_points_to_heatmap_pair(
        bl_local, [], patch_size, cfg.prompt.point_radius_vox, cfg.prompt.encoding,
        device=dev, intensity_scale=cfg.prompt.prompt_intensity_scale,
    )
    row[n_stream + n_img :] = bl_pr.float()
