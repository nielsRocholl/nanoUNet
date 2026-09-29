"""Single-timepoint inference row: image channels copied from the padded volume, prompt
heatmap channel(s) rendered from in-patch clicks.
extra_clicks: patch-local points used when no user click lands in the tile (expand face-FG)."""

from __future__ import annotations

import torch

from nanounet.prompt.cluster import cluster_prompts_patch_local
from nanounet.prompt.encoding import N_PROMPT_CHANNELS, encode_points_to_heatmap


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
    extra_clicks: tuple[tuple[int, int, int], ...] = (),
) -> None:
    n_stream = n_img + N_PROMPT_CHANNELS
    row[:n_img].copy_(pad[:n_img, sz, sy, sx], non_blocking=True)
    if not encode_prompt:
        row[n_img:n_stream].zero_()
        return
    loc = cluster_prompts_patch_local(cluster, sz, sy, sx)
    if not loc:
        loc = list(extra_clicks)
    assert loc, "seed tile must contain a click"
    pr = encode_points_to_heatmap(
        loc, patch_size, cfg.prompt.point_radius_vox, cfg.prompt.encoding,
        device=dev, intensity_scale=cfg.prompt.prompt_intensity_scale,
    ).unsqueeze(0)
    row[n_img:n_stream] = pr.float()
