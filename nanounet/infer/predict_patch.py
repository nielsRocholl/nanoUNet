"""Single centered patch forward for interactive viewers (TTA/border-expand off hot path)."""

from __future__ import annotations

import torch
from torch.amp import autocast

from nanounet.infer.longi_row import encode_inference_row
from nanounet.infer.roi_slices import centered_spatial_slices_at_point
from nanounet.infer.tta import predict_batch_with_tta


def predict_patch_logits(
    *,
    net,
    pad: torch.Tensor,
    point_zyx_padded: tuple[int, int, int],
    cfg,
    patch_size: tuple[int, int, int],
    dev: torch.device,
    encode_prompt: bool = True,
    use_tta: bool = False,
    use_amp: bool = True,
    is_longi: bool = False,
    bl_present: bool = False,
    bl_pts_pad: list[tuple[int, int, int]] | None = None,
) -> tuple[torch.Tensor, tuple[slice, slice, slice]]:
    """Centered crop + optional prompt heatmaps → one forward.

    Returns (logits (C,Z,Y,X) on CPU float32, padded spatial slices).
    """
    padded_shape = tuple(pad.shape[1:])
    sz, sy, sx = centered_spatial_slices_at_point(
        point_zyx_padded[0], point_zyx_padded[1], point_zyx_padded[2], patch_size, padded_shape
    )
    n_img = pad.shape[0] // 2 if (is_longi and bl_present) else pad.shape[0]
    n_stream = n_img + 2
    row_ch = 2 * n_stream if is_longi else n_stream
    row = torch.empty((row_ch, *patch_size), device=dev, dtype=torch.float32)
    encode_inference_row(
        row,
        pad,
        sz,
        sy,
        sx,
        n_img,
        [point_zyx_padded],
        encode_prompt,
        cfg,
        patch_size,
        dev,
        is_longi=is_longi,
        bl_present=bl_present,
        bl_pts_pad=bl_pts_pad,
    )
    amp_on = use_amp and getattr(dev, "type", str(dev)) == "cuda"
    with torch.inference_mode():
        with autocast(dev.type if hasattr(dev, "type") else "cpu", enabled=amp_on):
            out = predict_batch_with_tta(net, row.unsqueeze(0), use_tta)
    logits = out[0].float().cpu() if out.ndim == 5 else out.float().cpu()
    return logits, (sz, sy, sx)
