"""Point → heatmap (binary ball or EDT); pos/neg pair for two channels."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import ball


@lru_cache(maxsize=16)
def _build_ball_strel(radius: int, use_edt: bool) -> torch.Tensor:
    b = ball(radius, strict_radius=False)
    strel = torch.from_numpy(b.astype(np.float32))
    if use_edt:
        binary = (strel >= 0.5).numpy()
        edt = distance_transform_edt(binary)
        edt = edt / (edt.max() + 1e-8)
        strel = torch.from_numpy(edt.astype(np.float32))
    else:
        strel = (strel >= 0.5).float()
    return strel


def encode_points_to_heatmap(
    points_zyx: List[Tuple[int, int, int]],
    shape: Tuple[int, int, int],
    radius_vox: int,
    encoding: str,
    device: Union[torch.device, str, None] = None,
    intensity_scale: float = 1.0,
) -> torch.Tensor:
    heatmap = torch.zeros(shape, dtype=torch.float32, device=device)
    if not points_zyx:
        return heatmap
    use_edt = encoding == "edt"
    strel = _build_ball_strel(radius_vox, use_edt)
    r = radius_vox
    for pz, py, px in points_zyx:
        z0, z1 = max(0, pz - r), min(shape[0], pz + r + 1)
        y0, y1 = max(0, py - r), min(shape[1], py + r + 1)
        x0, x1 = max(0, px - r), min(shape[2], px + r + 1)
        sz0 = r - (pz - z0)
        sz1 = sz0 + (z1 - z0)
        sy0 = r - (py - y0)
        sy1 = sy0 + (y1 - y0)
        sx0 = r - (px - x0)
        sx1 = sx0 + (x1 - x0)
        slc = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        strel_slc = (slice(sz0, sz1), slice(sy0, sy1), slice(sx0, sx1))
        patch = strel[strel_slc].to(heatmap.device)
        torch.maximum(heatmap[slc], patch, out=heatmap[slc])
    return heatmap * intensity_scale


def encode_points_to_heatmap_pair(
    points_pos: List[Tuple[int, int, int]],
    points_neg: List[Tuple[int, int, int]],
    shape: Tuple[int, int, int],
    radius_vox: int,
    encoding: str,
    device: Union[torch.device, str, None] = None,
    intensity_scale: float = 1.0,
) -> torch.Tensor:
    pos = encode_points_to_heatmap(points_pos, shape, radius_vox, encoding, device, intensity_scale)
    neg = encode_points_to_heatmap(points_neg, shape, radius_vox, encoding, device, intensity_scale)
    return torch.stack([pos, neg], dim=0)
