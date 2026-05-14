"""Gaussian importance map on patch grid (cached by size + device + dtype)."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


@lru_cache(maxsize=8)
def compute_gaussian(
    tile_size: Tuple[int, int, int],
    sigma_scale: float,
    value_scaling_factor: float,
    dtype_name: str,
    device_str: str,
) -> torch.Tensor:
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    dev = torch.device(device_str)
    tmp = np.zeros(tile_size)
    sigmas = [i * sigma_scale for i in tile_size]
    cc = [i // 2 for i in tile_size]
    tmp[tuple(cc)] = 1
    g = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    t = torch.from_numpy(g)
    t /= torch.max(t) / value_scaling_factor
    t = t.to(device=dev, dtype=dt)
    m = t == 0
    if m.any():
        t[m] = torch.min(t[~m])
    return t


def gaussian_tile(
    patch_size: Union[Tuple[int, ...], List[int]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    sigma_scale: float = 1.0 / 8,
    value_scaling: float = 10.0,
) -> torch.Tensor:
    ps = tuple(int(x) for x in patch_size)
    return compute_gaussian(ps, sigma_scale, value_scaling, str(dtype).replace("torch.", ""), str(device))
