"""Bottleneck-grid random masks upsampled to voxel resolution (Spark3D-style)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bottleneck_mask(
    patch_size: tuple[int, int, int] | list[int],
    total_stride: tuple[int, int, int] | list[int],
    mask_ratio: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    ps = tuple(int(x) for x in patch_size)
    ts = tuple(int(x) for x in total_stride)
    for p, s in zip(ps, ts):
        assert p % s == 0, f"patch {p} not divisible by stride {s}"
    grid = tuple(p // s for p, s in zip(ps, ts))
    n_cells = grid[0] * grid[1] * grid[2]
    n_masked = int(round(mask_ratio * n_cells))
    assert 0 <= n_masked <= n_cells
    flat = torch.zeros(batch_size, n_cells, device=device)
    for b in range(batch_size):
        idx = torch.randperm(n_cells, device=device)[:n_masked]
        flat[b, idx] = 1.0
    m = flat.view(batch_size, 1, *grid)
    return F.interpolate(m, size=ps, mode="nearest")
