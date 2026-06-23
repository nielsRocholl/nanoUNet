"""Test-time mirroring over spatial axes (combinations, nnU-Net TTA)."""

from __future__ import annotations

import itertools

import torch


@torch.inference_mode()
def predict_with_optional_tta(net: torch.nn.Module, x: torch.Tensor, use_mirroring: bool, mirror_axes=(0, 1, 2)):
    if not use_mirroring:
        return net(x)
    ma = [m + 2 for m in mirror_axes]
    combs = [c for i in range(len(ma)) for c in itertools.combinations(ma, i + 1)]
    pred = net(x)
    for axes in combs:
        pred = pred + torch.flip(net(torch.flip(x, axes)), axes)
    pred = pred / (len(combs) + 1)
    return pred


@torch.inference_mode()
def predict_batch_with_tta(net: torch.nn.Module, x: torch.Tensor, use_mirroring: bool, mirror_axes=(0, 1, 2)):
    """x: (B, C, Z, Y, X) -> (B, n_heads, Z, Y, X), mirror-averaged."""
    if not use_mirroring:
        return net(x)
    ma = [m + 2 for m in mirror_axes]
    combs = [c for i in range(len(ma)) for c in itertools.combinations(ma, i + 1)]
    out = net(x)
    for axes in combs:
        out = out + torch.flip(net(torch.flip(x, axes)), axes)
    return out / (len(combs) + 1)
