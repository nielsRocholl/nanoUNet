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
