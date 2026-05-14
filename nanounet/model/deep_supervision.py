"""Downscale deep supervision targets + weight channels (nnU-Net port)."""

from __future__ import annotations

from torch import nn


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors: tuple):
        super().__init__()
        assert any(x != 0 for x in weight_factors)
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args):
        assert all(isinstance(i, (tuple, list)) for i in args)
        w = self.weight_factors
        return sum(w[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if w[i] != 0.0)
