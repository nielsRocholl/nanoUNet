"""RobustCrossEntropyLoss: float target with extra dim (nnU-Net port)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
