"""Dice + TP/FP/FN for loss and validation (nnU-Net dice.py port, ddp=False)."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, net_output.ndim))
    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))
        if net_output.shape == gt.shape:
            y_onehot = gt.to(torch.float32)
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.float32)
            y_onehot.scatter_(1, gt.long(), 1)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)
    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
    if square:
        tp, fp, fn, tn = tp**2, fp**2, fn**2, tn**2
    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False, dtype=torch.float32)
        fp = fp.sum(dim=axes, keepdim=False, dtype=torch.float32)
        fn = fn.sum(dim=axes, keepdim=False, dtype=torch.float32)
        tn = tn.sum(dim=axes, keepdim=False, dtype=torch.float32)
    return tp, fp, fn, tn


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable | None = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = False,
    ):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        assert not ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)
        if not self.do_bg:
            x = x[:, 1:]
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)
        if self.batch_dice:
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)
        dc = (2 * intersect + self.smooth) / (sum_gt + sum_pred + float(self.smooth)).clamp_min(1e-8)
        return -dc.mean()
