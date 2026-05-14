"""DC + CE (+ deep supervision wrapper)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nanounet.model.deep_supervision import DeepSupervisionWrapper
from nanounet.model.dice_helpers import MemoryEfficientSoftDiceLoss, softmax_helper_dim1
from nanounet.model.robust_ce import RobustCrossEntropyLoss
from nanounet.plan.labels import Labels
from nanounet.plan.plans import Config3d


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs = {**ce_kwargs, "ignore_index": ignore_label}
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


def build_loss(
    cm: Config3d,
    lm: Labels,
    enable_ds: bool,
    *,
    loss_type: str = "dc_ce",
    is_ddp: bool = False,
) -> nn.Module:
    assert not is_ddp
    sd_kw = {"batch_dice": cm.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": False}
    if loss_type == "dc_ce":
        loss = DC_and_CE_loss(
            sd_kw,
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=lm.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
    elif loss_type == "cc_dc_ce":
        from nanounet.model.cc_dice_ce import CC_DC_and_CE_loss

        loss = CC_DC_and_CE_loss({**sd_kw, "smooth": 0.0}, {}, ignore_label=lm.ignore_label, lam=1.0)
    else:
        raise ValueError(loss_type)
    if not enable_ds:
        return loss
    pool = cm.pool_op_kernel_sizes
    scales = list(list(i) for i in 1 / np.cumprod(np.vstack(pool), axis=0))[:-1]
    w = np.array([1 / (2**i) for i in range(len(scales))])
    w[-1] = 0
    w = tuple(w / w.sum())
    return DeepSupervisionWrapper(loss, w)
