"""CC-DiceCE: global Dice+CE plus per-connected-component Dice+CE over GT Voronoi cells.

Adds instance-balanced gradients (small lesions weighted up vs plain DiceCE). Global term is
standard DC+CE (smooth Dice); CC term follows Bouteille et al. with ε=0 on Dice.
Uses CPU cc3d + scipy EDT — DS scales re-run labeling (shrunk targets), no shared CC cache.

Paper: Bouteille et al., 'Learning to Look Closer', ISBI 2026, arXiv:2511.17146.
"""

from __future__ import annotations

import cc3d
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

from nanounet.model.dice_helpers import get_tp_fp_fn_tn, softmax_helper_dim1
from nanounet.model.losses import DC_and_CE_loss
from nanounet.model.robust_ce import RobustCrossEntropyLoss


def _cc_voronoi(seg_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """binary (Z,Y,X) → (cc_labels, voronoi_id, n_fg_cc). vor==0 if n==0."""
    lbl, n = cc3d.connected_components(seg_bin.astype(np.uint8), connectivity=26, return_N=True)
    if n == 0:
        z = np.zeros_like(lbl, dtype=np.int32)
        return lbl.astype(np.int32), z, 0
    edts = np.stack([distance_transform_edt(lbl != i) for i in range(1, n + 1)], axis=0)
    vor = np.argmin(edts, axis=0).astype(np.int32) + 1
    return lbl.astype(np.int32), vor, int(n)


class CC_DC_and_CE_loss(nn.Module):
    """L = DC+CE (global) + λ · mean_{s,c} (DiceCE on Voronoi cell of CC c in sample s)."""

    def __init__(self, soft_dice_kwargs: dict, ce_kwargs: dict, ignore_label: int | None = None, lam: float = 1.0):
        super().__init__()
        self.lam = lam
        self.ignore_label = ignore_label
        g_kw = {**soft_dice_kwargs, "smooth": 1e-5}
        self.global_loss = DC_and_CE_loss(g_kw, ce_kwargs, ignore_label=ignore_label)
        self.batch_dice = bool(soft_dice_kwargs["batch_dice"])
        self.smooth_cc = float(soft_dice_kwargs["smooth"])
        ce_local = dict(ce_kwargs)
        if ignore_label is not None:
            ce_local["ignore_index"] = ignore_label
        self.ce = RobustCrossEntropyLoss(reduction="none", **ce_local)

    def forward(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.global_loss(out, target) + self.lam * self._cc_term(out, target)

    def _cc_term(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, _, *_ = out.shape
        probs = softmax_helper_dim1(out)
        axes_sp = tuple(range(2, out.ndim))
        dev = out.device
        tgt0 = target[:, 0]

        dice_terms: list[torch.Tensor] = []
        ce_terms: list[torch.Tensor] = []
        TP = torch.zeros((), device=dev, dtype=torch.float32)
        FP = torch.zeros((), device=dev, dtype=torch.float32)
        FN = torch.zeros((), device=dev, dtype=torch.float32)
        ce_sum_bt = torch.zeros((), device=dev, dtype=torch.float32)
        ce_cnt_bt = torch.zeros((), device=dev, dtype=torch.float32)

        for b in range(B):
            t_np = tgt0[b].detach().cpu().numpy().astype(np.int64)
            if self.ignore_label is not None:
                valid = t_np != self.ignore_label
            else:
                valid = np.ones_like(t_np, dtype=bool)
            bin_fg = (t_np != 0) & valid
            lbl, vor, n_cc = _cc_voronoi(bin_fg)
            if n_cc == 0:
                continue
            lbl_t = torch.as_tensor(lbl, device=dev, dtype=torch.long)
            vor_t = torch.as_tensor(vor, device=dev, dtype=torch.long)

            for cid in range(1, n_cc + 1):
                R = (vor_t == cid).float().unsqueeze(0).unsqueeze(0)
                g = (lbl_t == cid).long().unsqueeze(0).unsqueeze(0)
                if self.ignore_label is not None:
                    ign = (tgt0[b : b + 1] == self.ignore_label).float().unsqueeze(0)
                    R = R * (1.0 - ign)
                if float(R.sum().item()) == 0.0:
                    continue
                tp, fp, fn, _ = get_tp_fp_fn_tn(probs[b : b + 1], g, axes=axes_sp, mask=R)
                tp1, fp1, fn1 = tp[0, 1], fp[0, 1], fn[0, 1]
                if self.batch_dice:
                    TP = TP + tp1
                    FP = FP + fp1
                    FN = FN + fn1
                else:
                    dc = (2 * tp1 + self.smooth_cc) / (tp1 + fn1 + tp1 + fp1 + self.smooth_cc).clamp_min(1e-8)
                    dice_terms.append(-dc)

                ce_map = self.ce(out[b : b + 1], target[b : b + 1, 0])
                w = R.squeeze(1)
                ce_m = ce_map * w
                cnt = w.sum().clamp_min(1.0)
                if self.batch_dice:
                    ce_sum_bt = ce_sum_bt + ce_m.sum()
                    ce_cnt_bt = ce_cnt_bt + cnt
                else:
                    ce_terms.append(ce_m.sum() / cnt)

        if self.batch_dice:
            if ce_cnt_bt.item() == 0.0 and TP.item() == 0.0:
                return out.sum() * 0.0
            d_cc = (2 * TP + self.smooth_cc) / (TP + FN + TP + FP + self.smooth_cc).clamp_min(1e-8)
            return -d_cc + ce_sum_bt / ce_cnt_bt.clamp_min(1.0)

        if not dice_terms:
            return out.sum() * 0.0
        d_st = torch.stack(dice_terms)
        c_st = torch.stack(ce_terms)
        return (d_st + c_st).mean()
