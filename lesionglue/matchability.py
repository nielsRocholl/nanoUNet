"""LightGlue-style permutation-invariant row matchability -> dustbin logit."""

from __future__ import annotations

import torch
from torch import nn

from lesionglue.train.sinkhorn import log_sinkhorn, superglue_marginals


def row_dust_marginals(
    pair: torch.Tensor, dust_bl: torch.Tensor, dust_fu: torch.Tensor, n_bl: int, n_fu: int, iters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn dustbin marginal per BL row / FU col: P(row->dustbin).

    A matchability prior fed back into the dust head. Detached (pure input feature)
    and computed with the supplied dustbin scores (zeros => parameter-free prior),
    so it does not collapse together with the learned dustbin logits it conditions.
    """
    S = torch.zeros((n_bl + 1, n_fu + 1), device=pair.device, dtype=pair.dtype)
    S[:n_bl, :n_fu] = pair.reshape(n_bl, n_fu).detach()
    S[:n_bl, n_fu] = dust_bl.detach()
    S[n_bl, :n_fu] = dust_fu.detach()
    la, lb = superglue_marginals(n_bl, n_fu, pair.device, pair.dtype)
    P = log_sinkhorn(S, iters, la, lb).exp()
    bl = P[:n_bl, n_fu] / P[:n_bl].sum(dim=1).clamp_min(1e-9)
    fu = P[n_bl, :n_fu] / P[:, :n_fu].sum(dim=0).clamp_min(1e-9)
    return bl, fu


class RowMatchability(nn.Module):
    """Dustbin logit from sorted top-k row pair logits, attention-pooled context over
    the opposite side, and the row's Sinkhorn dustbin marginal. Heavy dropout: the
    dustbin head over-fits tiny data, decaying late disappeared/newly-appearing acc.
    """

    def __init__(self, d: int, topk: int = 5, ctx: int = 16, drop: float = 0.5):
        super().__init__()
        self.topk = topk
        self.val = nn.Linear(d, ctx)
        h = max(8, d // 2)
        self.net = nn.Sequential(
            nn.Linear(d + topk + ctx + 1, h), nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(h, 1)
        )

    def forward(self, z_self: torch.Tensor, M: torch.Tensor, marginal: torch.Tensor, z_other: torch.Tensor) -> torch.Tensor:
        n_self, n_other = M.shape
        assert n_other >= 1
        k = min(self.topk, n_other)
        top = M.topk(k, dim=1).values
        if k < self.topk:  # pad short rows with the smallest selected logit (no spikes)
            top = torch.cat([top, top[:, -1:].expand(n_self, self.topk - k)], dim=1)
        ctx = torch.softmax(M, dim=1) @ self.val(z_other)
        feat = torch.cat([z_self, top, ctx, marginal.unsqueeze(1)], dim=1)
        return self.net(feat).squeeze(-1)
