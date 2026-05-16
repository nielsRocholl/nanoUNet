"""Log-domain Sinkhorn normalization and dustbin assignment loss (SuperGlue-style)."""

from __future__ import annotations

import math

import torch


def log_sinkhorn(M: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """M: (R, C) log-scores; marginals uniform 1/R and 1/C so total mass = 1."""
    r, c = M.shape
    log_a = torch.full((r,), -math.log(r), device=M.device, dtype=M.dtype)
    log_b = torch.full((c,), -math.log(c), device=M.device, dtype=M.dtype)
    u = torch.zeros(r, device=M.device, dtype=M.dtype)
    v = torch.zeros(c, device=M.device, dtype=M.dtype)
    for _ in range(iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(M + u.unsqueeze(1), dim=0)
    return M + u.unsqueeze(1) + v.unsqueeze(0)


def sinkhorn_loss(
    pair_logits: torch.Tensor,
    n_bl: int,
    n_fu: int,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    lab: torch.Tensor,
    iters: int = 20,
) -> torch.Tensor:
    S = torch.zeros((n_bl + 1, n_fu + 1), device=pair_logits.device, dtype=pair_logits.dtype)
    S[:n_bl, :n_fu] = pair_logits.reshape(n_bl, n_fu)
    S[:n_bl, n_fu] = dust_bl
    S[n_bl, :n_fu] = dust_fu
    P = log_sinkhorn(S, iters)
    pos = (lab.reshape(n_bl, n_fu) > 0.5)
    dev = S.device
    bl_tgt = torch.where(
        pos.any(dim=1),
        pos.float().argmax(dim=1),
        torch.full((n_bl,), n_fu, device=dev, dtype=torch.long),
    )
    fu_tgt = torch.where(
        pos.any(dim=0),
        pos.float().argmax(dim=0),
        torch.full((n_fu,), n_bl, device=dev, dtype=torch.long),
    )
    bl_term = -P[torch.arange(n_bl, device=dev), bl_tgt].mean()
    fu_term = -P[fu_tgt, torch.arange(n_fu, device=dev)].mean()
    return 0.5 * (bl_term + fu_term)
