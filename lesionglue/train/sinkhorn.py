"""Log-domain Sinkhorn + SuperGlue-style dustbin marginals for assignment."""

from __future__ import annotations

import math

import torch


def superglue_marginals(n_bl: int, n_fu: int, dev, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    norm = math.log(n_bl + n_fu)
    log_a = torch.zeros(n_bl + 1, device=dev, dtype=dtype)
    log_a[n_bl] = math.log(n_fu)
    log_b = torch.zeros(n_fu + 1, device=dev, dtype=dtype)
    log_b[n_fu] = math.log(n_bl)
    return log_a - norm, log_b - norm


def log_sinkhorn(
    M: torch.Tensor,
    iters: int = 20,
    log_a: torch.Tensor | None = None,
    log_b: torch.Tensor | None = None,
) -> torch.Tensor:
    r, c = M.shape
    if log_a is None:
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
    log_a, log_b = superglue_marginals(n_bl, n_fu, S.device, S.dtype)
    P = log_sinkhorn(S, iters, log_a, log_b)
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
