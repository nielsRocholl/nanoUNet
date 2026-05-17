"""Sinkhorn decoding with SuperGlue marginals + optional Hungarian."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from tracking.train.sinkhorn import log_sinkhorn, superglue_marginals


def decode_sinkhorn(
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    n_bl: int,
    n_fu: int,
    iters: int = 20,
    tau: float = 0.2,
) -> np.ndarray:
    device, dtype = pair_log.device, pair_log.dtype
    S = torch.zeros((n_bl + 1, n_fu + 1), device=device, dtype=dtype)
    S[:n_bl, :n_fu] = pair_log.reshape(n_bl, n_fu)
    S[:n_bl, n_fu] = dust_bl
    S[n_bl, :n_fu] = dust_fu
    la, lb = superglue_marginals(n_bl, n_fu, device, dtype)
    P = log_sinkhorn(S, iters, la, lb).exp()
    out = np.full(n_bl, -1, dtype=np.int64)
    for i in range(n_bl):
        row = P[i] / P[i].sum().clamp_min(1e-9)
        j = int(row.argmax().item())
        if j == n_fu or float(row[j].item()) < tau:
            continue
        out[i] = j
    return out


def decode_sinkhorn_hungarian(
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    n_bl: int,
    n_fu: int,
    iters: int = 20,
    tau: float = 0.2,
) -> np.ndarray:
    device, dtype = pair_log.device, pair_log.dtype
    S = torch.zeros((n_bl + 1, n_fu + 1), device=device, dtype=dtype)
    S[:n_bl, :n_fu] = pair_log.reshape(n_bl, n_fu)
    S[:n_bl, n_fu] = dust_bl
    S[n_bl, :n_fu] = dust_fu
    la, lb = superglue_marginals(n_bl, n_fu, device, dtype)
    P = log_sinkhorn(S, iters, la, lb).exp()
    Prows = P[:n_bl].cpu().numpy()
    cost = -np.log(np.clip(Prows[:, : n_fu + 1], 1e-12, 1.0))
    ri, ci = linear_sum_assignment(cost)
    out = np.full(n_bl, -1, dtype=np.int64)
    rs = np.sum(Prows, axis=1)
    for r, c in zip(ri, ci):
        r = int(r)
        if int(c) >= n_fu:
            continue
        if float(Prows[r, int(c)] / max(rs[r], 1e-12)) >= tau:
            out[r] = int(c)
    return out
