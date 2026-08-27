"""Sinkhorn decoding with SuperGlue marginals, dense threshold, optional Hungarian."""

from __future__ import annotations

import sys

import numpy as np
import torch
from rich.prompt import Prompt
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from tracking.common import DEPLOYED_DUST_TAU, cprint
from tracking.train.sinkhorn import log_sinkhorn, superglue_marginals

DECODE_CHOICES = ("dense", "sinkhorn", "hungarian")
DECODE_HELP = (
    "how to turn pair logits into matches: dense (keep all pairs above --thresh; merges and splits stay), "
    "sinkhorn (each baseline picks at most one follow-up; merges stay, splits drop), "
    "hungarian (strict 1-to-1; merges and splits drop). Omit to choose interactively."
)


def decode_sinkhorn(
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    n_bl: int,
    n_fu: int,
    iters: int = 20,
    tau: float = DEPLOYED_DUST_TAU,
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
    tau: float = DEPLOYED_DUST_TAU,
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


def decode_dense(pair_log: torch.Tensor, n_bl: int, n_fu: int, thresh: float = 0.5) -> np.ndarray:
    p = torch.sigmoid(pair_log).reshape(n_bl, n_fu)
    ii, jj = (p >= thresh).nonzero(as_tuple=True)
    if ii.numel() == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return torch.stack([ii, jj], dim=1).cpu().numpy().astype(np.int64)


def decode_pairs(
    method: str,
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    n_bl: int,
    n_fu: int,
    *,
    thresh: float,
    sinkhorn_iters: int,
    sinkhorn_tau: float,
) -> np.ndarray:
    assert method in DECODE_CHOICES
    if method == "dense":
        return decode_dense(pair_log, n_bl, n_fu, thresh)
    fn = decode_sinkhorn if method == "sinkhorn" else decode_sinkhorn_hungarian
    asg = fn(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters=sinkhorn_iters, tau=sinkhorn_tau)
    live = [(i, int(j)) for i, j in enumerate(asg) if int(j) >= 0]
    return np.asarray(live, dtype=np.int64).reshape(-1, 2)


def resolve_decode(cli_value: str | None) -> str:
    if cli_value is not None:
        return cli_value
    if not sys.stdin.isatty():
        raise SystemExit(
            "No --decode given and stdin is not a TTY.\n"
            "Expected one of: dense, sinkhorn, hungarian.\n"
            "Fix: lesion_track ... --decode dense"
        )
    t = Table(title="How should matches be decoded?", box=None, padding=(0, 2))
    t.add_column("choice", style="cyan")
    t.add_column("keeps")
    t.add_column("drops")
    t.add_row("dense", "every pair above threshold; one lesion can match many", "nothing (native model output)")
    t.add_row("sinkhorn", "each baseline → at most one follow-up; merges stay", "splits (one BL → many FU)")
    t.add_row("hungarian", "strict 1-to-1 list", "merges and splits")
    cprint(t)
    cprint("[dim]dense = what the network outputs. hungarian / sinkhorn = optional post-process.[/dim]")
    return Prompt.ask("decode", choices=["dense", "sinkhorn", "hungarian"])

