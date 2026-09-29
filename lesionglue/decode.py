"""Turn matcher outputs into BL->FU pairs: dense threshold, Sinkhorn threshold, or Sinkhorn + Hungarian.

Why the Sinkhorn decoder thresholds instead of taking a per-row argmax
---------------------------------------------------------------------
The transport plan P comes from `superglue_marginals`: every real BL row and every real
FU column carries one unit of mass; only the dustbins carry more. A merge of k BL lesions
into one FU lesion therefore cannot route k units into that column. Sinkhorn gives each
contributor about 1/k of its row mass on the merged column and the rest, (k-1)/k, on the
BL dustbin. A split is the mirror image: one row spreads about 1/k over each of k columns.

A per-row argmax therefore loses merges once the dustbin share wins (already at k=2,
where both shares sit at 0.5 and the dustbin edges ahead), and a Hungarian solve keeps
at most one contributor.
`decode_sinkhorn` reads the plan symmetrically instead: keep every real (i, j) whose
row-normalised mass clears `tau`. Merges and splits fall out of the same rule, and a
row whose mass sits on the dustbin emits nothing (disappeared).

Limit: because a contributor holds at most 1/k, a k-way merge is only recoverable while
1/k >= tau, and 1/k == tau sits on the numerical edge (k=8 at tau=0.125 reads 0.1249).
Larger merges need relaxed column marginals (unbalanced OT or column capacity), in
training and decode alike; that is not done here.

Labels note: `tracking/data/graph.py::_positive_matrix` encodes SPLIT as a single
(lesion_id, lesion_id) edge, so the current labels give every BL row at most one positive
and only merges are many-to-one there. The threshold rule still emits one-to-many links
if the model produces them, which are then scored as wrong rows by `tracking/report.py`.
`sinkhorn_loss` also targets only the first positive column per row; it would need to
average over all positives (as its FU side already does) before splits are labelled that way.
`tau` defaults to DEPLOYED_DUST_TAU, which was tuned for Hungarian, not for this rule.
"""

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
    "sinkhorn (keep every pair whose Sinkhorn row mass clears --sinkhorn-tau; merges and splits stay "
    "up to about 1/tau lesions), hungarian (strict 1-to-1; merges and splits drop). Omit to choose interactively."
)


def _transport_plan(
    pair_log: torch.Tensor, dust_bl: torch.Tensor, dust_fu: torch.Tensor, n_bl: int, n_fu: int, iters: int
) -> torch.Tensor:
    """(n_bl+1, n_fu+1) Sinkhorn plan with SuperGlue dustbin marginals; last row/col are dustbins."""
    device, dtype = pair_log.device, pair_log.dtype
    S = torch.zeros((n_bl + 1, n_fu + 1), device=device, dtype=dtype)
    S[:n_bl, :n_fu] = pair_log.reshape(n_bl, n_fu)
    S[:n_bl, n_fu] = dust_bl
    S[n_bl, :n_fu] = dust_fu
    la, lb = superglue_marginals(n_bl, n_fu, device, dtype)
    return log_sinkhorn(S, iters, la, lb).exp()


def decode_sinkhorn(
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    n_bl: int,
    n_fu: int,
    iters: int = 20,
    tau: float = DEPLOYED_DUST_TAU,
) -> np.ndarray:
    """(m, 2) int64 (bl_row, fu_col) pairs, sorted: every real cell with row-normalised mass >= tau."""
    P = _transport_plan(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters)
    R = P[:n_bl, :n_fu] / P[:n_bl].sum(dim=1, keepdim=True).clamp_min(1e-9)
    return (R >= tau).nonzero().cpu().numpy().astype(np.int64).reshape(-1, 2)


def decode_sinkhorn_hungarian(
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    n_bl: int,
    n_fu: int,
    iters: int = 20,
    tau: float = DEPLOYED_DUST_TAU,
) -> np.ndarray:
    """(n_bl,) int64: one FU column per BL row, -1 for dustbin. Strictly 1-to-1."""
    P = _transport_plan(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters)
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
    if method == "sinkhorn":
        return decode_sinkhorn(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters=sinkhorn_iters, tau=sinkhorn_tau)
    asg = decode_sinkhorn_hungarian(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters=sinkhorn_iters, tau=sinkhorn_tau)
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
    t.add_row("sinkhorn", "every pair with Sinkhorn row mass ≥ tau; merges and splits stay", "merges/splits of more than ~1/tau lesions")
    t.add_row("hungarian", "strict 1-to-1 list", "merges and splits")
    cprint(t)
    cprint("[dim]dense = what the network outputs. hungarian / sinkhorn = optional post-process.[/dim]")
    return Prompt.ask("decode", choices=["dense", "sinkhorn", "hungarian"])

