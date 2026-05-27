"""Focal BCE, batched Sinkhorn split, same-anatomy InfoNCE, Hungarian match accuracy."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, HeteroData

from tracking.decode import decode_sinkhorn_hungarian
from tracking.matcher import MatcherOutput


def focal_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt).clamp_min(1e-6).pow(gamma)
    return (w * bce).mean()


def split_per_graph(
    batch: Batch, out: MatcherOutput
) -> tuple[list[HeteroData], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    graphs = batch.to_data_list()
    es = [g["bl", "cross", "fu"].num_edges for g in graphs]
    nb = [g["bl"].num_nodes for g in graphs]
    nf = [g["fu"].num_nodes for g in graphs]
    pp, db, df = torch.split(out.pair, es), torch.split(out.dust_bl, nb), torch.split(out.dust_fu, nf)
    return graphs, list(pp), list(db), list(df)


def infonce_batch(
    z_bl: torch.Tensor,
    z_fu: torch.Tensor,
    proj: nn.Module,
    edge_index: torch.Tensor,
    edge_label: torch.Tensor,
    tau: float,
    batch: Batch,
) -> torch.Tensor:
    pb = F.normalize(proj(z_bl), dim=1)
    pf = F.normalize(proj(z_fu), dim=1)
    sim = (pb @ pf.T) / tau
    pos = edge_label > 0.5
    if not pos.any():
        return z_bl.sum() * 0.0
    lt_idx = batch["bl"].x.shape[1] - 1
    lt_bl = batch["bl"].x[:, lt_idx].long()
    lt_fu = batch["fu"].x[:, lt_idx].long()
    same = lt_bl[:, None] == lt_fu[None, :]
    bi, fj = edge_index[0, pos], edge_index[1, pos]
    sort_idx = torch.argsort(bi)
    sb, sf = bi[sort_idx], fj[sort_idx]
    mask = torch.ones(sb.shape[0], dtype=torch.bool, device=sb.device)
    mask[1:] = sb[1:] != sb[:-1]
    bi_u, fj_u = sb[mask], sf[mask]
    sim_a = sim.clone()
    for b in bi_u:
        if same[b].sum() > 1:
            sim_a[b] = sim_a[b].masked_fill(~same[b], float("-inf"))
    la = F.cross_entropy(sim_a[bi_u], fj_u)
    sort_idx = torch.argsort(fj)
    sb, si = fj[sort_idx], bi[sort_idx]
    mask = torch.ones(sb.shape[0], dtype=torch.bool, device=sb.device)
    mask[1:] = sb[1:] != sb[:-1]
    fj_u2, bi_u2 = sb[mask], si[mask]
    sim_t = sim_a.T.clone()
    for f in torch.unique(fj_u2):
        if same[:, f].sum() > 1:
            sim_t[f] = sim_t[f].masked_fill(~same[:, f], float("-inf"))
    lb = F.cross_entropy(sim_t[fj_u2], bi_u2)
    return 0.5 * (la + lb)


def row_hungarian_match_acc(
    data: HeteroData,
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    iters: int,
    tau: float,
) -> float:
    n_bl, n_fu = data["bl"].num_nodes, data["fu"].num_nodes
    lab = data["bl", "cross", "fu"].edge_label.reshape(n_bl, n_fu)
    dec = decode_sinkhorn_hungarian(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters=iters, tau=tau)
    ok = 0
    for i in range(n_bl):
        pos = torch.where(lab[i] > 0.5)[0]
        di = int(dec[i])
        ok += int(di < 0) if pos.numel() == 0 else int((pos == di).any().item())
    return ok / max(n_bl, 1)
