"""Focal BCE, graph splits, graph-scope InfoNCE, validation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, HeteroData

from lesionglue.decode import decode_sinkhorn_hungarian
from lesionglue.matcher import MatcherOutput
from lesionglue.train.sinkhorn import log_sinkhorn, superglue_marginals


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt).clamp_min(1e-6).pow(gamma)
    return (w * bce).mean()


def split_per_graph(batch: Batch, out: MatcherOutput) -> tuple[list[HeteroData], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    graphs = batch.to_data_list()
    es = [g["bl", "cross", "fu"].num_edges for g in graphs]
    nb = [g["bl"].num_nodes for g in graphs]
    nf = [g["fu"].num_nodes for g in graphs]
    pp, db, df = torch.split(out.pair, es), torch.split(out.dust_bl, nb), torch.split(out.dust_fu, nf)
    return graphs, list(pp), list(db), list(df)


def _infonce(
    z_bl: torch.Tensor,
    z_fu: torch.Tensor,
    x_bl: torch.Tensor,
    x_fu: torch.Tensor,
    proj: nn.Module,
    edge_index: torch.Tensor,
    edge_label: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    pos = edge_label > 0.5
    if not pos.any():
        return z_bl.sum() * 0.0
    sim = (F.normalize(proj(z_bl), dim=1) @ F.normalize(proj(z_fu), dim=1).T) / tau
    lt_bl, lt_fu = x_bl[:, -1].long(), x_fu[:, -1].long()
    same = lt_bl[:, None] == lt_fu[None, :]
    bi, fj = edge_index[0, pos], edge_index[1, pos]
    order = torch.argsort(bi)
    sb, sf = bi[order], fj[order]
    keep = torch.ones(sb.shape[0], dtype=torch.bool, device=sb.device)
    keep[1:] = sb[1:] != sb[:-1]
    bi_u, fj_u = sb[keep], sf[keep]
    sim_a = sim.clone()
    for b in bi_u:
        if same[b].sum() > 1:
            sim_a[b] = sim_a[b].masked_fill(~same[b], float("-inf"))
    la = F.cross_entropy(sim_a[bi_u], fj_u)
    order = torch.argsort(fj)
    sf, sb = fj[order], bi[order]
    keep = torch.ones(sf.shape[0], dtype=torch.bool, device=sf.device)
    keep[1:] = sf[1:] != sf[:-1]
    fj_u, bi_u = sf[keep], sb[keep]
    sim_t = sim_a.T.clone()
    for f in torch.unique(fj_u):
        if same[:, f].sum() > 1:
            sim_t[f] = sim_t[f].masked_fill(~same[:, f], float("-inf"))
    return 0.5 * (la + F.cross_entropy(sim_t[fj_u], bi_u))


def infonce_graphs(graphs: list[HeteroData], z_bl: torch.Tensor, z_fu: torch.Tensor, proj: nn.Module, tau: float) -> torch.Tensor:
    nb = [g["bl"].num_nodes for g in graphs]
    nf = [g["fu"].num_nodes for g in graphs]
    zbs, zfs = torch.split(z_bl, nb), torch.split(z_fu, nf)
    losses = [
        _infonce(zb, zf, g["bl"].x.to(zb.device), g["fu"].x.to(zf.device), proj, g["bl", "cross", "fu"].edge_index.to(zb.device), g["bl", "cross", "fu"].edge_label.to(zb.device), tau)
        for g, zb, zf in zip(graphs, zbs, zfs)
    ]
    return torch.stack(losses).mean() if losses else z_bl.sum() * 0.0


def sinkhorn_edge_scores(data: HeteroData, pair_log: torch.Tensor, dust_bl: torch.Tensor, dust_fu: torch.Tensor, iters: int) -> torch.Tensor:
    n_bl, n_fu = data["bl"].num_nodes, data["fu"].num_nodes
    dev, dt = pair_log.device, pair_log.dtype
    S = torch.zeros((n_bl + 1, n_fu + 1), device=dev, dtype=dt)
    S[:n_bl, :n_fu] = pair_log.reshape(n_bl, n_fu)
    S[:n_bl, n_fu] = dust_bl.detach()
    S[n_bl, :n_fu] = dust_fu.detach()
    P = log_sinkhorn(S, iters, *superglue_marginals(n_bl, n_fu, dev, dt)).exp()
    Rn = P[:n_bl] / P[:n_bl].sum(dim=1, keepdim=True).clamp_min(1e-9)
    ei = data["bl", "cross", "fu"].edge_index
    return Rn[ei[0], ei[1]].detach()


def graph_val_counts(data: HeteroData, pair_log: torch.Tensor, dust_bl: torch.Tensor, dust_fu: torch.Tensor, iters: int, tau: float) -> tuple[float, int, int, int, int, int, int]:
    n_bl, n_fu = data["bl"].num_nodes, data["fu"].num_nodes
    lab = data["bl", "cross", "fu"].edge_label.reshape(n_bl, n_fu)
    dec = decode_sinkhorn_hungarian(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters=iters, tau=tau)
    ok = uc_ok = uc_tot = dis_ok = dis_tot = new_ok = new_tot = 0
    for i in range(n_bl):
        pos = torch.where(lab[i] > 0.5)[0]
        di = int(dec[i])
        ok += int(di < 0) if pos.numel() == 0 else int((pos == di).any().item())
        if pos.numel():
            uc_tot += 1
            uc_ok += int((pos == di).any().item())
        elif float(data["bl"].no_match_label[i]) > 0.5:
            dis_tot += 1
            dis_ok += int(di < 0)
    claimed = {int(dec[i]) for i in range(n_bl) if int(dec[i]) >= 0}
    for j in range(n_fu):
        if float(data["fu"].no_match_label[j]) > 0.5:
            new_tot += 1
            new_ok += int(j not in claimed)
    return ok / max(n_bl, 1), uc_ok, uc_tot, dis_ok, dis_tot, new_ok, new_tot
