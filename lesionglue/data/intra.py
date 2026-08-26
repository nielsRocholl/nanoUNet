"""Intra-scan edges: kNN or complete; optional same-type / same-BL-image mask.

k<=0 is the complete directed graph (i≠j). Isolated nodes get a self-loop so
TransformerConv always sees a non-empty relation. Native BL millimetres from
different img_id_bl are not comparable — pass img to mask those pairs.
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from tracking.data.features import DESC_DIM, STAT_DIM, feat_layout
from tracking.data.graph import GraphConfig
from tracking.data.pairs import cross_attr, dense_pair_index, reverse_cross_attr

LT_IDX = DESC_DIM + STAT_DIM


def intra_edges(
    pos: torch.Tensor,
    k: int,
    types: torch.Tensor | None = None,
    img: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(pos.shape[0])
    dev = pos.device
    if n == 1:
        return torch.tensor([[0], [0]], dtype=torch.long, device=dev), torch.zeros((1, 1), device=dev)
    valid = ~torch.eye(n, dtype=torch.bool, device=dev)
    if types is not None:
        valid = valid & (types[:, None] == types[None, :])
    if img is not None:
        valid = valid & (img[:, None] == img[None, :])
    d_mat = torch.cdist(pos, pos).masked_fill(~valid, float("inf"))
    if k <= 0:
        ii, jj = valid.nonzero(as_tuple=True)
        dist = d_mat[ii, jj]
    else:
        ke = min(k, n - 1)
        dist, nei = d_mat.topk(ke, largest=False, dim=1)
        ii = torch.arange(n, device=dev).unsqueeze(1).expand_as(nei).reshape(-1)
        jj = nei.reshape(-1)
        dist = dist.reshape(-1)
        ok = torch.isfinite(dist)
        ii, jj, dist = ii[ok], jj[ok], dist[ok]
    iso = (~valid.any(dim=1)).nonzero(as_tuple=True)[0]
    if iso.numel():
        z = torch.zeros(iso.numel(), device=dev, dtype=d_mat.dtype)
        ii = torch.cat([ii, iso]) if ii.numel() else iso
        jj = torch.cat([jj, iso]) if jj.numel() else iso
        dist = torch.cat([dist, z]) if dist.numel() else z
    if ii.numel() == 0:
        idx = torch.arange(n, device=dev)
        return torch.stack([idx, idx]), torch.zeros((n, 1), device=dev, dtype=torch.float32)
    return torch.stack([ii, jj], dim=0), (dist.reshape(-1, 1) / 100.0).to(torch.float32)


def refresh_edges(data: HeteroData, cfg: GraphConfig) -> HeteroData:
    assert cfg.intra in ("knn", "complete"), cfg.intra
    use_native = cfg.drop_dp or cfg.intra == "complete"
    pos_bl = data["bl"].pos_native if use_native else data["bl"].pos
    pos_fu = data["fu"].pos
    types_bl = data["bl"].x[:, LT_IDX].long() if cfg.type_mask else None
    types_fu = data["fu"].x[:, LT_IDX].long() if cfg.type_mask else None
    img_bl = data["bl"].img_bl if use_native else None
    k = 0 if cfg.intra == "complete" else cfg.k_intra
    data["bl", "intra", "bl"].edge_index, data["bl", "intra", "bl"].edge_attr = intra_edges(
        pos_bl, k, types_bl, img_bl
    )
    data["fu", "intra", "fu"].edge_index, data["fu", "intra", "fu"].edge_attr = intra_edges(
        pos_fu, k, types_fu, None
    )
    n_bl, n_fu = int(data["bl"].num_nodes), int(data["fu"].num_nodes)
    ei = dense_pair_index(n_bl, n_fu, dev=pos_bl.device)
    ea = cross_attr(
        data["bl"].pos, data["fu"].pos, data["bl"].x, data["fu"].x, ei, feat_layout(), drop_dp=cfg.drop_dp
    )
    data["bl", "cross", "fu"].edge_index = ei
    data["bl", "cross", "fu"].edge_attr = ea
    data["fu", "cross", "bl"].edge_index = ei.flip(0)
    data["fu", "cross", "bl"].edge_attr = reverse_cross_attr(ea)
    return data
