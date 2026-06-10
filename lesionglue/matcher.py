"""Dense heterogeneous matcher: GNN pair logits + RowMatchability dustbin + bilinear identity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, TransformerConv

from tracking.data.features import DESC_DIM, STAT_DIM
from tracking.data.pairs import CROSS_DIM
from tracking.matchability import RowMatchability, row_dust_marginals


@dataclass
class ModelConfig:
    d: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.2
    lt_vocab: int = 12
    lt_embed: int = 8
    sinkhorn_iters: int = 20


@dataclass
class MatcherOutput:
    pair: torch.Tensor
    dust_bl: torch.Tensor
    dust_fu: torch.Tensor
    z_bl: torch.Tensor
    z_fu: torch.Tensor


class NodeEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.desc_dim = DESC_DIM
        self.stat_off = DESC_DIM
        self.lt_idx = DESC_DIM + STAT_DIM
        self.emb = nn.Embedding(cfg.lt_vocab, cfg.lt_embed)
        inn = DESC_DIM + STAT_DIM + cfg.lt_embed
        self.net = nn.Sequential(nn.Linear(inn, 256), nn.ReLU(inplace=True), nn.Linear(256, cfg.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = x[:, : self.desc_dim]
        st = x[:, self.stat_off : self.lt_idx]
        ti = x[:, self.lt_idx].long()
        return self.net(torch.cat([desc, st, self.emb(ti)], dim=1))


class HeteroGnn(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                HeteroConv(
                    {
                        ("bl", "intra", "bl"): TransformerConv(d, d // heads, heads=heads, edge_dim=1),
                        ("fu", "intra", "fu"): TransformerConv(d, d // heads, heads=heads, edge_dim=1),
                        ("bl", "cross", "fu"): TransformerConv((d, d), d // heads, heads=heads, edge_dim=CROSS_DIM),
                        ("fu", "cross", "bl"): TransformerConv((d, d), d // heads, heads=heads, edge_dim=CROSS_DIM),
                    },
                    aggr="sum",
                )
            )
            self.norms.append(nn.ModuleDict({"bl": nn.LayerNorm(d), "fu": nn.LayerNorm(d)}))

    def forward(self, x_dict: dict, edge_index_dict: dict, edge_attr_dict: dict) -> dict:
        for conv, norm in zip(self.layers, self.norms):
            h = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {k: self.drop(norm[k](F.relu(x_dict[k] + h[k]))) for k in x_dict}
        return x_dict


class Matcher(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = NodeEncoder(cfg)
        self.gnn = HeteroGnn(cfg.d, cfg.layers, cfg.heads, cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.d + CROSS_DIM, cfg.d), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout), nn.Linear(cfg.d, 1)
        )
        self.bilin = nn.Bilinear(cfg.d, cfg.d, 1, bias=False)
        self.dust_enc = RowMatchability(cfg.d)

    def _dust_graph(
        self, pair: torch.Tensor, data: HeteroData, z_bl: torch.Tensor, z_fu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z0 = torch.zeros(1, device=pair.device, dtype=pair.dtype)
        iters = self.cfg.sinkhorn_iters
        if hasattr(data["bl"], "batch") and data["bl"].batch is not None:
            ng = int(data.num_graphs)
            nb = torch.bincount(data["bl"].batch, minlength=ng)
            nf = torch.bincount(data["fu"].batch, minlength=ng)
            parts = torch.split(pair, (nb * nf).tolist())
            zbs, zfs = torch.split(z_bl, nb.tolist()), torch.split(z_fu, nf.tolist())
            dbl, dfu = [], []
            for p, zb, zf, nbg, nfg in zip(parts, zbs, zfs, nb.tolist(), nf.tolist()):
                M = p.reshape(nbg, nfg)
                mb, mf = row_dust_marginals(p, z0.expand(nbg), z0.expand(nfg), nbg, nfg, iters)
                dbl.append(self.dust_enc(zb, M, mb, zf))
                dfu.append(self.dust_enc(zf, M.T, mf, zb))
            return torch.cat(dbl, dim=0), torch.cat(dfu, dim=0)
        n_bl, n_fu = int(data["bl"].num_nodes), int(data["fu"].num_nodes)
        M = pair.reshape(n_bl, n_fu)
        mb, mf = row_dust_marginals(pair, z0.expand(n_bl), z0.expand(n_fu), n_bl, n_fu, iters)
        return self.dust_enc(z_bl, M, mb, z_fu), self.dust_enc(z_fu, M.T, mf, z_bl)

    def forward(self, data: HeteroData) -> MatcherOutput:
        z = {"bl": self.enc(data["bl"].x), "fu": self.enc(data["fu"].x)}
        z = self.gnn(z, data.edge_index_dict, data.edge_attr_dict)
        ei = data["bl", "cross", "fu"].edge_index
        ea = data["bl", "cross", "fu"].edge_attr
        h = torch.cat([z["bl"][ei[0]], z["fu"][ei[1]], ea], dim=1)
        pair = self.head(h).squeeze(-1) + self.bilin(z["bl"][ei[0]], z["fu"][ei[1]]).squeeze(-1)
        dust_bl, dust_fu = self._dust_graph(pair.detach(), data, z["bl"], z["fu"])
        return MatcherOutput(pair, dust_bl, dust_fu, z["bl"], z["fu"])
