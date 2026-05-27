"""Dense heterogeneous matcher: GNN pair logits + Sinkhorn dustbin."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, TransformerConv

from tracking.data.features import STAT_DIM
from tracking.data.pairs import CROSS_DIM
from tracking.set_attn import SetAttn


@dataclass
class ModelConfig:
    d: int = 128
    layers: int = 4
    heads: int = 4
    lt_vocab: int = 12
    lt_embed: int = 8
    dropout: float = 0.2
    desc_dim: int = 1372
    use_dust_pair_summary: bool = True
    dust_legacy_linear: bool = False
    set_attn_blocks: int = 0


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
        self.desc_dim = cfg.desc_dim
        self.stat_off = cfg.desc_dim
        self.lt_idx = cfg.desc_dim + STAT_DIM
        self.emb = nn.Embedding(cfg.lt_vocab, cfg.lt_embed)
        inn = cfg.desc_dim + STAT_DIM + cfg.lt_embed
        self.net = nn.Sequential(nn.Linear(inn, 256), nn.ReLU(inplace=True), nn.Linear(256, cfg.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = x[:, : self.desc_dim]
        st = x[:, self.stat_off : self.lt_idx]
        ti = x[:, self.lt_idx].long()
        return self.net(torch.cat([desc, st, self.emb(ti)], dim=1))


class DustHead(nn.Module):
    def __init__(self, d: int, drop: float = 0.5):
        super().__init__()
        h = max(8, d // 2)
        self.net = nn.Sequential(
            nn.Linear(d + 3, h),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(h, 1),
        )

    def forward(self, z: torch.Tensor, pair_summary: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, pair_summary], dim=1)).squeeze(-1)


def _row_summaries(M: torch.Tensor) -> torch.Tensor:
    maxv = M.max(dim=1).values
    meanv = M.mean(dim=1)
    if M.size(1) >= 2:
        t = M.topk(2, dim=1).values
        gap = t[:, 0] - t[:, 1]
    else:
        gap = torch.zeros_like(maxv)
    return torch.stack([maxv, meanv, gap], dim=1)


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
        self.set_attn = (
            SetAttn(cfg.d, cfg.heads, cfg.dropout, cfg.set_attn_blocks) if cfg.set_attn_blocks > 0 else None
        )
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.d + CROSS_DIM, cfg.d), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout), nn.Linear(cfg.d, 1)
        )
        if cfg.dust_legacy_linear:
            self.dust_mlp: DustHead | None = None
            self.dust_lin = nn.Linear(cfg.d, 1)
        else:
            self.dust_lin = None
            self.dust_mlp = DustHead(cfg.d, drop=0.5)

    def _dust_from_pair(self, pair: torch.Tensor, data: HeteroData, z_bl: torch.Tensor, z_fu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.dust_legacy_linear:
            assert self.dust_lin is not None
            return self.dust_lin(z_bl).squeeze(-1), self.dust_lin(z_fu).squeeze(-1)
        assert self.dust_mlp is not None
        dev, dt = pair.device, pair.dtype
        use_sum = self.cfg.use_dust_pair_summary
        if hasattr(data["bl"], "batch") and data["bl"].batch is not None:
            ng = int(data.num_graphs)
            nb = torch.bincount(data["bl"].batch, minlength=ng)
            nf = torch.bincount(data["fu"].batch, minlength=ng)
            es = (nb * nf).tolist()
            parts = torch.split(pair, es)
            zbs = torch.split(z_bl, nb.tolist())
            zfs = torch.split(z_fu, nf.tolist())
            dbl, dfu = [], []
            for p, zb, zf, nbg, nfg in zip(parts, zbs, zfs, nb.tolist(), nf.tolist()):
                M = p.reshape(nbg, nfg)
                sbl = _row_summaries(M) if use_sum else torch.zeros((nbg, 3), device=dev, dtype=dt)
                sfu = _row_summaries(M.T) if use_sum else torch.zeros((nfg, 3), device=dev, dtype=dt)
                dbl.append(self.dust_mlp(zb, sbl))
                dfu.append(self.dust_mlp(zf, sfu))
            return torch.cat(dbl, dim=0), torch.cat(dfu, dim=0)
        n_bl, n_fu = int(data["bl"].num_nodes), int(data["fu"].num_nodes)
        M = pair.reshape(n_bl, n_fu)
        sbl = _row_summaries(M) if use_sum else torch.zeros((n_bl, 3), device=dev, dtype=dt)
        sfu = _row_summaries(M.T) if use_sum else torch.zeros((n_fu, 3), device=dev, dtype=dt)
        return self.dust_mlp(z_bl, sbl), self.dust_mlp(z_fu, sfu)

    def forward(self, data: HeteroData) -> MatcherOutput:
        z = {"bl": self.enc(data["bl"].x), "fu": self.enc(data["fu"].x)}
        z = self.gnn(z, data.edge_index_dict, data.edge_attr_dict)
        if self.set_attn is not None:
            bbl = data["bl"].batch
            bfu = data["fu"].batch
            dev = z["bl"].device
            if bbl is None:
                bbl = torch.zeros(z["bl"].size(0), dtype=torch.long, device=dev)
            if bfu is None:
                bfu = torch.zeros(z["fu"].size(0), dtype=torch.long, device=dev)
            z["bl"], z["fu"] = self.set_attn(z["bl"], z["fu"], bbl, bfu)
        ei = data["bl", "cross", "fu"].edge_index
        ea = data["bl", "cross", "fu"].edge_attr
        h = torch.cat([z["bl"][ei[0]], z["fu"][ei[1]], ea], dim=1)
        pair = self.head(h).squeeze(-1)
        dust_bl, dust_fu = self._dust_from_pair(pair, data, z["bl"], z["fu"])
        return MatcherOutput(pair, dust_bl, dust_fu, z["bl"], z["fu"])
