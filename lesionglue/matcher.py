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
from tracking.matchability import RowMatchability, row_dust_marginals
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
    desc_norm: bool = False
    edge_cross_attn: bool = False
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
        self.desc_dim = cfg.desc_dim
        self.stat_off = cfg.desc_dim
        self.lt_idx = cfg.desc_dim + STAT_DIM
        self.desc_norm = nn.LayerNorm(cfg.desc_dim, elementwise_affine=False) if cfg.desc_norm else nn.Identity()
        self.emb = nn.Embedding(cfg.lt_vocab, cfg.lt_embed)
        inn = cfg.desc_dim + STAT_DIM + cfg.lt_embed
        self.net = nn.Sequential(nn.Linear(inn, 256), nn.ReLU(inplace=True), nn.Linear(256, cfg.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = self.desc_norm(x[:, : self.desc_dim])
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


class _EdgeCrossAttn(nn.Module):
    def __init__(self, d: int, heads: int, cross_dim: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.bias = nn.Linear(cross_dim, heads)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.heads = heads
        self.hd = d // heads

    def forward(self, z_bl: torch.Tensor, z_fu: torch.Tensor, ei: torch.Tensor, ea: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i, j = ei
        qb, kf, vf = self.q(z_bl[i]), self.k(z_fu[j]), self.v(z_fu[j])
        bh = self.bias(ea).unsqueeze(-1)
        sc = (qb * kf).view(-1, self.heads, self.hd).sum(-1) / (self.hd**0.5) + bh.squeeze(-1)
        w = torch.softmax(sc, dim=0)
        msg = (w.unsqueeze(-1) * vf.view(-1, self.heads, self.hd)).view(-1, self.heads * self.hd)
        zb = z_bl.clone()
        zf = z_fu.clone()
        zb.index_add_(0, i, self.drop(self.out(msg)))
        zf.index_add_(0, j, self.drop(self.out(msg)))
        return zb, zf


class Matcher(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = NodeEncoder(cfg)
        self.gnn = HeteroGnn(cfg.d, cfg.layers, cfg.heads, cfg.dropout)
        self.set_attn = SetAttn(cfg.d, cfg.heads, cfg.dropout, cfg.set_attn_blocks) if cfg.set_attn_blocks > 0 else None
        self.edge_xattn = _EdgeCrossAttn(cfg.d, cfg.heads, CROSS_DIM, cfg.dropout) if cfg.edge_cross_attn else None
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.d + CROSS_DIM, cfg.d), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout), nn.Linear(cfg.d, 1)
        )
        self.bilin = nn.Bilinear(cfg.d, cfg.d, 1, bias=False)
        if cfg.dust_legacy_linear:
            self.dust_enc: RowMatchability | None = None
            self.dust_lin = nn.Linear(cfg.d, 1)
        else:
            self.dust_lin = None
            self.dust_enc = RowMatchability(cfg.d)

    def _dust_graph(
        self, pair: torch.Tensor, data: HeteroData, z_bl: torch.Tensor, z_fu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.dust_legacy_linear:
            assert self.dust_lin is not None
            return self.dust_lin(z_bl).squeeze(-1), self.dust_lin(z_fu).squeeze(-1)
        assert self.dust_enc is not None
        use = self.cfg.use_dust_pair_summary
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
                if use:
                    mb, mf = row_dust_marginals(p, z0.expand(nbg), z0.expand(nfg), nbg, nfg, iters)
                    dbl.append(self.dust_enc(zb, M, mb, zf))
                    dfu.append(self.dust_enc(zf, M.T, mf, zb))
                else:
                    zr = torch.zeros(nbg, device=M.device, dtype=M.dtype)
                    zc = torch.zeros(nfg, device=M.device, dtype=M.dtype)
                    dbl.append(self.dust_enc(zb, M * 0, zr, zf))
                    dfu.append(self.dust_enc(zf, M.T * 0, zc, zb))
            return torch.cat(dbl, dim=0), torch.cat(dfu, dim=0)
        n_bl, n_fu = int(data["bl"].num_nodes), int(data["fu"].num_nodes)
        M = pair.reshape(n_bl, n_fu)
        if use:
            mb, mf = row_dust_marginals(pair, z0.expand(n_bl), z0.expand(n_fu), n_bl, n_fu, iters)
            return self.dust_enc(z_bl, M, mb, z_fu), self.dust_enc(z_fu, M.T, mf, z_bl)
        zr = torch.zeros(n_bl, device=M.device, dtype=M.dtype)
        zc = torch.zeros(n_fu, device=M.device, dtype=M.dtype)
        return self.dust_enc(z_bl, M * 0, zr, z_fu), self.dust_enc(z_fu, M.T * 0, zc, z_bl)

    def forward(self, data: HeteroData) -> MatcherOutput:
        z = {"bl": self.enc(data["bl"].x), "fu": self.enc(data["fu"].x)}
        z = self.gnn(z, data.edge_index_dict, data.edge_attr_dict)
        if self.set_attn is not None:
            bbl, bfu = data["bl"].batch, data["fu"].batch
            dev = z["bl"].device
            if bbl is None:
                bbl = torch.zeros(z["bl"].size(0), dtype=torch.long, device=dev)
            if bfu is None:
                bfu = torch.zeros(z["fu"].size(0), dtype=torch.long, device=dev)
            z["bl"], z["fu"] = self.set_attn(z["bl"], z["fu"], bbl, bfu)
        ei = data["bl", "cross", "fu"].edge_index
        ea = data["bl", "cross", "fu"].edge_attr
        if self.edge_xattn is not None:
            z["bl"], z["fu"] = self.edge_xattn(z["bl"], z["fu"], ei, ea)
        h = torch.cat([z["bl"][ei[0]], z["fu"][ei[1]], ea], dim=1)
        pair = self.head(h).squeeze(-1) + self.bilin(z["bl"][ei[0]], z["fu"][ei[1]]).squeeze(-1)
        dust_bl, dust_fu = self._dust_graph(pair.detach(), data, z["bl"], z["fu"])
        return MatcherOutput(pair, dust_bl, dust_fu, z["bl"], z["fu"])
