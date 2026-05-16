"""Dense heterogeneous matcher: GNN pair logits + learnable Sinkhorn dustbin scalars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, TransformerConv

from tracking.data.pairs import CROSS_DIM
from tracking.train.sinkhorn import log_sinkhorn, superglue_marginals


@dataclass
class ModelConfig:
    d: int = 128
    layers: int = 4
    heads: int = 4
    lt_vocab: int = 12
    lt_embed: int = 8
    dropout: float = 0.2
    use_dust_pair_summary: bool = True
    dust_legacy_linear: bool = False


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
        self.emb = nn.Embedding(cfg.lt_vocab, cfg.lt_embed)
        inn = 1372 + 14 + cfg.lt_embed
        self.net = nn.Sequential(nn.Linear(inn, 256), nn.ReLU(inplace=True), nn.Linear(256, cfg.d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = x[:, :1372]
        st = x[:, 1372:1386]
        ti = x[:, 1386].long()
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
        ei = data["bl", "cross", "fu"].edge_index
        ea = data["bl", "cross", "fu"].edge_attr
        h = torch.cat([z["bl"][ei[0]], z["fu"][ei[1]], ea], dim=1)
        pair = self.head(h).squeeze(-1)
        dust_bl, dust_fu = self._dust_from_pair(pair, data, z["bl"], z["fu"])
        return MatcherOutput(pair, dust_bl, dust_fu, z["bl"], z["fu"])


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
