"""Dense within-graph self + cross-attention after heterogeneous GNN.

UNUSED by default (ModelConfig.set_attn_blocks=0).

Disabled after Round 7 empirical run: stacking 2 SetAttn blocks on top of the
dense bipartite TransformerConv in HeteroGnn grew params 1.5M -> 2.6M and
regressed val_match_score 0.92 -> 0.78 and val_acc_unchanged_split 0.85 -> 0.70
(~224 train graphs).

Mechanism: HeteroGnn already attends within each scan plus across BL/FU with
the full 27-D cross_attr as TransformerConv edge_bias. These blocks attend on
embedding space only — extra capacity without new relational signal, i.e.
over-parameterization on tiny data.

May be worth revisiting after swapping the frozen L0 descriptor for a learned
encoder (Round 8+). Wire set_attn_blocks>0 only when that experiment is explicit.

Implementation: torch_geometric.utils.to_dense_batch (+ padding masks) feeding
torch.nn.MultiheadAttention; unchanged when disabled via set_attn_blocks=0.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch


class SetAttn(nn.Module):
    """Pre-LN stack: self(BL), self(FU), cross(BL<-FU), cross(FU<-BL), FFN per side."""

    def __init__(self, d: int, heads: int, dropout: float, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([_SetAttnBlock(d, heads, dropout) for _ in range(num_blocks)])

    def forward(self, z_bl: torch.Tensor, z_fu: torch.Tensor, batch_bl: torch.Tensor, batch_fu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for b in self.blocks:
            z_bl, z_fu = b(z_bl, z_fu, batch_bl, batch_fu)
        return z_bl, z_fu


class _SetAttnBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ln_sb = nn.LayerNorm(d)
        self.ln_sf = nn.LayerNorm(d)
        self.sa_bl = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.sa_fu = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_cb = nn.LayerNorm(d)
        self.ln_kf = nn.LayerNorm(d)
        self.ln_vf = nn.LayerNorm(d)
        self.x_bf = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_cf = nn.LayerNorm(d)
        self.ln_kb = nn.LayerNorm(d)
        self.ln_vb = nn.LayerNorm(d)
        self.x_fb = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_fbb = nn.LayerNorm(d)
        self.ln_fbf = nn.LayerNorm(d)
        self.ff_b = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.ff_f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(
        self, z_bl: torch.Tensor, z_fu: torch.Tensor, batch_bl: torch.Tensor, batch_fu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xb, mb = to_dense_batch(z_bl, batch_bl)
        xf, mf = to_dense_batch(z_fu, batch_fu)
        kpb, kpf = ~mb, ~mf
        hb = self.ln_sb(xb)
        sb, _ = self.sa_bl(hb, hb, hb, key_padding_mask=kpb, need_weights=False)
        xb = xb + sb
        hf = self.ln_sf(xf)
        sf, _ = self.sa_fu(hf, hf, hf, key_padding_mask=kpf, need_weights=False)
        xf = xf + sf
        qb, kf, vf = self.ln_cb(xb), self.ln_kf(xf), self.ln_vf(xf)
        cx, _ = self.x_bf(qb, kf, vf, key_padding_mask=kpf, need_weights=False)
        xb = xb + cx
        qf, kb, vb = self.ln_cf(xf), self.ln_kb(xb), self.ln_vb(xb)
        cx2, _ = self.x_fb(qf, kb, vb, key_padding_mask=kpb, need_weights=False)
        xf = xf + cx2
        xb = xb + self.ff_b(self.ln_fbb(xb))
        xf = xf + self.ff_f(self.ln_fbf(xf))
        return xb[mb], xf[mf]
