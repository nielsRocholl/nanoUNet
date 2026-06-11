"""Assignment-anchored cross-attention refinement of the matching subspace.

One gated block per call. Projects GNN nodes into a separate match space, then
lets each BL node attend over its FU candidates (and symmetric) with attention
biased by (a) the current soft Sinkhorn assignment log P -- the anchoring signal
the bipartite GNN structurally lacks -- and (b) a learned projection of the 27-D
cross_attr. The residual is zero-initialized (g=0 => exact R9 baseline) so on 240
patients the block can only help: it starts as identity and earns its weight.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import scatter, softmax

from tracking.train.sinkhorn import log_sinkhorn, superglue_marginals


def assignment_logp(pair: torch.Tensor, n_bl: int, n_fu: int, iters: int) -> torch.Tensor:
    S = torch.zeros((n_bl + 1, n_fu + 1), device=pair.device, dtype=pair.dtype)
    S[:n_bl, :n_fu] = pair.reshape(n_bl, n_fu).detach()
    la, lb = superglue_marginals(n_bl, n_fu, pair.device, pair.dtype)
    P = log_sinkhorn(S, iters, la, lb)
    return P[:n_bl, :n_fu].reshape(-1).detach()


class MatchRefine(nn.Module):
    def __init__(self, d: int, heads: int, cross_dim: int, drop: float = 0.3):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.proj = nn.Linear(d, d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.edge_bias = nn.Linear(cross_dim, heads)
        self.assign_w = nn.Parameter(torch.zeros(heads))
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.g_bl = nn.Parameter(torch.zeros(1))
        self.g_fu = nn.Parameter(torch.zeros(1))
        self.score = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def _attend(
        self,
        q_nodes: torch.Tensor,
        kv_nodes: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        e_bias: torch.Tensor,
        logp: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q(q_nodes)[src].view(-1, self.heads, self.dh)
        k = self.k(kv_nodes)[dst].view(-1, self.heads, self.dh)
        v = self.v(kv_nodes)[dst].view(-1, self.heads, self.dh)
        att = (q * k).sum(-1) / math.sqrt(self.dh) + e_bias + self.assign_w * logp.unsqueeze(-1)
        a = softmax(att, src)
        msg = a.unsqueeze(-1) * v
        n_q = q_nodes.shape[0]
        agg = scatter(msg, src, dim=0, dim_size=n_q, reduce="sum").view(n_q, -1)
        return agg

    def forward(
        self,
        z_bl: torch.Tensor,
        z_fu: torch.Tensor,
        edge_index: torch.Tensor,
        cross_attr: torch.Tensor,
        logp: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        m_bl, m_fu = self.proj(z_bl), self.proj(z_fu)
        i, j = edge_index[0], edge_index[1]
        e_bias = self.edge_bias(cross_attr)
        a_bl = self._attend(m_bl, m_fu, i, j, e_bias, logp)
        a_fu = self._attend(m_fu, m_bl, j, i, e_bias, logp)
        m_bl = m_bl + self.g_bl * self.drop(self.out(a_bl))
        m_fu = m_fu + self.g_fu * self.drop(self.out(a_fu))
        refine_logit = self.score(m_bl[i] * m_fu[j]).squeeze(-1)
        return refine_logit, (m_bl, m_fu)
