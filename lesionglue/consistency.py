"""Registration-free second-order geometry consistency for the matching logits.

The bipartite GNN and the cross features both ride on registration-propagated BL
coordinates (cog_propagated -> FU frame), so their geometry carries registration
error. This module adds a signal that needs NO registration: pairwise distances
*within* each cloud are invariant to the (rigid) BL->FU transform. For a candidate
match i->j we measure how much it breaks the surrounding constellation -- the known
native BL distance D_bl[i,i'] vs the FU distance from j to wherever neighbour i'
currently maps under the soft Sinkhorn assignment P. The per-edge discrepancy is
turned into a logit added to pair0 by a zero-initialised Linear (starts == geo-off
baseline, can only earn weight). Cross-baseline-image BL pairs are masked out
because their native distances live in different frames.
"""

from __future__ import annotations

import torch
from torch import nn

from tracking.train.sinkhorn import log_sinkhorn, superglue_marginals

DIST_SCALE = 50.0  # mm; brings distance discrepancies to O(1) before the linear head


def soft_assignment(pair: torch.Tensor, n_bl: int, n_fu: int, iters: int) -> torch.Tensor:
    """Detached soft Sinkhorn assignment P[:n_bl, :n_fu] from pair logits.

    Dust rows/cols are zero (parameter-free matchability prior, same convention as
    row_dust_marginals), so P is a pure function of the current pair landscape.
    """
    S = torch.zeros((n_bl + 1, n_fu + 1), device=pair.device, dtype=pair.dtype)
    S[:n_bl, :n_fu] = pair.reshape(n_bl, n_fu).detach()
    la, lb = superglue_marginals(n_bl, n_fu, pair.device, pair.dtype)
    P = log_sinkhorn(S, iters, la, lb).exp()
    return P[:n_bl, :n_fu].detach()


class GeoConsistency(nn.Module):
    def __init__(self, knn: int = 3):
        super().__init__()
        self.knn = knn
        self.head = nn.Linear(2, 1)  # [mean discrepancy, knn discrepancy] -> logit
        nn.init.zeros_(self.head.weight)  # zero-init => geo term starts at 0 == geo-off baseline
        nn.init.zeros_(self.head.bias)

    def _knn_mask(self, d_bl: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        n = d_bl.shape[0]
        k = min(self.knn, n - 1)
        if k <= 0:
            return torch.zeros_like(valid)
        dm = d_bl.masked_fill(~valid, float("inf"))
        idx = dm.topk(k, dim=1, largest=False).indices
        m = torch.zeros_like(valid)
        m.scatter_(1, idx, True)
        return m & valid  # drop padded picks when a row has < k valid neighbours

    def forward(
        self, pos_bl: torch.Tensor, pos_fu: torch.Tensor, img_bl: torch.Tensor, p_soft: torch.Tensor
    ) -> torch.Tensor:
        n_bl = pos_bl.shape[0]
        d_bl = torch.cdist(pos_bl, pos_bl)  # (n_bl,n_bl) native BL mm  -- registration-free
        d_fu = torch.cdist(pos_fu, pos_fu)  # (n_fu,n_fu) native FU mm
        eye = torch.eye(n_bl, dtype=torch.bool, device=pos_bl.device)
        valid = (img_bl[:, None] == img_bl[None, :]) & ~eye  # same baseline image, exclude self
        efu = p_soft @ d_fu  # (n_bl,n_fu): FU dist from FU node j to where neighbour i' maps
        disc = (d_bl.unsqueeze(-1) - efu.unsqueeze(0)).abs()  # (n_bl,n_bl,n_fu) indexed [i,i',j]
        vf = valid.float().unsqueeze(-1)
        c_mean = (vf * disc).sum(1) / valid.float().sum(1, keepdim=True).clamp_min(1.0)
        kf = self._knn_mask(d_bl, valid).float().unsqueeze(-1)
        c_knn = (kf * disc).sum(1) / kf.sum(1).clamp_min(1.0)
        feat = torch.stack([c_mean, c_knn], dim=-1) / DIST_SCALE  # (n_bl,n_fu,2)
        return self.head(feat).squeeze(-1).reshape(-1)  # (n_bl*n_fu,) row-major == dense_pair_index
