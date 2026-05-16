"""Train-time jitter of BL/FU positions (mm) + refresh intra kNN + cross edge_attr."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from tracking.common import PROP_SIGMA
from tracking.data.graph import intra_knn
from tracking.data.pairs import cross_attr, reverse_cross_attr


def jitter_both(
    data: HeteroData,
    k_intra: int = 8,
    sigma_fu_scale: float = 0.3,
    rng: np.random.Generator | None = None,
) -> HeteroData:
    if rng is None:
        rng = np.random.default_rng()
    sp = data.sp_fu.cpu().numpy()
    sig_bl = np.asarray(PROP_SIGMA, dtype=np.float64) * sp
    sig_fu = sigma_fu_scale * sig_bl
    nb = rng.normal(0.0, sig_bl, size=data["bl"].pos.shape).astype(np.float32)
    nf = rng.normal(0.0, sig_fu, size=data["fu"].pos.shape).astype(np.float32)
    dev, dt = data["bl"].pos.device, data["bl"].pos.dtype
    data["bl"].pos = data["bl"].pos + torch.from_numpy(nb).to(device=dev, dtype=dt)
    data["fu"].pos = data["fu"].pos + torch.from_numpy(nf).to(device=dev, dtype=dt)
    data["bl", "intra", "bl"].edge_index, data["bl", "intra", "bl"].edge_attr = intra_knn(data["bl"].pos, k_intra)
    data["fu", "intra", "fu"].edge_index, data["fu", "intra", "fu"].edge_attr = intra_knn(data["fu"].pos, k_intra)
    ei = data["bl", "cross", "fu"].edge_index
    ea = cross_attr(data["bl"].pos, data["fu"].pos, data["bl"].x, data["fu"].x, ei)
    data["bl", "cross", "fu"].edge_attr = ea
    data["fu", "cross", "bl"].edge_attr = reverse_cross_attr(ea)
    return data
