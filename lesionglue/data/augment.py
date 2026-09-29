"""Train-time jitter of BL/FU positions (mm) + refresh intra/cross edges.

jitter_both perturbs propagated `pos` (registration noise). pos_native is never
written — it is the registration-free channel.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from tracking.common import PROP_SIGMA
from tracking.data.graph import GraphConfig
from tracking.data.intra import refresh_edges
from tracking.data.pairs import dense_pair_index


def drop_nodes(
    data: HeteroData,
    p_drop_fu: float,
    p_drop_bl: float,
    cfg: GraphConfig,
    rng: np.random.Generator | None = None,
) -> HeteroData:
    if p_drop_fu <= 0.0 and p_drop_bl <= 0.0:
        return data
    rng = rng or np.random.default_rng()
    n_bl = int(data["bl"].num_nodes)
    n_fu = int(data["fu"].num_nodes)
    keep_bl = rng.random(n_bl) >= p_drop_bl
    keep_fu = rng.random(n_fu) >= p_drop_fu
    if not bool(keep_bl.any()):
        keep_bl[int(rng.integers(n_bl))] = True
    if not bool(keep_fu.any()):
        keep_fu[int(rng.integers(n_fu))] = True
    kb = torch.from_numpy(keep_bl).to(device=data["bl"].x.device)
    kf = torch.from_numpy(keep_fu).to(device=data["fu"].x.device)
    lab = data["bl", "cross", "fu"].edge_label.reshape(n_bl, n_fu).clone()
    nm_bl = data["bl"].no_match_label.clone()
    nm_fu = data["fu"].no_match_label.clone()
    pos_r = lab > 0.5
    for i in range(n_bl):
        if not keep_bl[i]:
            continue
        row = pos_r[i]
        if row.any() and not (row & kf).any():
            nm_bl[i] = 1.0
    for j in range(n_fu):
        if not keep_fu[j]:
            continue
        col = pos_r[:, j]
        if col.any() and not (col & kb).any():
            nm_fu[j] = 1.0
    data["bl"].x = data["bl"].x[kb]
    data["bl"].pos = data["bl"].pos[kb]
    data["bl"].pos_native = data["bl"].pos_native[kb]
    data["bl"].img_bl = data["bl"].img_bl[kb]
    data["bl"].sp_bl = data["bl"].sp_bl[kb]
    data["bl"].no_match_label = nm_bl[kb]
    if hasattr(data["bl"], "lesion_id") and data["bl"].lesion_id is not None:
        data["bl"].lesion_id = data["bl"].lesion_id[kb]
    data["fu"].x = data["fu"].x[kf]
    data["fu"].pos = data["fu"].pos[kf]
    data["fu"].no_match_label = nm_fu[kf]
    if hasattr(data["fu"], "lesion_id") and data["fu"].lesion_id is not None:
        data["fu"].lesion_id = data["fu"].lesion_id[kf]
    lab_s = lab[kb][:, kf]
    data["bl", "cross", "fu"].edge_label = lab_s.reshape(-1)
    data["bl", "cross", "fu"].edge_index = dense_pair_index(int(lab_s.shape[0]), int(lab_s.shape[1]), dev=data["bl"].pos.device)
    return refresh_edges(data, cfg)


def jitter_both(
    data: HeteroData,
    cfg: GraphConfig,
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
    return refresh_edges(data, cfg)
