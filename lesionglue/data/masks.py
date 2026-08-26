"""CSV-free dense graph builder from CTs, instance masks, and optional propagated BL centroids.

drop_dp graphs use native mask centroids only. pos_native is always stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from tracking.common import LESION_TYPES
from tracking.data.appearance import centroids, mask_stats_all
from tracking.data.descriptor import descriptor_l0
from tracking.data.features import pack_node
from tracking.data.graph import GraphConfig
from tracking.data.intra import refresh_edges
from tracking.data.propagate import load_propagated, load_types


def _labels(mask: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(mask.astype(np.int64)) if int(x) != 0)


def _lt(lid: int, table: dict[int, str], default: str | None) -> int:
    if lid in table:
        return LESION_TYPES.index(table[lid])
    assert default is not None, f"lesion {lid} needs lesion_type or explicit default"
    assert default in LESION_TYPES, f"unknown default lesion_type {default!r}"
    return LESION_TYPES.index(default)


def build_mask_graph(
    ct_bl: np.ndarray,
    aff_bl: np.ndarray,
    sp_bl: np.ndarray,
    mk_bl: np.ndarray,
    ct_fu: np.ndarray,
    aff_fu: np.ndarray,
    sp_fu: np.ndarray,
    mk_fu: np.ndarray,
    propagated_csv: Path | None,
    cfg: GraphConfig,
    default_lesion_type: str | None = None,
    types_csv: Path | None = None,
    img_id: int | None = None,
) -> HeteroData | None:
    all_bl, fu_ids = _labels(mk_bl), _labels(mk_fu)
    types: dict[int, str] = {}
    if cfg.drop_dp:
        bl_ids = all_bl
        prop: dict = {}
    else:
        if propagated_csv is None:
            raise FileNotFoundError(
                "No propagated file for a geo matcher.\n"
                "Expected meta CSV, slim CSV, or FU-frame JSON.\n"
                "Fix: pass --propagated /nnunet_data/Longitudinal-CT/meta/<pid>.csv"
            )
        prop, types = load_propagated(propagated_csv, all_bl, img_id=img_id)
        bl_ids = [i for i in all_bl if i in prop]
    if types_csv is not None:
        types.update(load_types(types_csv))
    if not bl_ids or not fu_ids:
        return None
    c_bl, c_fu = centroids(mk_bl, bl_ids), centroids(mk_fu, fu_ids)
    mf_bl, mf_fu = mask_stats_all(mk_bl, bl_ids, sp_bl, ct_bl), mask_stats_all(mk_fu, fu_ids, sp_fu, ct_fu)
    xb, pb, pbl = [], [], []
    for lid in bl_ids:
        xb.append(pack_node(descriptor_l0(ct_bl, aff_bl, c_bl[lid]), mf_bl[lid], _lt(lid, types, default_lesion_type)))
        pbl.append(c_bl[lid] * sp_bl)
        pb.append(pbl[-1] if cfg.drop_dp else prop[lid] * sp_fu)
    xf, pf = [], []
    for lid in fu_ids:
        xf.append(pack_node(descriptor_l0(ct_fu, aff_fu, c_fu[lid]), mf_fu[lid], _lt(lid, types, default_lesion_type)))
        pf.append(c_fu[lid] * sp_fu)
    data = HeteroData()
    data["bl"].x, data["fu"].x = torch.tensor(np.stack(xb)), torch.tensor(np.stack(xf))
    data["bl"].pos = torch.tensor(np.stack(pb), dtype=torch.float32)
    data["fu"].pos = torch.tensor(np.stack(pf), dtype=torch.float32)
    data["bl"].pos_native = torch.tensor(np.stack(pbl), dtype=torch.float32)
    data["bl"].img_bl = torch.zeros(len(bl_ids), dtype=torch.long)
    data["bl"].sp_bl = torch.tensor(np.stack([sp_bl] * len(bl_ids)), dtype=torch.float32)
    data["bl"].lesion_id, data["fu"].lesion_id = torch.tensor(bl_ids), torch.tensor(fu_ids)
    data["bl"].no_match_label = torch.zeros(len(bl_ids))
    data["fu"].no_match_label = torch.zeros(len(fu_ids))
    data["bl", "cross", "fu"].edge_label = torch.zeros(len(bl_ids) * len(fu_ids))
    data.pid = "mask_graph"
    data.img_id_fu_used = 0
    data.sp_fu = torch.tensor(sp_fu.astype(np.float32))
    data.feat_mode = "l0"
    return refresh_edges(data, cfg)
