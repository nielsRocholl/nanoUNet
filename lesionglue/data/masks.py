"""CSV-free dense graph builder from CTs, instance masks, and propagated BL centroids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from tracking.common import LESION_TYPES
from tracking.data.appearance import centroids, mask_stats_all
from tracking.data.descriptor import descriptor_l0
from tracking.data.features import feat_layout, pack_node
from tracking.data.graph import GraphConfig, intra_knn, _load_vol
from tracking.data.pairs import cross_attr, dense_pair_index, reverse_cross_attr
from tracking.data.propagate import load_propagated, load_types


def _labels(mask: np.ndarray) -> list[int]:
    ids = sorted(int(x) for x in np.unique(mask.astype(np.int64)) if int(x) != 0)
    assert ids
    return ids


def _lt(lid: int, table: dict[int, str], default: str | None) -> int:
    if lid in table:
        return LESION_TYPES.index(table[lid])
    assert default is not None, f"lesion {lid} needs lesion_type or explicit default"
    assert default in LESION_TYPES, f"unknown default lesion_type {default!r}"
    return LESION_TYPES.index(default)


def build_mask_graph(
    bl_img: Path,
    bl_mask: Path,
    fu_img: Path,
    fu_mask: Path,
    propagated_csv: Path,
    cfg: GraphConfig,
    default_lesion_type: str | None = None,
    types_csv: Path | None = None,
) -> HeteroData:
    ct_bl, aff_bl, sp_bl = _load_vol(bl_img)
    mk_bl, _, _ = _load_vol(bl_mask)
    ct_fu, aff_fu, sp_fu = _load_vol(fu_img)
    mk_fu, _, _ = _load_vol(fu_mask)
    bl_ids, fu_ids = _labels(mk_bl), _labels(mk_fu)
    c_bl, c_fu = centroids(mk_bl, bl_ids), centroids(mk_fu, fu_ids)
    mf_bl, mf_fu = mask_stats_all(mk_bl, bl_ids, sp_bl, ct_bl), mask_stats_all(mk_fu, fu_ids, sp_fu, ct_fu)
    prop, bl_types = load_propagated(propagated_csv, bl_ids)
    if types_csv is not None:
        bl_types.update(load_types(types_csv))
    layout = feat_layout()

    xb, pb = [], []
    for lid in bl_ids:
        xb.append(pack_node(descriptor_l0(ct_bl, aff_bl, c_bl[lid]), mf_bl[lid], _lt(lid, bl_types, default_lesion_type)))
        pb.append(prop[lid] * sp_fu)

    xf, pf = [], []
    for lid in fu_ids:
        xf.append(pack_node(descriptor_l0(ct_fu, aff_fu, c_fu[lid]), mf_fu[lid], _lt(lid, {}, default_lesion_type)))
        pf.append(c_fu[lid] * sp_fu)

    data = HeteroData()
    data["bl"].x, data["fu"].x = torch.tensor(np.stack(xb)), torch.tensor(np.stack(xf))
    data["bl"].pos, data["fu"].pos = torch.tensor(np.stack(pb), dtype=torch.float32), torch.tensor(np.stack(pf), dtype=torch.float32)
    data["bl"].lesion_id, data["fu"].lesion_id = torch.tensor(bl_ids), torch.tensor(fu_ids)
    data["bl"].no_match_label = torch.zeros(len(bl_ids))
    data["fu"].no_match_label = torch.zeros(len(fu_ids))
    data["bl", "intra", "bl"].edge_index, data["bl", "intra", "bl"].edge_attr = intra_knn(data["bl"].pos, cfg.k_intra)
    data["fu", "intra", "fu"].edge_index, data["fu", "intra", "fu"].edge_attr = intra_knn(data["fu"].pos, cfg.k_intra)
    ei = dense_pair_index(len(bl_ids), len(fu_ids))
    ea = cross_attr(data["bl"].pos, data["fu"].pos, data["bl"].x, data["fu"].x, ei, layout)
    data["bl", "cross", "fu"].edge_index = ei
    data["bl", "cross", "fu"].edge_attr = ea
    data["bl", "cross", "fu"].edge_label = torch.zeros(ei.shape[1])
    data["fu", "cross", "bl"].edge_index = ei.flip(0)
    data["fu", "cross", "bl"].edge_attr = reverse_cross_attr(ea)
    data.pid = "mask_graph"
    data.img_id_fu_used = 0
    data.sp_fu = torch.tensor(sp_fu.astype(np.float32))
    data.feat_mode = "l0"
    return data
