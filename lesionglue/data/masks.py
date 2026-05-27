"""CSV-free dense graph builder from CTs, instance masks, and propagated BL centroids."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from tracking.common import LESION_TYPES
from tracking.data.appearance import mask_stats
from tracking.data.descriptor import descriptor_l0, descriptor_yerebakan
from tracking.data.features import FeatConfig, feat_layout, pack_node
from tracking.data.graph import GraphConfig, intra_knn, _load_vol
from tracking.data.pairs import cross_attr, dense_pair_index, reverse_cross_attr

if TYPE_CHECKING:
    from tracking.data.mae import MaeExtractor


def _labels(mask: np.ndarray) -> list[int]:
    ids = sorted(int(x) for x in np.unique(mask.astype(np.int64)) if int(x) != 0)
    assert ids
    return ids


def _centroids(mask: np.ndarray, ids: list[int]) -> dict[int, np.ndarray]:
    out = {}
    for lid in ids:
        pts = np.argwhere(mask == lid)
        assert pts.size, f"empty mask label {lid}"
        out[lid] = pts.mean(axis=0).astype(np.float64) + 0.5
    return out


def _propagated(path: Path, bl_ids: list[int]) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    df = pd.read_csv(path)
    need = {"lesion_id", "z", "y", "x"}
    assert need.issubset(df.columns), f"{path} needs columns {sorted(need)}"
    prop, typ = {}, {}
    for _, r in df.iterrows():
        lid = int(r["lesion_id"])
        prop[lid] = np.asarray([float(r["z"]), float(r["y"]), float(r["x"])], dtype=np.float64)
        if "lesion_type" in df.columns and str(r["lesion_type"]).strip():
            lt = str(r["lesion_type"]).strip()
            assert lt in LESION_TYPES, f"unknown lesion_type {lt!r}"
            typ[lid] = lt
    assert set(prop) == set(bl_ids), "propagated centroid CSV must match baseline mask labels exactly"
    return prop, typ


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
    mae: MaeExtractor | None = None,
) -> HeteroData:
    ct_bl, aff_bl, sp_bl = _load_vol(bl_img)
    mk_bl, _, _ = _load_vol(bl_mask)
    ct_fu, aff_fu, sp_fu = _load_vol(fu_img)
    mk_fu, _, _ = _load_vol(fu_mask)
    bl_ids, fu_ids = _labels(mk_bl), _labels(mk_fu)
    c_bl, c_fu = _centroids(mk_bl, bl_ids), _centroids(mk_fu, fu_ids)
    prop, bl_types = _propagated(propagated_csv, bl_ids)
    feat = cfg.feat
    layout = feat_layout(feat)

    xb, pb = [], []
    if feat.mode == "mae":
        assert mae is not None
        mae.clear_cache()
        for lid in bl_ids:
            pb.append(prop[lid] * sp_fu)
        pooled = mae.pool_lesions(ct_bl, mk_bl, sp_bl, bl_ids, c_bl)
        for lid in bl_ids:
            mf_b = mask_stats(mk_bl, lid, sp_bl, ct_bl)
            xb.append(pack_node(pooled[lid], mf_b, _lt(lid, bl_types, default_lesion_type), feat))
    else:
        for lid in bl_ids:
            mf_b = mask_stats(mk_bl, lid, sp_bl, ct_bl)
            center = c_bl[lid]
            if feat.mode == "yerebakan":
                desc = descriptor_yerebakan(ct_bl, aff_bl, center)
            else:
                desc = descriptor_l0(ct_bl, aff_bl, center)
            xb.append(pack_node(desc, mf_b, _lt(lid, bl_types, default_lesion_type), feat))
            pb.append(prop[lid] * sp_fu)

    xf, pf = [], []
    if feat.mode == "mae":
        assert mae is not None
        mae.clear_cache()
        pooled = mae.pool_lesions(ct_fu, mk_fu, sp_fu, fu_ids, c_fu)
        for lid in fu_ids:
            mf_f = mask_stats(mk_fu, lid, sp_fu, ct_fu)
            xf.append(pack_node(pooled[lid], mf_f, _lt(lid, {}, default_lesion_type), feat))
            pf.append(c_fu[lid] * sp_fu)
    else:
        for lid in fu_ids:
            mf_f = mask_stats(mk_fu, lid, sp_fu, ct_fu)
            center = c_fu[lid]
            if feat.mode == "yerebakan":
                desc = descriptor_yerebakan(ct_fu, aff_fu, center)
            else:
                desc = descriptor_l0(ct_fu, aff_fu, center)
            xf.append(pack_node(desc, mf_f, _lt(lid, {}, default_lesion_type), feat))
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
    data.feat_mode = feat.mode
    return data
