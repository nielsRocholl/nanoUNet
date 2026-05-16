"""One-patient dense HeteroData: normalized nodes, mm-kNN intra, full BL-FU pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch_geometric.data import HeteroData

from tracking.common import LESION_TYPES, print0
from tracking.data.appearance import descriptor_l0, mask_stats
from tracking.data.meta import LesionRow, V2Paths, parse_meta_csv
from tracking.data.pairs import cross_attr, dense_pair_index, reverse_cross_attr

FEAT_DIM = 1379


@dataclass
class GraphConfig:
    k_intra: int = 8


def _intra_knn(pos: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = pos.shape[0]
    dev = pos.device
    if n == 1:
        return torch.tensor([[0], [0]], dtype=torch.long, device=dev), torch.zeros((1, 1), device=dev)
    d_mat = torch.cdist(pos, pos)
    d_mat.fill_diagonal_(torch.inf)
    ke = min(k, n - 1)
    dists, nei = d_mat.topk(ke, largest=False, dim=1)
    row = torch.arange(n, device=dev).unsqueeze(1).expand(n, ke).reshape(-1)
    return torch.stack([row, nei.reshape(-1)], dim=0), (dists.reshape(-1, 1) / 100.0).to(torch.float32)


def _load_vol(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    aff = np.asarray(img.affine, dtype=np.float64)
    vol = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
    sp = np.linalg.norm(aff[:3, :3], axis=0).astype(np.float64)
    return vol, aff, sp


def _dom_fu(rows: list[LesionRow]) -> int:
    c = Counter(r.img_id_fu for r in rows)
    dom = c.most_common(1)[0][0]
    if len(c) > 1:
        print0(f"multi img_id_fu {dict(c)} -> dominant {dom}")
    return dom


def _pack(desc: np.ndarray, lv: float, mh: float, sph: float, lt_i: int, pos: np.ndarray) -> np.ndarray:
    x = np.zeros(FEAT_DIM, np.float32)
    x[:1372] = np.clip(desc, -1000.0, 1000.0) / 1000.0
    x[1372] = lv / 10.0
    x[1373] = np.clip(mh, -1000.0, 1000.0) / 1000.0
    x[1374] = np.clip(sph, 0.0, 2.0)
    x[1375] = float(lt_i)
    x[1376:1379] = pos.astype(np.float32)
    return x


def _positive_matrix(rows: list[LesionRow], bi: dict[int, int], fj: dict[int, int]) -> torch.Tensor:
    y = torch.zeros((len(bi), len(fj)), dtype=torch.float32)
    for r in rows:
        if r.topology in ("UNCHANGED", "SPLIT") and r.cog_propagated and r.cog_fu:
            if r.lesion_id in bi and r.lesion_id in fj:
                y[bi[r.lesion_id], fj[r.lesion_id]] = 1.0
        elif r.topology == "MERGED" and r.cog_propagated and r.merged_into is not None:
            if r.lesion_id in bi and r.merged_into in fj:
                y[bi[r.lesion_id], fj[r.merged_into]] = 1.0
    return y


def _node_rows(rows: list[LesionRow], pid: str) -> tuple[dict[int, LesionRow], dict[int, LesionRow]]:
    bt = frozenset({"UNCHANGED", "DISAPPEARED", "MERGED", "SPLIT"})
    ft = frozenset({"UNCHANGED", "NEWLYAPPEARING", "SPLIT"})
    bl: dict[int, LesionRow] = {}
    fu: dict[int, LesionRow] = {}
    for r in rows:
        if r.topology in bt:
            if r.cog_propagated is None:
                print0(f"drop BL pid={pid} lid={r.lesion_id} (no cog_propagated)")
            else:
                bl.setdefault(r.lesion_id, r)
        if r.topology in ft and r.cog_fu is not None:
            fu.setdefault(r.lesion_id, r)
    return bl, fu


def build_hetero_data(pid: str, root: Path, cfg: GraphConfig) -> HeteroData | None:
    vp = V2Paths(Path(root), pid)
    rows = parse_meta_csv(vp.meta)
    if not rows:
        return None
    dom = _dom_fu(rows)
    rows = [r for r in rows if r.img_id_fu == dom]
    bl_rep, fu_rep = _node_rows(rows, pid)
    bl_ids, fu_ids = sorted(bl_rep), sorted(fu_rep)
    if not bl_ids or not fu_ids:
        print0(f"skip pid={pid}: empty bl={len(bl_ids)} fu={len(fu_ids)}")
        return None

    ct_fu, aff_fu, sp_fu = _load_vol(vp.fu_img(dom))
    mk_fu, _, _ = _load_vol(vp.fu_mask(dom))
    den = np.maximum(np.asarray(ct_fu.shape, dtype=np.float64) - 1.0, 1.0)
    bl_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for lid in bl_ids:
        k = bl_rep[lid].img_id_bl
        if k not in bl_cache:
            cb, ab, sb = _load_vol(vp.bl_img(k))
            mb, _, _ = _load_vol(vp.bl_mask(k))
            bl_cache[k] = (cb, ab, sb, mb)

    xb, pb = [], []
    for lid in bl_ids:
        r = bl_rep[lid]
        assert r.cog_bl is not None
        cb, ab, sb, mb = bl_cache[r.img_id_bl]
        lv, mh, sph = mask_stats(mb, lid, sb, cb)
        cp = np.asarray(r.cog_propagated, dtype=np.float64)
        desc = descriptor_l0(cb, ab, np.asarray(r.cog_bl, dtype=np.float64))
        xb.append(_pack(desc, lv, mh, sph, LESION_TYPES.index(r.lesion_type), cp / den))
        pb.append(cp * sp_fu)

    xf, pf = [], []
    for lid in fu_ids:
        r = fu_rep[lid]
        cf = np.asarray(r.cog_fu, dtype=np.float64)
        lv, mh, sph = mask_stats(mk_fu, lid, sp_fu, ct_fu)
        desc = descriptor_l0(ct_fu, aff_fu, cf)
        xf.append(_pack(desc, lv, mh, sph, LESION_TYPES.index(r.lesion_type), cf / den))
        pf.append(cf * sp_fu)

    data = HeteroData()
    data["bl"].x, data["fu"].x = torch.tensor(np.stack(xb)), torch.tensor(np.stack(xf))
    data["bl"].pos, data["fu"].pos = torch.tensor(np.stack(pb), dtype=torch.float32), torch.tensor(np.stack(pf), dtype=torch.float32)
    data["bl"].lesion_id, data["fu"].lesion_id = torch.tensor(bl_ids), torch.tensor(fu_ids)
    bi, fj = {lid: i for i, lid in enumerate(bl_ids)}, {lid: j for j, lid in enumerate(fu_ids)}
    lab = _positive_matrix(rows, bi, fj)
    data["bl"].no_match_label = (~lab.bool().any(dim=1)).float()
    data["fu"].no_match_label = (~lab.bool().any(dim=0)).float()

    data["bl", "intra", "bl"].edge_index, data["bl", "intra", "bl"].edge_attr = _intra_knn(data["bl"].pos, cfg.k_intra)
    data["fu", "intra", "fu"].edge_index, data["fu", "intra", "fu"].edge_attr = _intra_knn(data["fu"].pos, cfg.k_intra)
    ei = dense_pair_index(len(bl_ids), len(fu_ids))
    ea = cross_attr(data["bl"].pos, data["fu"].pos, data["bl"].x, data["fu"].x, ei)
    data["bl", "cross", "fu"].edge_index = ei
    data["bl", "cross", "fu"].edge_attr = ea
    data["bl", "cross", "fu"].edge_label = lab.reshape(-1)
    data["fu", "cross", "bl"].edge_index = ei.flip(0)
    data["fu", "cross", "bl"].edge_attr = reverse_cross_attr(ea)
    data.pid = pid
    data.img_id_fu_used = int(dom)
    return data
