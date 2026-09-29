"""One-patient dense HeteroData: L0 nodes, native BL mm, full BL-FU pairs.

pos is cog_propagated in FU mm; pos_native is cog_bl in BL mm. One graph per
follow-up body-region volume (img_id_fu); intra/cross edges from GraphConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch_geometric.data import HeteroData

from lesionglue.common import LESION_TYPES, print0
from lesionglue.data.appearance import mask_stats_all
from lesionglue.data.descriptor import descriptor_l0
from lesionglue.data.features import pack_node
from lesionglue.data.meta import LesionRow, V2Paths, parse_meta_csv


@dataclass
class GraphConfig:
    k_intra: int = 8
    drop_dp: bool = False
    intra: str = "knn"
    type_mask: bool = False


def graph_config(cfg) -> GraphConfig:
    return GraphConfig(k_intra=cfg.k_intra, drop_dp=cfg.drop_dp, intra=cfg.intra, type_mask=cfg.type_mask)


_NII_CACHE: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def vol_from_zyx(zyx: np.ndarray, sitk_stuff: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XYZ float32 + RAS affine + spacing. Same layout as nibabel get_fdata."""
    vol = np.ascontiguousarray(np.asarray(zyx, dtype=np.float32).transpose(2, 1, 0))
    sp = np.asarray(sitk_stuff["spacing"], dtype=np.float64)
    origin = np.asarray(sitk_stuff["origin"], dtype=np.float64)
    direction = np.asarray(sitk_stuff["direction"], dtype=np.float64).reshape(3, 3)
    lps = np.eye(4)
    lps[:3, :3] = direction @ np.diag(sp)
    lps[:3, 3] = origin
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps
    spacing = np.linalg.norm(aff[:3, :3], axis=0).astype(np.float64)
    return vol, aff, spacing


def _load_vol(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    hit = _NII_CACHE.get(path)
    if hit is not None:
        return hit
    itk = sitk.ReadImage(str(path))
    hit = vol_from_zyx(
        sitk.GetArrayFromImage(itk),
        {"spacing": itk.GetSpacing(), "origin": itk.GetOrigin(), "direction": itk.GetDirection()},
    )
    _NII_CACHE[path] = hit
    return hit


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


def _one_region(pid: str, vp: V2Paths, rows: list[LesionRow], fu_id: int, cfg: GraphConfig) -> HeteroData | None:
    from lesionglue.data.intra import refresh_edges

    bl_rep, fu_rep = _node_rows(rows, pid)
    bl_ids, fu_ids = sorted(bl_rep), sorted(fu_rep)
    if not bl_ids or not fu_ids:
        print0(f"skip pid={pid} fu={fu_id}: empty bl={len(bl_ids)} fu={len(fu_ids)}")
        return None

    ct_fu, aff_fu, sp_fu = _load_vol(vp.fu_img(fu_id))
    mk_fu, _, _ = _load_vol(vp.fu_mask(fu_id))
    bl_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in sorted({bl_rep[lid].img_id_bl for lid in bl_ids}):
        cb, ab, sb = _load_vol(vp.bl_img(k))
        mb, _, _ = _load_vol(vp.bl_mask(k))
        bl_cache[k] = (cb, ab, sb, mb)

    mf_fu = mask_stats_all(mk_fu, fu_ids, sp_fu, ct_fu)
    mf_bl = {}
    for k, (cb, ab, sb, mb) in bl_cache.items():
        lids = [lid for lid in bl_ids if bl_rep[lid].img_id_bl == k]
        mf_bl.update(mask_stats_all(mb, lids, sb, cb))
    xb, pb, pbl, ibl, sbl = [], [], [], [], []
    for lid in bl_ids:
        r = bl_rep[lid]
        assert r.cog_bl is not None
        cb, ab, sb, mb = bl_cache[r.img_id_bl]
        c = np.asarray(r.cog_bl, dtype=np.float64)
        xb.append(pack_node(descriptor_l0(cb, ab, c), mf_bl[lid], LESION_TYPES.index(r.lesion_type)))
        pb.append(np.asarray(r.cog_propagated, dtype=np.float64) * sp_fu)
        pbl.append(c * sb)
        ibl.append(int(r.img_id_bl))
        sbl.append(sb)

    xf, pf = [], []
    for lid in fu_ids:
        r = fu_rep[lid]
        cf = np.asarray(r.cog_fu, dtype=np.float64)
        xf.append(pack_node(descriptor_l0(ct_fu, aff_fu, cf), mf_fu[lid], LESION_TYPES.index(r.lesion_type)))
        pf.append(cf * sp_fu)

    data = HeteroData()
    data["bl"].x, data["fu"].x = torch.tensor(np.stack(xb)), torch.tensor(np.stack(xf))
    data["bl"].pos = torch.tensor(np.stack(pb), dtype=torch.float32)
    data["fu"].pos = torch.tensor(np.stack(pf), dtype=torch.float32)
    data["bl"].pos_native = torch.tensor(np.stack(pbl), dtype=torch.float32)
    data["bl"].img_bl = torch.tensor(ibl, dtype=torch.long)
    data["bl"].sp_bl = torch.tensor(np.stack(sbl), dtype=torch.float32)
    data["bl"].lesion_id, data["fu"].lesion_id = torch.tensor(bl_ids), torch.tensor(fu_ids)
    bi, fj = {lid: i for i, lid in enumerate(bl_ids)}, {lid: j for j, lid in enumerate(fu_ids)}
    lab = _positive_matrix(rows, bi, fj)
    data["bl"].no_match_label = (~lab.bool().any(dim=1)).float()
    data["fu"].no_match_label = (~lab.bool().any(dim=0)).float()
    data["bl", "cross", "fu"].edge_label = lab.reshape(-1)
    data.pid = pid
    data.img_id_fu_used = int(fu_id)
    data.graph_id = f"{pid}_{fu_id:02d}"
    data.sp_fu = torch.tensor(sp_fu.astype(np.float32))
    data.feat_mode = "l0"
    return refresh_edges(data, cfg)


def build_hetero_data(pid: str, root: Path, cfg: GraphConfig) -> list[HeteroData]:
    _NII_CACHE.clear()
    vp = V2Paths(Path(root), pid)
    rows = parse_meta_csv(vp.meta)
    if not rows:
        return []
    out: list[HeteroData] = []
    for fu_id in sorted({r.img_id_fu for r in rows}):
        g = _one_region(pid, vp, [r for r in rows if r.img_id_fu == fu_id], fu_id, cfg)
        if g is not None:
            out.append(g)
    return out
