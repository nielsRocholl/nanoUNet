"""Graph QC view model: HeteroData→Cytoscape elements, overlap-spread preset coords, legend + tap decode."""

from __future__ import annotations

import numpy as np
import torch
from dash import html
from torch_geometric.data import HeteroData

from tracking.common import LESION_TYPES
from tracking.data.dataset import LesionDataset

FEAT_DIM = 1379
_UI_FG = "#e6edf3"
_UI_FG_DIM = "#b7c0ca"
_PRE = {"margin": 0, "fontSize": 13, "whiteSpace": "pre-wrap", "color": _UI_FG}
_STYLE = [
    {"selector": "node", "style": {"label": "data(label)", "text-valign": "center", "text-halign": "center", "width": 46, "height": 46, "font-size": "11px", "color": "#f0f6fc", "border-width": 1, "border-color": "#30363d"}},
    {"selector": 'node[side = "bl"]', "style": {"background-color": "#388bfd"}},
    {"selector": 'node[side = "fu"]', "style": {"background-color": "#d29922"}},
    {"selector": "edge", "style": {"curve-style": "bezier", "opacity": 0.88, "width": 2, "target-arrow-shape": "none"}},
    {"selector": 'edge[etype = "intra_bl"]', "style": {"line-color": "#8b949e", "line-style": "dashed"}},
    {"selector": 'edge[etype = "intra_fu"]', "style": {"line-color": "#b87fff", "line-style": "dashed"}},
    {"selector": 'edge[etype = "cross_pos"]', "style": {"line-color": "#3fb950"}},
    {"selector": 'edge[etype = "cross_neg"]', "style": {"line-color": "#f85149"}},
]
_EDGE_HELP = {"intra_bl": "Intra BL kNN. edge_attr[0]=distance_mm/100.", "intra_fu": "Intra FU kNN. edge_attr[0]=distance_mm/100.", "cross_pos": "Dense BL→FU supervised positive.", "cross_neg": "Dense BL→FU supervised negative."}

def normalize_case(case: str) -> str:
    if "_" in case:
        b, s = case.rsplit("_", 1)
        if s.isdigit():
            return b
    return case

def pick_hetero(ds: LesionDataset, pid: str) -> HeteroData:
    for i in range(len(ds)):
        g = ds[i]
        if str(g.pid) == pid:
            return g
    raise ValueError(f"pid={pid!r} not in split={ds.split} (run preprocess for this split)")

def _xy_mm(pos: torch.Tensor) -> np.ndarray:
    p = pos.detach().cpu().numpy()
    xy = np.stack([p[:, 2], p[:, 1]], axis=1)
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1e-6)
    return (xy - lo) / span * 820 + 40

def _spread_xy(xy: np.ndarray, mind: float = 56.0, iters: int = 160) -> np.ndarray:
    q = xy.astype(np.float64).copy()
    for i in range(len(q)):
        q[i, 0] += (i % 13) * 0.08
        q[i, 1] += ((i * 7) % 11) * 0.07
    for _ in range(iters):
        mv = False
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                d = q[j] - q[i]
                dist = float(np.hypot(d[0], d[1]))
                if dist < 1e-9:
                    q[j, 0] += mind * 0.35
                    dist = mind * 0.35
                    mv = True
                if dist < mind:
                    u = d / dist
                    push = (mind - dist) * 0.52
                    q[i] -= u * push
                    q[j] += u * push
                    mv = True
        if not mv:
            break
    q[:, 0] = np.clip(q[:, 0], 28, 972)
    q[:, 1] = np.clip(q[:, 1], 28, 972)
    return q

def _node_dict(x: np.ndarray, pos_mm: np.ndarray, side: str, lid: int) -> dict:
    d = x[:1372].astype(np.float64)
    lv = float(x[1372])
    lt_i = int(round(float(x[1375])))
    ant = LESION_TYPES[lt_i] if 0 <= lt_i < len(LESION_TYPES) else f"idx_{lt_i}"
    return {
        "id": f"{side}-{lid}",
        "label": f"{side.upper()} {lid}",
        "side": side,
        "lesion_id": lid,
        "anatomy": ant,
        "lesion_type_idx": lt_i,
        "volume_mm3": float(np.expm1(lv * 10.0)),
        "log1p_volume_mm3": lv * 10.0,
        "mean_hu": float(x[1373] * 1000.0),
        "sphericity": float(x[1374]),
        "norm_pos_zyx": (float(x[1376]), float(x[1377]), float(x[1378])),
        "pos_mm_zyx": pos_mm.tolist(),
        "descriptor_min": float(d.min()),
        "descriptor_max": float(d.max()),
        "descriptor_mean": float(d.mean()),
        "descriptor_std": float(d.std()),
    }

def _append_intra(out: list[dict], ei: torch.Tensor, ea: torch.Tensor, ids: list[int], tag: str) -> None:
    et = f"intra_{tag}"
    ea_np = ea.detach().cpu().numpy()
    for k in range(ei.shape[1]):
        s, t = int(ei[0, k]), int(ei[1, k])
        dmm = float(ea_np[k, 0] * 100.0)
        out.append({"data": {"id": f"i{tag}-{k}-{ids[s]}-{ids[t]}", "source": f"{tag}-{ids[s]}", "target": f"{tag}-{ids[t]}", "etype": et, "distance_mm": dmm}})

def hetero_to_elements(data: HeteroData) -> list[dict]:
    bl = data["bl"].lesion_id.detach().cpu().numpy().tolist()
    fu = data["fu"].lesion_id.detach().cpu().numpy().tolist()
    xb, xf = data["bl"].x.detach().cpu().numpy(), data["fu"].x.detach().cpu().numpy()
    pb, pf = data["bl"].pos.detach().cpu().numpy(), data["fu"].pos.detach().cpu().numpy()
    yb, yf = _xy_mm(data["bl"].pos), _xy_mm(data["fu"].pos)
    stacked = _spread_xy(np.vstack([yb, yf]))
    yb, yf = stacked[: len(bl)], stacked[len(bl) :]
    out: list[dict] = []
    for i, lid in enumerate(bl):
        out.append({"data": _node_dict(xb[i], pb[i], "bl", int(lid)), "position": {"x": float(yb[i, 0]), "y": float(yb[i, 1])}})
    for j, lid in enumerate(fu):
        out.append({"data": _node_dict(xf[j], pf[j], "fu", int(lid)), "position": {"x": float(yf[j, 0]), "y": float(yf[j, 1])}})
    _append_intra(out, data["bl", "intra", "bl"].edge_index, data["bl", "intra", "bl"].edge_attr, bl, "bl")
    _append_intra(out, data["fu", "intra", "fu"].edge_index, data["fu", "intra", "fu"].edge_attr, fu, "fu")
    ei = data["bl", "cross", "fu"].edge_index
    lab = data["bl", "cross", "fu"].edge_label.detach().cpu().numpy()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        pos = float(lab[k]) >= 0.5
        et = "cross_pos" if pos else "cross_neg"
        out.append({"data": {"id": f"x-{k}-{bl[i]}-{fu[j]}", "source": f"bl-{bl[i]}", "target": f"fu-{fu[j]}", "etype": et, "edge_label": float(lab[k])}})
    return out

def tap_payload(prop) -> dict | None:
    if prop is None:
        return None
    if isinstance(prop, dict):
        return prop if prop else None
    if isinstance(prop, list) and prop:
        x = prop[0]
        return x if isinstance(x, dict) else None
    return None

def format_detail(nd: dict | None, ed: dict | None, pid: str, img_fu: int) -> html.Pre:
    h = f"pid={pid}  img_id_fu_used={img_fu}\n---\n"
    if ed:
        et = str(ed.get("etype", ""))
        s = f"{h}EDGE\n  id: {ed.get('id')}\n  {ed.get('source')} → {ed.get('target')}\n  etype: {et}\n  {_EDGE_HELP.get(et, '')}\n"
        if "distance_mm" in ed:
            s += f"  distance_mm (edge_attr): {ed['distance_mm']:.6g}\n"
        if "edge_label" in ed:
            s += f"  edge_label: {ed['edge_label']}\n"
        s += "  tensors: intra edge_attr dim 1; cross edge_attr dim 11.\n"
        return html.Pre(s, style=_PRE)
    if nd:
        lv = nd["log1p_volume_mm3"]
        s = (
            f"{h}NODE\n  id: {nd.get('id')}  side: {nd.get('side')}  lesion_id: {nd.get('lesion_id')}\n"
            f"  anatomy: {nd.get('anatomy')}  lesion_type_idx: {nd.get('lesion_type_idx')}\n"
            f"  log1p(vol_mm3)= {lv:.6g}   volume_mm3(expm1)= {nd['volume_mm3']:.6g}\n"
            f"  mean_hu= {nd['mean_hu']:.6g}   sphericity= {nd['sphericity']:.6g}\n"
            f"  norm_pos z,y,x (x[1376:]): {nd.get('norm_pos_zyx')}\n"
            f"  pos_mm z,y,x (graph.pos): {nd.get('pos_mm_zyx')}\n"
            f"  descriptor L0 [0:1372]: min={nd['descriptor_min']:.6g} max={nd['descriptor_max']:.6g} mean={nd['descriptor_mean']:.6g} std={nd['descriptor_std']:.6g}\n"
            f"  full x length {FEAT_DIM}\n"
        )
        return html.Pre(s, style=_PRE)
    return html.Pre(h + "Tap a node or edge.", style=_PRE)

def legend() -> html.Div:
    sw = {"display": "inline-block", "width": 14, "height": 14, "marginRight": 8, "border": "1px solid #30363d", "verticalAlign": "middle"}
    tx = {"fontSize": 13, "color": _UI_FG}

    def chip(c, t):
        return html.Div([html.Span(style={**sw, "background": c}), html.Span(t, style=tx)], style={"display": "flex", "alignItems": "center", "marginBottom": 4})

    def edge_row(c, dash, t):
        bs = f"2px {'dashed' if dash else 'solid'} {c}"
        return html.Div([html.Span(style={"display": "inline-block", "width": 36, "borderBottom": bs, "marginRight": 8}), html.Span(t, style=tx)], style={"display": "flex", "alignItems": "center", "marginBottom": 4})

    kids = [
        html.Div("Legend", style={"fontWeight": "600", "marginBottom": 6, "fontSize": 13, "color": _UI_FG}),
        chip("#388bfd", "Baseline node"),
        chip("#d29922", "Follow-up node"),
        edge_row("#8b949e", True, "Intra BL kNN"),
        edge_row("#b87fff", True, "Intra FU kNN"),
        edge_row("#3fb950", False, "Cross · positive"),
        edge_row("#f85149", False, "Cross · negative"),
        html.Div("Preset: overlap separation on mm projection; use cose if still dense.", style={"marginTop": 8, "fontSize": 11, "color": _UI_FG_DIM}),
    ]
    box = {"padding": "8px 12px", "background": "#21262d", "border": "1px solid #444c56", "borderRadius": 6, "marginBottom": 8, "maxWidth": 440, "color": _UI_FG}
    return html.Div(kids, style=box)
