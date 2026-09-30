"""Dash server for lesion graph QC (loads cached graph; opens browser URL printed via print0)."""

from __future__ import annotations

import argparse
from pathlib import Path

import dash
import dash_cytoscape as cyto
import torch
from dash import Input, Output, dcc, html

from lesionglue.cli.qc_view import (
    _STYLE,
    format_detail,
    hetero_to_elements,
    legend,
    normalize_case,
    pick_hetero,
    tap_payload,
)
from lesionglue.common import CACHE_ROOT, DATASET_ROOT, print0
from lesionglue.data.cache.dataset import LesionDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="patient id to show; a trailing _NN image index is stripped")
    ap.add_argument("--split", choices=("train", "val", "test"), default="val", help="which cached split to look the patient up in")
    ap.add_argument("--cache", default=str(CACHE_ROOT), help="cached graph root (output of lesionglue_preprocess)")
    ap.add_argument("--root", default=str(DATASET_ROOT), help="Longitudinal-CT dataset root passed to the graph dataset")
    ap.add_argument("--port", type=int, default=8050, help="local port for the Dash server on 127.0.0.1")
    args = ap.parse_args()
    pid = normalize_case(args.case)
    ds = LesionDataset(root=Path(args.cache), split=args.split, dataset_root=Path(args.root))
    hetero = pick_hetero(ds, pid)
    elems = hetero_to_elements(hetero)
    img_fu = int(torch.as_tensor(hetero.img_id_fu_used).reshape(-1)[0].item())
    dd_opts = [{"label": "preset (mm + spread)", "value": "preset"}, {"label": "cose", "value": "cose"}, {"label": "breadthfirst", "value": "breadthfirst"}, {"label": "circle", "value": "circle"}]
    cy = cyto.Cytoscape(
        id="cyto",
        elements=elems,
        layout={"name": "circle", "animate": False},
        stylesheet=_STYLE,
        style={"width": "100%", "height": "820px", "background": "#0d1117"},
        zoomingEnabled=True,
        panningEnabled=True,
        boxSelectionEnabled=False,
        minZoom=0.15,
        maxZoom=4,
    )
    pan = {"flex": "1", "minWidth": 280, "maxHeight": 980, "overflowY": "auto", "padding": "8px 12px", "background": "#21262d", "color": "#e6edf3", "borderLeft": "1px solid #444c56", "fontFamily": "ui-monospace, monospace"}
    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.H3(f"Lesion graph QC · {pid}", style={"margin": "8px 12px", "fontFamily": "system-ui", "color": "#f0f6fc", "fontWeight": "600"}),
            html.Div(
                [html.Label("Layout ", style={"marginRight": 8, "color": "#e6edf3"}), dcc.Dropdown(id="layout-name", options=dd_opts, value="circle", clearable=False, style={"width": 300})],
                style={"display": "flex", "alignItems": "center", "margin": "0 12px 8px", "color": "#e6edf3"},
            ),
            html.Div([html.Div([legend(), cy], style={"flex": "3", "minWidth": 0, "display": "flex", "flexDirection": "column"}), html.Div(id="detail-panel", children=format_detail(None, None, pid, img_fu), style=pan)], style={"display": "flex", "flexDirection": "row", "width": "100%"}),
        ],
        style={"background": "#010409", "minHeight": "100vh"},
    )

    @app.callback(Output("cyto", "layout"), Input("layout-name", "value"))
    def _layout(name: str):
        assert name in {"preset", "cose", "breadthfirst", "circle"}
        if name == "cose":
            return {"name": "cose", "animate": False, "nodeRepulsion": 8000000, "idealEdgeLength": 120, "numIter": 1200}
        return {"name": name, "animate": False}

    @app.callback(Output("detail-panel", "children"), Input("cyto", "tapNodeData"), Input("cyto", "tapEdgeData"))
    def _detail(nd, ed):
        e = tap_payload(ed)
        if e:
            return format_detail(None, e, pid, img_fu)
        n = tap_payload(nd)
        if n:
            return format_detail(n, None, pid, img_fu)
        return format_detail(None, None, pid, img_fu)

    print0(f"QC graph {pid}: http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
