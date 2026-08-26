"""BL lesion_id → FU-frame centroid: meta CSV, slim CSV, or nanoUNet JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracking.common import LESION_TYPES
from tracking.data.meta import parse_zyx

_SLIM = {"lesion_id", "z", "y", "x"}


def load_propagated(path: Path, bl_ids: list[int]) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No propagated file at {path}.\n"
            f"Expected meta CSV (cog_propagated), slim CSV (lesion_id,z,y,x), or FU-frame JSON.\n"
            f"Fix: --propagated /nnunet_data/Longitudinal-CT/meta/<pid>.csv"
        )
    if path.suffix.lower() == ".json":
        prop, typ = _from_json(path)
    else:
        df = pd.read_csv(path)
        cols = set(df.columns)
        if "cog_propagated" in cols:
            prop, typ = _from_meta(df)
        elif _SLIM <= cols:
            prop, typ = _from_slim(df)
        else:
            raise SystemExit(
                f"Cannot read propagated centroids at {path}.\n"
                f"Expected meta CSV (column cog_propagated), slim CSV (lesion_id,z,y,x), "
                f"or nanoUNet JSON points in the FU frame.\n"
                f"Fix: --propagated /nnunet_data/Longitudinal-CT/meta/<pid>.csv"
            )
    want, got = set(int(x) for x in bl_ids), set(prop)
    missing = sorted(want - got)
    if missing:
        raise SystemExit(
            f"No FU-frame point for BL mask ids {missing} at {path}.\n"
            f"Expected a JSON/CSV point for every BL instance {sorted(want)} (extra JSON names are ignored).\n"
            f"Fix: pass the FU click JSON in follow-up space, not the baseline JSON  (see docs/steps/track.md)"
        )
    return prop, typ


def _from_json(path: Path) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    pts = json.loads(path.read_text()).get("points")
    if not isinstance(pts, list):
        raise SystemExit(
            f"'points' missing or not a list in {path}.\n"
            f"Expected {{'points': [{{'name': '<id>', 'point': [x,y,z]}}, ...]}} in the FU frame.\n"
            f"Fix: pass registration-warped BL JSON, not inputsTrBL native clicks"
        )
    prop: dict[int, np.ndarray] = {}
    for item in pts:
        raw = item.get("name") if isinstance(item, dict) else None
        try:
            lid = int(raw)
        except (TypeError, ValueError):
            raise SystemExit(
                f"Click in {path} has missing or non-integer name: {item!r}.\n"
                f"Expected points[].name to be the lesion_id integer.\n"
                f"Fix: nanoUNet click JSON with integer name"
            ) from None
        p = item["point"]
        prop.setdefault(lid, np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64))
    return prop, {}


def _from_meta(df: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    prop, typ = {}, {}
    for _, r in df.iterrows():
        c = parse_zyx(r["cog_propagated"])
        if c is None:
            continue
        lid = int(r["lesion_id"])
        prop[lid] = np.asarray(c, dtype=np.float64)
        if "lesion_type" in df.columns and str(r["lesion_type"]).strip():
            lt = str(r["lesion_type"]).strip()
            if lt not in LESION_TYPES:
                raise SystemExit(
                    f"unknown lesion_type {lt!r} in meta.\n"
                    f"Expected one of {list(LESION_TYPES)}.\n"
                    f"Fix: edit the meta row or pass --default-lesion-type"
                )
            typ[lid] = lt
    return prop, typ


def load_types(path: Path) -> dict[int, str]:
    """lesion_id, lesion_type only. Not used for coordinates."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No types CSV at {path}.\n"
            f"Expected columns lesion_id, lesion_type.\n"
            f"Fix: --meta /nnunet_data/Longitudinal-CT/meta/<pid>.csv  (see docs/steps/track.md)"
        )
    df = pd.read_csv(path)
    if "lesion_id" not in df.columns or "lesion_type" not in df.columns:
        raise SystemExit(
            f"Types CSV at {path} is missing lesion_id or lesion_type.\n"
            f"Expected columns lesion_id, lesion_type (other columns ignored).\n"
            f"Fix: pass a meta CSV or omit --meta  (see docs/steps/track.md)"
        )
    out: dict[int, str] = {}
    for _, r in df.iterrows():
        lt = str(r["lesion_type"]).strip()
        if not lt or lt == "nan":
            continue
        if lt not in LESION_TYPES:
            raise SystemExit(
                f"unknown lesion_type {lt!r} in {path}.\n"
                f"Expected one of {list(LESION_TYPES)}.\n"
                f"Fix: edit the meta row or omit --meta  (see docs/steps/track.md)"
            )
        out[int(r["lesion_id"])] = lt
    return out


def _from_slim(df: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    prop, typ = {}, {}
    for _, r in df.iterrows():
        lid = int(r["lesion_id"])
        prop[lid] = np.asarray([float(r["z"]), float(r["y"]), float(r["x"])], dtype=np.float64)
        if "lesion_type" in df.columns and str(r["lesion_type"]).strip():
            lt = str(r["lesion_type"]).strip()
            assert lt in LESION_TYPES, f"unknown lesion_type {lt!r}"
            typ[lid] = lt
    return prop, typ
