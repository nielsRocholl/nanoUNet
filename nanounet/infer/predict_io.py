"""Predict-side host IO: BL points JSON, patient-id CSV filter, single-case CPU preprocess."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor

import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from nanounet.plan.prep.case_pp import run_case
from nanounet.prompt.coords import load_points_xyz


def load_bl_points(json_path: str) -> list[tuple[float, float, float] | None]:
    import json
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("points")
    if pts is None:
        raise KeyError(f"'points' missing in {json_path}")
    out: list[tuple[float, float, float] | None] = []
    for item in pts:
        if item is None:
            out.append(None)
            continue
        p = item.get("point") if isinstance(item, dict) else item
        if p is None:
            out.append(None)
            continue
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def patient_ids_from_csv(path: str) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty patients csv: {path}")
    col = "patient" if "patient" in rows[0] else next(iter(rows[0]))
    out = {r[col].strip() for r in rows if r[col].strip()}
    if not out:
        raise ValueError(f"no patient ids in {path}")
    return out


def preprocess_case(scan: str, json_path: str, pl, cm, dj, bl_scan: str | None = None, bl_json: str | None = None):
    # FU and (when present) BL preprocessing (`run_case`, mostly resampling) are independent and
    # each cost ~30s wall time; running them concurrently roughly halves this per-case cost since
    # the underlying numpy/SimpleITK work releases the GIL.
    bl_future = None
    if bl_scan is not None:
        bl_future = ThreadPoolExecutor(max_workers=1).submit(run_case, [bl_scan], None, pl, cm, dj, verbose=False)

    data, _seg, props = run_case([scan], None, pl, cm, dj, verbose=False)
    data_t = torch.from_numpy(data).float()
    pad, slicer_revert = pad_nd_image(data_t, tuple(cm.patch_size), "constant", {"value": 0}, True, None)
    points = load_points_xyz(json_path)

    bl_pack = None
    if bl_future is not None:
        bl_data, _bl_seg, bl_props = bl_future.result()
        bl_t = torch.from_numpy(bl_data).float()
        pad_bl, bl_slicer = pad_nd_image(bl_t, tuple(cm.patch_size), "constant", {"value": 0}, True, None)
        bl_pts = load_bl_points(bl_json) if bl_json else [None] * len(points)
        assert len(bl_pts) == len(points), (len(bl_pts), len(points))
        bl_pack = (pad_bl, bl_slicer, bl_props, bl_pts)
    return pad, slicer_revert, props, points, bl_pack
