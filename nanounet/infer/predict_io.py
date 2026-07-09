"""Predict-side host IO: patient-id CSV filter, single-case CPU preprocess (joint longi 2-ch)."""

from __future__ import annotations

import csv
import os

import numpy as np
import SimpleITK as sitk
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from nanounet.plan.prep.case_pp import run_case
from nanounet.prompt.coords import load_points_xyz


def _assert_bl_geometry(scan: str, bl_scan: str) -> None:
    fu, bl = sitk.ReadImage(scan), sitk.ReadImage(bl_scan)
    if fu.GetSize() != bl.GetSize() or not np.allclose(fu.GetSpacing(), bl.GetSpacing()):
        raise ValueError(
            f"Baseline geometry does not match follow-up (joint longi preprocess needs one grid).\n"
            f"  FU {scan}: size={fu.GetSize()} spacing={fu.GetSpacing()}\n"
            f"  BL {bl_scan}: size={bl.GetSize()} spacing={bl.GetSpacing()}\n"
            f"Fix: register BL into the FU frame first with nanounet_register_longi "
            f"(see docs/steps/longi.md)."
        )


def baseline_resolver(baseline_image, baseline_points, baseline_dir, end):
    """cid -> (bl_scan|None, bl_json|None) and a bl_present bool, for CLI longi inference.

    Dataset mode: per-case siblings <baseline_dir>/<cid>{end} + <baseline_dir>/<cid>.json.
    Single mode: the two explicit --baseline-* paths (or (None, None) when not longi).
    """
    if baseline_dir is not None:
        def resolve(cid):
            return os.path.join(baseline_dir, cid + end), os.path.join(baseline_dir, cid + ".json")
        return resolve, True
    def resolve(_cid):
        return baseline_image, baseline_points
    return resolve, baseline_image is not None


def check_baseline_files(cases, resolve_bl, baseline_dir, end):
    missing = []
    for cid, *_ in cases:
        bs, bj = resolve_bl(cid)
        if not os.path.isfile(bs):
            missing.append(bs)
        if not os.path.isfile(bj):
            missing.append(bj)
    if missing:
        raise FileNotFoundError(
            "Missing baseline files for longi dataset inference:\n  "
            + "\n  ".join(missing[:10])
            + ("\n  ..." if len(missing) > 10 else "")
            + f"\nExpected per FU case <cid>: {baseline_dir}/<cid>{end} and <cid>.json.\n"
              "Fix: build them with nanounet_register_longi (see docs/steps/longi.md)."
        )


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


def preprocess_case(scan: str, json_path: str, pl, cm, dj, bl_scan=None, bl_json=None):
    files = [scan]
    if bl_scan is not None:
        _assert_bl_geometry(scan, bl_scan)  # joint 2-ch crop keeps FU/BL voxel-aligned (design: §2)
        files = [scan, bl_scan]
    data, _seg, props = run_case(files, None, pl, cm, dj, verbose=False)
    data_t = torch.from_numpy(data).float()
    pad, slicer_revert = pad_nd_image(data_t, tuple(cm.patch_size), "constant", {"value": 0}, True, None)
    points = load_points_xyz(json_path)
    bl_points = load_points_xyz(bl_json) if bl_json else None
    return pad, slicer_revert, props, points, bl_points
