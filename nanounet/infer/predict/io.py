"""Predict-side host IO: patient-id CSV filter, raw case load + pad-to-patch packing."""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from nanounet.plan.prep.case_pp import run_case, run_case_npy
from nanounet.prompt.coords import load_points_xyz


def patient_ids_from_csv(path: str) -> set[str]:
    if not os.path.isfile(path):
        raise SystemExit(
            f"No patients CSV at '{path}'.\n"
            f"Expected a CSV with a 'patient' column of id prefixes.\n"
            f"Fix: --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv  (see nanounet/docs/steps/predict.md)"
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(
            f"Empty patients CSV '{path}'.\n"
            f"Expected a header plus one patient id per row.\n"
            f"Fix: --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv  (see nanounet/docs/steps/predict.md)"
        )
    col = "patient" if "patient" in rows[0] else next(iter(rows[0]))
    out = {r[col].strip() for r in rows if r[col].strip()}
    if not out:
        raise SystemExit(
            f"No patient ids in '{path}' (column '{col}').\n"
            f"Expected non-empty 'patient' cells matching -i stem prefixes.\n"
            f"Fix: --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv  (see nanounet/docs/steps/predict.md)"
        )
    return out


def _pack(data, props, json_path: str, cm):
    data_t = torch.from_numpy(data).float()
    pad, slicer_revert = pad_nd_image(data_t, tuple(cm.patch_size), "constant", {"value": 0}, True, None)
    points = load_points_xyz(json_path)
    return pad, slicer_revert, props, points


def preprocess_loaded(data: np.ndarray, props: dict, json_path: str, pl, cm, dj):
    """Same as preprocess_case, but CT already in memory (no second read)."""
    data, _seg, props = run_case_npy(data, None, props, pl, cm, dj, verbose=False)
    return _pack(data, props, json_path, cm)


def preprocess_case(scan: str, json_path: str, pl, cm, dj):
    data, _seg, props = run_case([scan], None, pl, cm, dj, verbose=False)
    return _pack(data, props, json_path, cm)
