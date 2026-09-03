"""Balanced train/val splits; same on-disk file format as nnU-Net (list of {train, val})."""

from __future__ import annotations

import json
import re
from typing import List

import numpy as np

ALL_FOLD = "all"


def parse_fold(v: str) -> int | str:
    return ALL_FOLD if v == ALL_FOLD else int(v)


def fold_seed(fold: int | str) -> int:
    return 0 if fold == ALL_FOLD else fold


def load_splits(path: str, dataset_id: int, plans_identifier: str) -> List[dict]:
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No splits_final.json at {path}.\n"
            f"Expected output of the balanced-split step for dataset {dataset_id}, plans {plans_identifier}.\n"
            f"Fix: nanounet_build_splits -d {dataset_id} --plans {plans_identifier} --val-frac 0.15"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fold_keys(splits: List[dict], fold: int | str) -> tuple[list[str], list[str]]:
    if fold == ALL_FOLD:
        ids = sorted(splits[0]["train"] + splits[0]["val"])
        return ids, ids
    if not 0 <= fold < len(splits):
        raise IndexError(
            f"--fold {fold} but splits_final.json holds {len(splits)} split(s) (valid: 0"
            f"{'' if len(splits) == 1 else f'-{len(splits) - 1}'}).\n"
            f"This dataset uses a single balanced split (see docs/steps/valset.md).\n"
            f"Fix: pass --fold 0"
        )
    return splits[fold]["train"], splits[fold]["val"]


def cohort_of(identifier: str) -> str:
    """Source-dataset prefix of a merged-pool case id: 'd010_CECT_P0001_ct_C1' -> 'd010'."""
    return identifier.split("_")[0]


_PATIENT_RE = {
    "d013": re.compile(r"^d013_Longitudinal_CT_([0-9a-f]+)_"),
    "d029": re.compile(r"^d029_RIDER_LungCT_Seg_(RIDER-\d+)_"),
}


def patient_of(identifier: str) -> str:
    """Patient key of a case id -- two scans of one patient must never straddle train/val.

    Only d013 (BL/FU pairs, 240 patients / 537 scans) and d029 (TEST/RETEST, 31/59) carry more
    than one scan per patient; every other cohort is one scan per patient, so the case id is its
    own key."""
    rx = _PATIENT_RE.get(cohort_of(identifier))
    if rx is None:
        return identifier
    m = rx.match(identifier)
    if m is None:
        raise ValueError(
            f"{identifier!r} is in cohort {cohort_of(identifier)} but does not match its patient-id "
            f"pattern {rx.pattern!r}, so its patient cannot be determined and it could leak across "
            f"the split.\nFix: update _PATIENT_RE in nanounet/plan/splits.py to the new naming."
        )
    return f"{cohort_of(identifier)}_{m.group(1)}"


def make_balanced_split(identifiers: List[str], val_frac: float, seed: int) -> List[dict]:
    """One train/val split with `val_frac` applied WITHIN each source dataset, drawn by PATIENT.

    The plain KFold above is dataset-blind: on the 17-cohort merged pool it drifted to 13-25% val
    per cohort, so small cohorts got val sets too small to plot. The draw fills the val quota with
    whole patients (see patient_of), not individual cases, so a patient's repeat scans (d013
    BL/FU, d029 TEST/RETEST) never straddle train/val. Returns a ONE-element list so it stays
    format-compatible with splits_final.json; fold 0 is the only valid fold."""
    assert 0.0 < val_frac < 1.0, val_frac
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for cid in sorted(identifiers):
        groups.setdefault(cohort_of(cid), []).append(cid)
    train: list[str] = []
    val: list[str] = []
    for _, ids in sorted(groups.items()):
        by_pat: dict[str, list[str]] = {}
        for cid in ids:
            by_pat.setdefault(patient_of(cid), []).append(cid)
        pats = sorted(by_pat)
        perm = rng.permutation(len(pats))
        n_val = int(round(len(ids) * val_frac))
        n_val = min(max(n_val, 1), len(ids) - 1)  # every cohort appears in BOTH sides
        # whole patients into val until the case quota is met, never the last one left -- a
        # per-case draw put 71 of d013's 240 patients on both sides of the boundary.
        k, v = 0, []
        while k < len(perm) - 1 and len(v) < n_val:
            v += by_pat[pats[perm[k]]]
            k += 1
        val += v
        for j in perm[k:]:
            train += by_pat[pats[j]]
    assert not set(map(patient_of, train)) & set(map(patient_of, val)), "patient leaked across split"
    return [{"train": sorted(train), "val": sorted(val)}]
