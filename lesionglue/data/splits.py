"""Deterministic patient-level k-fold map for CV (no BL/FU leakage across folds)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tracking.data.meta import load_split_json

CV_METRICS = ("val_match_score_ema", "val_match_score_raw", "val_match_score_peak", "val_acc_unchanged_split", "val_acc_disappeared", "val_acc_newly_appearing")

CV_SPLIT_SEED = 0


def fold_map(pids: list[str], n_folds: int, seed: int = CV_SPLIT_SEED) -> dict[str, int]:
    # Patient id is the fold unit, so a patient's BL/FU graph stays in one fold.
    # Sort first to make the map invariant to caller order; seeded round-robin keeps folds size-balanced.
    assert n_folds >= 2
    ids = sorted(set(map(str, pids)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[int(i)]: r % n_folds for r, i in enumerate(perm)}


def pool_patient_ids(dataset_root: Path | str) -> list[str]:
    sp = load_split_json(Path(dataset_root) / "data_split.json")
    return list(sp["train"]) + list(sp["val"])


def patient_fold_map(dataset_root: Path | str, n_folds: int, seed: int = CV_SPLIT_SEED) -> dict[str, int]:
    return fold_map(pool_patient_ids(dataset_root), n_folds, seed)


def fold_patient_sets(
    dataset_root: Path | str, fold: int, n_folds: int, seed: int = CV_SPLIT_SEED
) -> tuple[set[str], set[str]]:
    assert fold in range(n_folds)
    fm = patient_fold_map(dataset_root, n_folds, seed)
    val = {p for p, f in fm.items() if f == fold}
    train = {p for p, f in fm.items() if f != fold}
    assert not (train & val)
    return train, val


def _mean_std(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    mu = sum(vals) / n
    return {"mean": mu, "std": math.sqrt(sum((x - mu) ** 2 for x in vals) / max(1, n - 1))}


def aggregate_cv_folds(fold_rows: list[dict]) -> dict[str, object]:
    if not fold_rows:
        raise ValueError("empty fold_rows")
    out: dict[str, object] = {"folds": fold_rows}
    for key in CV_METRICS:
        vals = [float(r[key]) for r in fold_rows if key in r and r[key] is not None]
        if vals:
            out[key] = _mean_std(vals)
    return out


def load_cv_summary(path: Path | str) -> dict[str, object]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())
