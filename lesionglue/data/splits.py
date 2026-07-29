"""Deterministic patient-level k-fold map for CV (no BL/FU leakage across folds)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tracking.data.meta import load_split_json

CV_METRICS = ("val_match_score_ema", "val_match_score_raw", "val_match_score_peak", "val_acc_unchanged_split", "val_acc_disappeared", "val_acc_newly_appearing")

CV_SPLIT_SEED = 0

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 0

# Order matches MatcherModule's per-patient accumulator (tracking/train/module.py::validation_step).
_COUNT_KEYS = ("uc_ok", "uc_tot", "dis_ok", "dis_tot", "new_ok", "new_tot")
_SUB_WEIGHTS = (("uc", 0.5), ("dis", 0.25), ("new", 0.25))


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


def match_score_from_counts(counts: dict) -> float:
    """Weighted 0.5/0.25/0.25 match score from pooled per-patient counts.

    Must reproduce MatcherModule.on_validation_epoch_end exactly: sub-metrics with tot==0 are
    DROPPED from both numerator and denominator, not counted as zero.
    """
    num = den = 0.0
    for name, w in _SUB_WEIGHTS:
        tot = counts[f"{name}_tot"]
        if tot:
            num += w * counts[f"{name}_ok"] / tot
            den += w
    return num / den if den else 0.0


def _counts_matrix(per_patient: dict) -> tuple[list[str], np.ndarray]:
    pids = list(per_patient)
    mat = np.array([[per_patient[p][k] for k in _COUNT_KEYS] for p in pids], dtype=np.float64)
    return pids, mat


def _score_from_pooled(pooled: np.ndarray) -> np.ndarray:
    # Vectorized twin of match_score_from_counts, operating on the last axis of an (..., 6) array
    # of pooled counts so the whole bootstrap resample runs as one numpy call, not a python loop.
    num = np.zeros(pooled.shape[:-1])
    den = np.zeros(pooled.shape[:-1])
    for i, (_, w) in enumerate(_SUB_WEIGHTS):
        ok, tot = pooled[..., 2 * i], pooled[..., 2 * i + 1]
        active = tot > 0
        num = num + np.where(active, w * ok / np.where(active, tot, 1.0), 0.0)
        den = den + np.where(active, w, 0.0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def bootstrap_match_score(per_patient: dict, b: int = BOOTSTRAP_B) -> tuple[float, float, float]:
    """Point estimate + (2.5, 97.5) percentile CI, resampling PATIENTS not lesions.

    Lesions within a patient share anatomy and registration error, so a lesion-level bootstrap
    would understate the interval substantially.
    """
    pids, mat = _counts_matrix(per_patient)
    n = len(pids)
    point = match_score_from_counts(dict(zip(_COUNT_KEYS, mat.sum(axis=0))))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(b, n))
    pooled = mat[idx].sum(axis=1)
    scores = _score_from_pooled(pooled)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_delta_ci(a: dict, b_: dict, b: int = BOOTSTRAP_B) -> tuple[float, float, float]:
    """Bootstrap CI of the PAIRED delta (config a minus config b_) on the POOLED score.

    Pairing is carried by reusing one set of resample indices for both configs, which cancels the
    fold-identity variance that dominates the marginal bands (measured spread of
    val_acc_unchanged_split across folds: 0.833-0.957).

    The estimand is the pooled, lesion-weighted score -- the same quantity val_match_score reports.
    An unweighted mean over patients would be a different number here (lesion counts run 1..86) and
    would let a patient with no labelled events contribute a spurious 0.0.
    """
    assert set(a) == set(b_), "paired delta requires the same patient set for both configs"
    pids = sorted(a)  # sorted so both matrices index the same patient per row
    ma = np.array([[a[p][k] for k in _COUNT_KEYS] for p in pids], dtype=np.float64)
    mb = np.array([[b_[p][k] for k in _COUNT_KEYS] for p in pids], dtype=np.float64)
    point = match_score_from_counts(dict(zip(_COUNT_KEYS, ma.sum(axis=0)))) - match_score_from_counts(
        dict(zip(_COUNT_KEYS, mb.sum(axis=0)))
    )
    n = len(pids)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(b, n))
    deltas = _score_from_pooled(ma[idx].sum(axis=1)) - _score_from_pooled(mb[idx].sum(axis=1))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def load_cv_summary(path: Path | str) -> dict[str, object]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())
