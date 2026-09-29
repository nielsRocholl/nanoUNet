"""Patient-level bootstrap CI for the weighted match score."""

from __future__ import annotations

import numpy as np

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 0
_COUNT_KEYS = ("uc_ok", "uc_tot", "dis_ok", "dis_tot", "new_ok", "new_tot")
_SUB_WEIGHTS = (("uc", 0.5), ("dis", 0.25), ("new", 0.25))


def match_score_from_counts(counts: dict) -> float:
    """Weighted 0.5/0.25/0.25 match score. Sub-metrics with tot==0 are dropped, not zeroed."""
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
    num = np.zeros(pooled.shape[:-1])
    den = np.zeros(pooled.shape[:-1])
    for i, (_, w) in enumerate(_SUB_WEIGHTS):
        ok, tot = pooled[..., 2 * i], pooled[..., 2 * i + 1]
        active = tot > 0
        num = num + np.where(active, w * ok / np.where(active, tot, 1.0), 0.0)
        den = den + np.where(active, w, 0.0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def bootstrap_match_score(per_patient: dict, b: int = BOOTSTRAP_B) -> tuple[float, float, float]:
    """Point estimate + (2.5, 97.5) percentile CI, resampling patients not lesions."""
    pids, mat = _counts_matrix(per_patient)
    n = len(pids)
    point = match_score_from_counts(dict(zip(_COUNT_KEYS, mat.sum(axis=0))))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(b, n))
    scores = _score_from_pooled(mat[idx].sum(axis=1))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_delta_ci(a: dict, b_: dict, b: int = BOOTSTRAP_B) -> tuple[float, float, float]:
    """Bootstrap CI of paired delta (a minus b_) on the pooled score."""
    assert set(a) == set(b_), "paired delta requires the same patient set for both configs"
    pids = sorted(a)
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
