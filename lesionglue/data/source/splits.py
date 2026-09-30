"""Deterministic patient-level k-fold map for CV (no BL/FU leakage across folds)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from lesionglue.common import HOLDOUT_CSV, SPLIT_PATH
from lesionglue.data.source.meta import load_split_json

CV_METRICS = ("val_match_score_ema", "val_match_score_raw", "val_match_score_peak", "val_acc_unchanged_split", "val_acc_disappeared", "val_acc_newly_appearing")

CV_SPLIT_SEED = 0


def fold_map(pids: list[str], n_folds: int, seed: int = CV_SPLIT_SEED) -> dict[str, int]:
    # Patient id is the fold unit, so a patient's BL/FU graph stays in one fold.
    # Sort first to make the map invariant to caller order; seeded round-robin keeps folds size-balanced.
    assert n_folds >= 2
    ids = sorted(set(map(str, pids)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[int(i)]: r % n_folds for r, i in enumerate(perm)}


def load_holdout(csv_path: Path | str = HOLDOUT_CSV) -> list[str]:
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"No holdout CSV at '{p}'.\n"
            f"Expected a CSV with a 'patient' column of id prefixes.\n"
            f"Fix: --holdout /nnunet_data/Longitudinal-CT/test_patients.csv"
        )
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(
            f"Empty holdout CSV '{p}'.\nExpected header plus one patient id per row.\n"
            f"Fix: --holdout /nnunet_data/Longitudinal-CT/test_patients.csv"
        )
    col = "patient" if "patient" in rows[0] else next(iter(rows[0]))
    out = [r[col].strip() for r in rows if r[col].strip()]
    if not out:
        raise FileNotFoundError(
            f"No patient ids in '{p}' (column '{col}').\n"
            f"Expected non-empty 'patient' cells.\n"
            f"Fix: --holdout /nnunet_data/Longitudinal-CT/test_patients.csv"
        )
    return out


def load_tracking_split(path: Path | str | None = None) -> dict:
    p = Path(path) if path else SPLIT_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"No tracking split at {p}.\n"
            f"Expected output of: python3 lesionglue/cli/split.py\n"
            f"Fix: python3 lesionglue/cli/split.py --root /nnunet_data/Longitudinal-CT"
            f" --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out lesionglue/configs/split.json"
        )
    return load_split_json(p)


def build_split(
    dataset_root: Path | str,
    holdout_csv: Path | str,
    n_folds: int = 5,
    seed: int = 0,
    val_fold: int = 0,
) -> dict:
    official = load_split_json(Path(dataset_root) / "data_split.json")
    holdout = set(load_holdout(holdout_csv))
    train240 = [str(x) for x in official["train"]]
    leak = set(train240) & holdout
    if leak:
        raise SystemExit(
            f"{len(leak)} holdout ids in official train: {sorted(leak)[:8]}...\n"
            "Expected the holdout CSV to list only patients from the official val and test splits.\n"
            f"Fix: remove those ids from {holdout_csv}, or from the train list of {dataset_root}/data_split.json"
        )
    extra = (set(map(str, official["val"])) | set(map(str, official["test"]))) - holdout
    if extra:
        raise SystemExit(
            f"official val/test id not in holdout csv: {sorted(extra)}\n"
            "Expected the holdout CSV to contain every official val and test patient.\n"
            f"Fix: add {sorted(extra)} to {holdout_csv}, or remove them from val/test in {dataset_root}/data_split.json"
        )
    missing = holdout - set(map(str, official["val"])) - set(map(str, official["test"]))
    if missing:
        raise SystemExit(
            f"holdout id not in official val∪test: {sorted(missing)}\n"
            "Expected every holdout CSV patient to be in the official val or test split.\n"
            f"Fix: remove {sorted(missing)} from {holdout_csv}, or add them to val/test in {dataset_root}/data_split.json"
        )
    fm = fold_map(train240, n_folds, seed)
    val = sorted(p for p, f in fm.items() if f == val_fold)
    train = sorted(p for p, f in fm.items() if f != val_fold)
    test = sorted(holdout)
    assert not (set(train) | set(val)) & set(test)
    return {"train": train, "val": val, "test": test, "n_folds": n_folds, "seed": seed, "val_fold": val_fold}


def pool_patient_ids(dataset_root: Path | str) -> list[str]:
    del dataset_root
    sp = load_tracking_split()
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
        # nanochat-style: allow E1 (internal invariant: cli/cv.py asserts start_fold < end before it calls this, so rows exist)
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
        raise FileNotFoundError(
            f"{p}\n"
            "Expected the cv_summary.json written by lesionglue_cv into its --out directory.\n"
            f"Fix: lesionglue_cv --config lesionglue/configs/base.json --out {p.parent}"
        )
    return json.loads(p.read_text())
