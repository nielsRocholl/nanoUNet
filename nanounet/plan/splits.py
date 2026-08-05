"""K-fold splits; same file format as nnU-Net."""

from __future__ import annotations

import json
from typing import List

import numpy as np
from sklearn.model_selection import KFold

ALL_FOLD = "all"


def parse_fold(v: str) -> int | str:
    return ALL_FOLD if v == ALL_FOLD else int(v)


def fold_seed(fold: int | str) -> int:
    return 0 if fold == ALL_FOLD else fold


def make_splits(identifiers: List[str], n_splits: int = 5, seed: int = 12345) -> List[dict]:
    ids = sorted(identifiers)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out: List[dict] = []
    for tr, va in kf.split(ids):
        out.append({"train": [ids[i] for i in tr], "val": [ids[i] for i in va]})
    return out


def load_or_create_splits(path: str, tr_keys: List[str], n_splits: int, seed: int) -> List[dict]:
    import os

    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    sp = make_splits(tr_keys, n_splits, seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sp, f)
    return sp


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


def make_balanced_split(identifiers: List[str], val_frac: float, seed: int) -> List[dict]:
    """One train/val split with `val_frac` applied WITHIN each source dataset.

    The plain KFold above is dataset-blind: on the 17-cohort merged pool it drifted to 13-25% val
    per cohort, so small cohorts got val sets too small to plot. Returns a ONE-element list so it
    stays format-compatible with splits_final.json; fold 0 is the only valid fold."""
    assert 0.0 < val_frac < 1.0, val_frac
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for cid in sorted(identifiers):
        groups.setdefault(cohort_of(cid), []).append(cid)
    train: list[str] = []
    val: list[str] = []
    for _, ids in sorted(groups.items()):
        perm = rng.permutation(len(ids))
        n_val = int(round(len(ids) * val_frac))
        n_val = min(max(n_val, 1), len(ids) - 1)  # every cohort appears in BOTH sides
        val += [ids[i] for i in perm[:n_val]]
        train += [ids[i] for i in perm[n_val:]]
    return [{"train": sorted(train), "val": sorted(val)}]
