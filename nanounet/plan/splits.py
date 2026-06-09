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
    return splits[fold]["train"], splits[fold]["val"]
