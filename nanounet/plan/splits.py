"""K-fold splits; same file format as nnU-Net."""

from __future__ import annotations

import json
from typing import List

import numpy as np
from sklearn.model_selection import KFold


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


def fold_keys(splits: List[dict], fold: int) -> tuple[list[str], list[str]]:
    return splits[fold]["train"], splits[fold]["val"]
