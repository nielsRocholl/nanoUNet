"""Cohort-weighted case sampling: draw a source dataset, then a case uniformly inside it.

Dataset999 merges 17 source datasets of very unequal size (CECT 18%, MSD_Lung 1%), so a uniform
draw over cases makes the training mixture an accident of how much data each site happened to
contribute. Weights come from cohorts.json, written into the preprocessed dataset dir by
nanounet_preprocess -- it is the single derived, always-correct source, so it is loaded by
default instead of being re-specified (and silently left incomplete) in the training config.

This composes with, and does not replace, the *_weights.json lesion-type weights: cohort weights
pick WHICH case, lesion weights pick WHERE inside it (patch_bbox.py). Two independent knobs.

Setup is O(#cases) once at construction -- there is no per-draw prefix scan, because this runs on
the dataloader hot path for every patch.
"""

from __future__ import annotations

import os

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from nanounet.plan.splits import cohort_of

COHORTS_FILENAME = "cohorts.json"


def load_cohort_weights(dataset_dir: str) -> dict[str, float]:
    """Read the {prefix: weight} map written by preprocess for `dataset_dir`."""
    path = os.path.join(dataset_dir, COHORTS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No {COHORTS_FILENAME} at {path}.\n"
            f"Expected the site-balanced cohort weights written by the preprocess step.\n"
            f"Fix: nanounet_preprocess -d <id> ...   (this now also writes {COHORTS_FILENAME})"
        )
    return {str(k): float(v) for k, v in load_json(path)["weights"].items()}


class CohortSampler:
    def __init__(self, keys: list[str], dataset_dir: str, override: dict[str, float] | None = None):
        groups: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(cohort_of(k), []).append(i)
        self.keys = keys
        self.names = sorted(groups)
        self.idx = [np.asarray(groups[n], dtype=np.int64) for n in self.names]

        if override:
            weights = {str(k): float(v) for k, v in override.items()}
            unknown = sorted(set(weights) - set(self.names))
            if unknown:
                raise ValueError(
                    f"sampling.cohorts names {unknown} are not present in this key list.\n"
                    f"Available cohorts: {', '.join(self.names)}\n"
                    f"Fix: use a bare dataset prefix such as \"d013\" (no trailing underscore)"
                )
            missing = sorted(set(self.names) - set(weights))
            if missing:
                raise ValueError(
                    f"sampling.cohorts overrides {sorted(weights)} but does not name {missing}, "
                    f"which are present in the dataset.\n"
                    f"An override must cover every cohort in the dataset -- there is no more "
                    f"silent leftover-mass fill.\n"
                    f"Fix: add {missing} to sampling.cohorts, or delete the `cohorts` block "
                    f"entirely to use the derived weights in {COHORTS_FILENAME}"
                )
        else:
            weights = load_cohort_weights(dataset_dir)
            unknown = sorted(set(weights) - set(self.names))
            if unknown:
                raise ValueError(
                    f"{COHORTS_FILENAME} names {unknown} which are not present in this key list.\n"
                    f"Available cohorts: {', '.join(self.names)}\n"
                    f"Fix: rebuild it with nanounet_preprocess -d <id> ..."
                )
            missing = sorted(set(self.names) - set(weights))
            if missing:
                raise ValueError(
                    f"{COHORTS_FILENAME} at {os.path.join(dataset_dir, COHORTS_FILENAME)} does not "
                    f"name {missing}, which are present in this key list.\n"
                    f"Fix: rebuild it with nanounet_preprocess -d <id> ..."
                )

        p = np.array([weights[n] for n in self.names], dtype=np.float64)
        s = p.sum()
        assert s > 0, "cohort weights collapsed to zero"
        self.cdf = np.cumsum(p / s)

    def draw(self, rng: np.random.Generator) -> str:
        g = int(np.searchsorted(self.cdf, rng.random(), side="right"))
        pool = self.idx[min(g, len(self.idx) - 1)]
        return self.keys[int(pool[rng.integers(len(pool))])]

    def realised(self) -> dict[str, float]:
        """Per-cohort draw probability, for the startup config table."""
        prev = 0.0
        out = {}
        for n, c in zip(self.names, self.cdf):
            out[n] = float(c - prev)
            prev = float(c)
        return out
