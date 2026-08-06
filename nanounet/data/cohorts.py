"""Cohort-weighted case sampling: draw a source dataset, then a case uniformly inside it.

Dataset999 merges 17 source datasets of very unequal size (CECT 18%, MSD_Lung 1%), so a uniform
draw over cases makes the training mixture an accident of how much data each site happened to
contribute. Named prefixes take their stated probability; the remaining mass is spread over all
other cases in proportion to their counts, so an absent or empty `sampling.cohorts` block
reproduces the uniform draw exactly.

This composes with, and does not replace, the *_weights.json lesion-type weights: cohort weights
pick WHICH case, lesion weights pick WHERE inside it (patch_bbox.py). Two independent knobs.

Setup is O(#cases) once at construction -- there is no per-draw prefix scan, because this runs on
the dataloader hot path for every patch.
"""

from __future__ import annotations

import numpy as np

from nanounet.plan.splits import cohort_of


class CohortSampler:
    def __init__(self, keys: list[str], weights: dict[str, float]):
        groups: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(cohort_of(k), []).append(i)
        self.keys = keys
        self.names = sorted(groups)
        self.idx = [np.asarray(groups[n], dtype=np.int64) for n in self.names]

        named = {n: float(w) for n, w in weights.items()}
        unknown = sorted(set(named) - set(self.names))
        if unknown:
            raise ValueError(
                f"sampling.cohorts names {unknown} are not present in this key list.\n"
                f"Available cohorts: {', '.join(self.names)}\n"
                f"Fix: use a bare dataset prefix such as \"d013\" (no trailing underscore)"
            )
        total = sum(named.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"sampling.cohorts weights sum to {total:.4f}, which exceeds 1.0.\n"
                f"The remainder is what is left for every unnamed cohort.\n"
                f"Fix: lower the weights so they sum to at most 1.0"
            )
        rest = [n for n in self.names if n not in named]
        rest_cases = sum(len(self.idx[self.names.index(n)]) for n in rest)
        if rest and rest_cases == 0:
            raise ValueError("every cohort is named but the weights do not sum to 1.0")
        spare = max(0.0, 1.0 - total)
        p = np.empty(len(self.names), dtype=np.float64)
        for j, n in enumerate(self.names):
            if n in named:
                p[j] = named[n]
            else:
                p[j] = spare * len(self.idx[j]) / rest_cases if rest_cases else 0.0
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
