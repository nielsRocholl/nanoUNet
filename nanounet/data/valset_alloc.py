"""Patch-budget allocation for nanounet_build_valset: per-cohort totals, and the global
subset_clicked allocation that respects multi-lesion supply. Split out of valset_build.py (which
holds the per-patch scenario search) purely to keep both files under the 200-LOC ceiling -- there
is no cc3d, no I/O, and no randomness here, only arithmetic over case counts."""

from __future__ import annotations

import numpy as np


def allocate(cohort_val_counts: dict[str, int], n_patches: int, floor: int) -> dict[str, int]:
    """floor per cohort, remainder proportional to val-case count. Largest-remainder rounding so
    the totals add up exactly."""
    if floor * len(cohort_val_counts) > n_patches:
        raise ValueError(
            f"--floor {floor} x {len(cohort_val_counts)} cohorts = "
            f"{floor * len(cohort_val_counts)} > --n-patches {n_patches}.\n"
            f"Fix: lower --floor or raise --n-patches "
            f"(>= {floor * len(cohort_val_counts)})"
        )
    cohorts = sorted(cohort_val_counts)
    total_val = sum(cohort_val_counts[c] for c in cohorts)
    remainder_budget = n_patches - floor * len(cohorts)
    raw = {c: floor + remainder_budget * cohort_val_counts[c] / total_val for c in cohorts}
    base = {c: int(np.floor(raw[c])) for c in cohorts}
    left = n_patches - sum(base.values())
    order = sorted(cohorts, key=lambda c: raw[c] - base[c], reverse=True)
    for c in order[:left]:
        base[c] += 1
    return base


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Distribute `total` units across `weights` (need not sum to 1), largest-remainder rounding.
    A zero-weight key always gets 0: its remainder is 0, so it sorts last and is never topped up
    while any positive-remainder key remains."""
    s = sum(weights.values())
    if s <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    left = total - sum(base.values())
    order = sorted(weights, key=lambda k: raw[k] - base[k], reverse=True)
    for k in order[: max(left, 0)]:
        base[k] += 1
    return base


def split_counts(count: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder rounding of `count` across the keys of `shares` (which sum to 1)."""
    return _largest_remainder(count, shares)


def load_cohorts(pp: str, ds_name: str, cohort_of) -> tuple[dict[str, list[str]], dict[str, float]]:
    """val cases grouped by cohort, and cohort_weights = true (train+val) proportions (D7)."""
    from collections import Counter

    from batchgenerators.utilities.file_and_folder_operations import join, load_json

    splits = load_json(join(pp, ds_name, "splits_final.json"))
    train_ids, val_ids = splits[0]["train"], splits[0]["val"]
    all_ids = sorted(train_ids + val_ids)
    by_cohort: dict[str, list[str]] = {}
    for cid in val_ids:
        by_cohort.setdefault(cohort_of(cid), []).append(cid)
    totals = Counter(cohort_of(cid) for cid in all_ids)
    weights = {c: totals[c] / len(all_ids) for c in by_cohort}
    return by_cohort, weights


def multi_lesion_counts(case_dir: str, by_cohort: dict[str, list[str]], case_info, cache: dict) -> dict[str, int]:
    """Per-cohort count of val cases with >=2 lesion instances, read straight off the centroid
    sidecars (len(centroids_zyx)) -- no cc3d, no b2nd opens. `case_info` is valset_build.case_info,
    passed in rather than imported to avoid a cycle; it populates `cache` as a side effect so the
    later per-scenario fill loop does not re-read the same properties."""
    return {
        cohort: sum(1 for cid in ids if len(case_info(case_dir, cid, cache).cts_global) >= 2)
        for cohort, ids in by_cohort.items()
    }


def allocate_subset(
    cohort_totals: dict[str, int], multi_counts: dict[str, int], n_subset_total: int, cap_frac: float = 0.40
) -> dict[str, int]:
    """Global subset_clicked allocation, weighted by multi-lesion supply rather than cohort size:
    single-lesion cohorts (multi_counts[c] == 0, e.g. single-tumor datasets) get exactly 0, and no
    cohort exceeds `cap_frac` of its own patch budget, however much multi-lesion data it has."""
    cohorts = sorted(cohort_totals)
    caps = {c: int(cap_frac * cohort_totals[c]) for c in cohorts}
    weights = {c: float(multi_counts.get(c, 0)) for c in cohorts}
    subset = _largest_remainder(n_subset_total, weights)
    for c in cohorts:
        if weights[c] <= 0:
            subset[c] = 0
        subset[c] = min(subset[c], caps[c])
    shortfall = n_subset_total - sum(subset.values())
    while shortfall > 0:
        headroom = {c: caps[c] - subset[c] for c in cohorts if weights[c] > 0 and caps[c] - subset[c] > 0}
        if not headroom:
            break
        add = _largest_remainder(shortfall, {c: weights[c] for c in headroom})
        add = {c: min(a, headroom[c]) for c, a in add.items()}
        if sum(add.values()) == 0:
            break
        for c, a in add.items():
            subset[c] += a
        shortfall = n_subset_total - sum(subset.values())
    if shortfall > 0:
        n_multi = sum(1 for c in cohorts if weights[c] > 0)
        raise RuntimeError(
            f"subset_clicked target {n_subset_total} exceeds available capacity "
            f"{n_subset_total - shortfall} (only {n_multi}/{len(cohorts)} cohorts have multi-lesion "
            f"cases, each capped at {int(cap_frac * 100)}% of its patch budget).\n"
            f"Fix: lower subset_clicked in --mix, raise cap_frac (now {cap_frac}), or lower --floor. "
            f"Raising --n-patches does not help: the gap is cap_frac times the single-lesion-cohort budget."
        )
    return subset


def scenario_allocation(
    case_dir: str, by_cohort: dict[str, list[str]], per_cohort_total: dict[str, int], shares: dict[str, float],
    case_info, cache: dict,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Two-stage per-cohort scenario counts: subset_clicked globally by multi-lesion supply
    (allocate_subset), the rest of each cohort's budget by --mix renormalised over the other 3
    scenarios. Returns (per_cohort_counts, cohort_multi_counts)."""
    multi_counts = multi_lesion_counts(case_dir, by_cohort, case_info, cache)
    n_subset_total = round(sum(per_cohort_total.values()) * shares["subset_clicked"])
    subset_alloc = allocate_subset(per_cohort_total, multi_counts, n_subset_total)
    rest_raw = {k: v for k, v in shares.items() if k != "subset_clicked"}
    rest_shares = {k: v / sum(rest_raw.values()) for k, v in rest_raw.items()}
    per_cohort_counts = {}
    for cohort in sorted(per_cohort_total):
        rest = split_counts(per_cohort_total[cohort] - subset_alloc[cohort], rest_shares)
        per_cohort_counts[cohort] = {**rest, "subset_clicked": subset_alloc[cohort]}
    return per_cohort_counts, multi_counts
