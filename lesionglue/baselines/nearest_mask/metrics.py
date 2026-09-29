"""Metrics for the nearest-mask baseline."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

import numpy as np

from lesionglue.baselines.nearest_mask.baseline import LINKABLE_TOPOLOGIES, PredictionRow
from lesionglue.data.source.meta import LesionRow


def _safe_div(num: int, den: int) -> float | None:
    return None if den == 0 else float(num) / float(den)


def _distance_stats(prefix: str, distances: list[float]) -> dict[str, object]:
    if not distances:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean_mm": None,
            f"{prefix}_median_mm": None,
            f"{prefix}_p90_mm": None,
        }
    arr = np.asarray(distances, dtype=np.float64)
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean_mm": float(mean(arr)),
        f"{prefix}_median_mm": float(median(arr)),
        f"{prefix}_p90_mm": float(np.percentile(arr, 90)),
    }


def summarize(
    split: str,
    predictions: list[PredictionRow],
    rows_by_pid: dict[str, list[LesionRow]],
    skipped_missing_cog: int = 0,
) -> dict[str, object]:
    linkable = [p for p in predictions if p.topology in LINKABLE_TOPOLOGIES and p.gt_fu_lesion_id is not None]
    disappeared = [p for p in predictions if p.topology == "DISAPPEARED"]

    linkable_ok = sum(int(p.correct) for p in linkable)
    disappeared_ok = sum(int(p.correct) for p in disappeared)
    all_ok = sum(int(p.correct) for p in predictions)

    claims: dict[tuple[str, int], set[int]] = defaultdict(set)
    for pred in predictions:
        if pred.pred_fu_lesion_id >= 0:
            claims[(pred.pid, pred.img_id_fu)].add(pred.pred_fu_lesion_id)

    new_total = 0
    new_ok = 0
    for pid, rows in rows_by_pid.items():
        for row in rows:
            if row.topology != "NEWLYAPPEARING" or row.cog_fu is None:
                continue
            new_total += 1
            new_ok += int(row.lesion_id not in claims[(pid, int(row.img_id_fu))])

    weighted_parts: list[tuple[float, float]] = []
    linkable_acc = _safe_div(linkable_ok, len(linkable))
    disappeared_acc = _safe_div(disappeared_ok, len(disappeared))
    newly_appearing_acc = _safe_div(new_ok, new_total)
    if linkable_acc is not None:
        weighted_parts.append((0.5, linkable_acc))
    if disappeared_acc is not None:
        weighted_parts.append((0.25, disappeared_acc))
    if newly_appearing_acc is not None:
        weighted_parts.append((0.25, newly_appearing_acc))
    weight_sum = sum(w for w, _ in weighted_parts)
    match_score = None if weight_sum == 0 else sum(w * acc for w, acc in weighted_parts) / weight_sum

    correct_dist = [float(p.distance_mm) for p in predictions if p.correct and p.distance_mm is not None]
    incorrect_dist = [float(p.distance_mm) for p in predictions if not p.correct and p.distance_mm is not None]

    summary: dict[str, object] = {
        "split": split,
        "patients": len(rows_by_pid),
        "prediction_rows": len(predictions),
        "skipped_missing_cog": int(skipped_missing_cog),
        "linkable_total": len(linkable),
        "linkable_correct": linkable_ok,
        "linkable_acc": linkable_acc,
        "disappeared_total": len(disappeared),
        "disappeared_correct": disappeared_ok,
        "disappeared_acc": disappeared_acc,
        "newly_appearing_total": new_total,
        "newly_appearing_correct": new_ok,
        "newly_appearing_acc": newly_appearing_acc,
        "row_acc_all_bl": _safe_div(all_ok, len(predictions)),
        "match_score": match_score,
    }
    summary.update(_distance_stats("distance_correct", correct_dist))
    summary.update(_distance_stats("distance_incorrect", incorrect_dist))
    return summary
