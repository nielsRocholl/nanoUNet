"""Nearest-mask matching baseline.

For every eligible baseline lesion, query the closest foreground voxel in the
matching follow-up instance mask. This intentionally does not enforce one-to-one
matching and only emits -1 when the follow-up mask has no candidate instances.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lesionglue.baselines.nearest_mask.io import load_fu_mask
from lesionglue.data.meta import LesionRow, V2Paths, parse_meta_csv

BASELINE_TOPOLOGIES = frozenset({"UNCHANGED", "DISAPPEARED", "MERGED", "SPLIT"})
LINKABLE_TOPOLOGIES = frozenset({"UNCHANGED", "MERGED", "SPLIT"})


@dataclass(frozen=True)
class NearestMaskMatch:
    pred_fu_lesion_id: int
    distance_mm: float | None
    status: str


@dataclass(frozen=True)
class PredictionRow:
    pid: str
    img_id_fu: int
    bl_lesion_id: int
    topology: str
    gt_fu_lesion_id: int | None
    pred_fu_lesion_id: int
    distance_mm: float | None
    correct: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "img_id_fu": self.img_id_fu,
            "bl_lesion_id": self.bl_lesion_id,
            "topology": self.topology,
            "gt_fu_lesion_id": "" if self.gt_fu_lesion_id is None else self.gt_fu_lesion_id,
            "pred_fu_lesion_id": self.pred_fu_lesion_id,
            "distance_mm": "" if self.distance_mm is None else f"{self.distance_mm:.8g}",
            "correct": int(self.correct),
            "status": self.status,
        }


@dataclass(frozen=True)
class PatientBaselineResult:
    pid: str
    rows: list[LesionRow]
    predictions: list[PredictionRow]
    skipped_missing_cog: int


class NearestMaskIndex:
    def __init__(self, mask: np.ndarray, spacing: np.ndarray):
        self.mask = np.asarray(mask, dtype=np.int64)
        self.spacing = np.asarray(spacing, dtype=np.float64)
        if self.spacing.shape != (3,):
            raise ValueError(f"spacing must have shape (3,), got {self.spacing.shape}")
        if np.any(self.spacing <= 0):
            raise ValueError(f"spacing must be positive, got {self.spacing.tolist()}")

        foreground = np.argwhere(self.mask != 0)
        self._labels = self.mask[tuple(foreground.T)].astype(np.int64, copy=False) if foreground.size else np.empty(0)
        if foreground.size:
            points_mm = (foreground.astype(np.float64) + 0.5) * self.spacing
            self._tree: cKDTree | None = cKDTree(points_mm)
        else:
            self._tree = None

    def query(self, point_vox: tuple[float, float, float] | np.ndarray) -> NearestMaskMatch:
        point = np.asarray(point_vox, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            return NearestMaskMatch(-1, None, "invalid_point")

        inside_idx = np.floor(point).astype(np.int64)
        if np.all((0 <= inside_idx) & (inside_idx < np.asarray(self.mask.shape))):
            label = int(self.mask[tuple(inside_idx)])
            if label != 0:
                return NearestMaskMatch(label, 0.0, "inside_mask")

        if self._tree is None:
            return NearestMaskMatch(-1, None, "no_fu_candidate")

        distance_mm, nearest = self._tree.query(point * self.spacing, k=1)
        return NearestMaskMatch(int(self._labels[int(nearest)]), float(distance_mm), "nearest_mask")


def expected_fu_lesion_id(row: LesionRow) -> int | None:
    if row.topology in ("UNCHANGED", "SPLIT"):
        return row.lesion_id
    if row.topology == "MERGED":
        return row.merged_into
    if row.topology == "DISAPPEARED":
        return -1
    return None


def dominant_fu_id(rows: list[LesionRow]) -> int | None:
    counts = Counter(r.img_id_fu for r in rows)
    if not counts:
        return None
    return int(counts.most_common(1)[0][0])


def graph_compatible_rows(rows: list[LesionRow]) -> list[LesionRow]:
    dom = dominant_fu_id(rows)
    if dom is None:
        return []
    return [r for r in rows if r.img_id_fu == dom]


def eligible_baseline_rows(rows: list[LesionRow]) -> list[LesionRow]:
    return [r for r in rows if r.topology in BASELINE_TOPOLOGIES and r.cog_propagated is not None]


def run_patient(root: Path, pid: str, graph_compatible: bool = False) -> PatientBaselineResult:
    root = Path(root)
    paths = V2Paths(root, pid)
    rows = parse_meta_csv(paths.meta)
    if graph_compatible:
        rows = graph_compatible_rows(rows)

    eligible = eligible_baseline_rows(rows)
    skipped_missing_cog = sum(
        1 for r in rows if r.topology in BASELINE_TOPOLOGIES and r.cog_propagated is None
    )
    index_cache: dict[int, NearestMaskIndex | None] = {}
    predictions: list[PredictionRow] = []

    for row in eligible:
        if row.img_id_fu not in index_cache:
            mask_path = paths.fu_mask(row.img_id_fu)
            if mask_path.exists():
                mask, spacing = load_fu_mask(mask_path)
                index_cache[row.img_id_fu] = NearestMaskIndex(mask, spacing)
            else:
                index_cache[row.img_id_fu] = None

        index = index_cache[row.img_id_fu]
        if index is None:
            match = NearestMaskMatch(-1, None, "missing_fu_mask")
        else:
            assert row.cog_propagated is not None
            match = index.query(row.cog_propagated)

        gt = expected_fu_lesion_id(row)
        correct = gt is not None and int(match.pred_fu_lesion_id) == int(gt)
        predictions.append(
            PredictionRow(
                pid=pid,
                img_id_fu=int(row.img_id_fu),
                bl_lesion_id=int(row.lesion_id),
                topology=row.topology,
                gt_fu_lesion_id=gt,
                pred_fu_lesion_id=int(match.pred_fu_lesion_id),
                distance_mm=match.distance_mm,
                correct=bool(correct),
                status=match.status if gt is not None else f"{match.status};missing_gt",
            )
        )

    return PatientBaselineResult(
        pid=pid,
        rows=rows,
        predictions=predictions,
        skipped_missing_cog=skipped_missing_cog,
    )
