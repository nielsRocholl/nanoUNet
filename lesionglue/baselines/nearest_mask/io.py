"""I/O utilities for the isolated nearest-mask baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import nibabel as nib
import numpy as np

from lesionglue.data.source.meta import load_split_json

if TYPE_CHECKING:
    from lesionglue.baselines.nearest_mask.baseline import PredictionRow

ROW_FIELDS = [
    "pid",
    "img_id_fu",
    "bl_lesion_id",
    "topology",
    "gt_fu_lesion_id",
    "pred_fu_lesion_id",
    "distance_mm",
    "correct",
    "status",
]


def load_split_ids(root: Path, split: str) -> dict[str, list[str]]:
    split_map = load_split_json(Path(root) / "data_split.json")
    if split == "all":
        return {name: list(split_map[name]) for name in ("train", "val", "test") if name in split_map}
    if split not in split_map:
        raise KeyError(f"split {split!r} not found in data_split.json")
    return {split: list(split_map[split])}


def load_fu_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    mask = np.asarray(img.dataobj, dtype=np.int64)
    affine = np.asarray(img.affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0).astype(np.float64)
    return np.ascontiguousarray(mask), spacing


def write_prediction_rows(path: Path, rows: Iterable["PredictionRow"]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def write_summary_csv(path: Path, summaries: Iterable[dict[str, object]]) -> None:
    summaries = list(summaries)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        path.write_text("")
        return
    fields: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
