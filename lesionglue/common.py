"""Cross-cutting helpers: dataset/cache paths, constants, print0, seed, JSON I/O."""

from pathlib import Path
import json
import multiprocessing as mp
import random

import numpy as np
import torch

DATASET_ROOT = Path("/nnunet_data/unprocessed-universal-lesion-segmentation/")
CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache" / "graphs"

LESION_TYPES = (
    "Adrenals",
    "CNS",
    "Heart",
    "Kidney",
    "Liver",
    "Lung",
    "Lymph node",
    "Others",
    "Skeleton",
    "Soft tissue / Skin",
    "Spleen",
    "unclear",
)

PROP_SIGMA = (2.75, 5.19, 5.40)
PROP_MAX_VOX = 34.0


def print0(msg: str) -> None:
    # Pool workers share stdout with Rich Progress on the parent — suppress noise / torn lines.
    if mp.current_process().name != "MainProcess":
        return
    print(msg)


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def dump_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
