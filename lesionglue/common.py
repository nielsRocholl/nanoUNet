"""Dataset/cache paths, Rich rank-0 UI, seed, JSON I/O.

Deployed matcher: v7_complete last.ckpt, EMA, hungarian, dust_tau=0.125.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DATASET_ROOT = Path("/nnunet_data/Longitudinal-CT")
CACHE_ROOT = Path("/nnunet_data/lesion_tracking/cache")
REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = REPO_ROOT / "configs" / "split.json"
HOLDOUT_CSV = DATASET_ROOT / "test_patients.csv"
DEPLOYED_CKPT = Path("/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt")
DEPLOYED_DUST_TAU = 0.125

_CONSOLE = Console(stderr=True)

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


def _rank0() -> bool:
    return mp.current_process().name == "MainProcess"


def cprint(msg: object, **kw: Any) -> None:
    if _rank0():
        _CONSOLE.print(msg, **kw)


print0 = cprint


def nano_header(title: str, color: str = "cyan") -> None:
    if _rank0():
        _CONSOLE.print(Panel(f"[bold {color}]{title}[/bold {color}]", border_style=color))


def config_table(rows: list[tuple[str, object, str]], title: str = "config") -> None:
    if not _rank0():
        return
    t = Table(title=title, box=None, padding=(0, 2))
    t.add_column("argument", style="cyan")
    t.add_column("value")
    t.add_column("source", style="dim")
    for name, value, source in rows:
        t.add_row(str(name), str(value), source)
    _CONSOLE.print(t)


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def require_ckpt(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"No checkpoint at {p}.\n"
            f"Expected the deployed matcher at {DEPLOYED_CKPT} "
            f"(EMA, hungarian, dust_tau={DEPLOYED_DUST_TAU}).\n"
            f"Fix: --ckpt {DEPLOYED_CKPT}"
        )
    return p


def dump_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def eval_device(preference: str) -> torch.device:
    p = preference.lower().strip()
    if p == "cpu":
        return torch.device("cpu")
    if p == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device cuda but CUDA not available.\n"
                f"Expected a visible GPU.\n"
                f"Fix: pass --device cpu"
            )
        return torch.device("cuda")
    if p == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                f"--device mps but MPS not available.\n"
                f"Expected Apple Silicon MPS.\n"
                f"Fix: pass --device cpu"
            )
        return torch.device("mps")
    if p == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(
        f"unknown device preference: {preference!r}.\n"
        f"Expected cuda, cpu, mps, or auto.\n"
        f"Fix: --device cuda"
    )
