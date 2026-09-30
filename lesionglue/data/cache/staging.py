"""Per-patient preprocess staging until collated {split}_v8_native.pt is written.

Each {pid}.pt is a list of region graphs (possibly empty) so --resume is exact.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from lesionglue.data.features.layout import CACHE_TAG


def dir(processed_dir: Path, split: str) -> Path:
    return processed_dir / "staging" / f"{split}_{CACHE_TAG}"


def clear(staging: Path) -> None:
    if not staging.is_dir():
        return
    for p in staging.glob("*.pt"):
        p.unlink()


def path(staging: Path, pid: str) -> Path:
    return staging / f"{pid}.pt"


def has(staging: Path, pid: str) -> bool:
    return path(staging, pid).is_file()


def save(staging: Path, pid: str, graphs: list[HeteroData]) -> None:
    torch.save(graphs, path(staging, pid))


def load_all(staging: Path, pids: list[str]) -> list[HeteroData]:
    out: list[HeteroData] = []
    for pid in pids:
        p = path(staging, pid)
        if p.is_file():
            item = torch.load(p, weights_only=False)
            out.extend(item if isinstance(item, list) else [item])
    return out


def rm(staging: Path) -> None:
    if not staging.is_dir():
        return
    for p in staging.glob("*.pt"):
        p.unlink()
    staging.rmdir()


def todo(pids: list[str], staging: Path, resume: bool) -> list[str]:
    if not resume:
        return list(pids)
    return [p for p in pids if not has(staging, p)]
