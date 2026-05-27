"""Per-patient preprocess staging until collated {split}_v5_{feat}.pt is written.

staging/{split}_v5_{feat}/{pid}.pt survives interrupts; --resume skips files already there."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from tracking.data.features import FeatConfig, cache_tag


def dir(processed_dir: Path, split: str, feat: FeatConfig) -> Path:
    return processed_dir / "staging" / f"{split}_{cache_tag(feat)}"


def clear(staging: Path) -> None:
    if not staging.is_dir():
        return
    for p in staging.glob("*.pt"):
        p.unlink()


def path(staging: Path, pid: str) -> Path:
    return staging / f"{pid}.pt"


def has(staging: Path, pid: str) -> bool:
    return path(staging, pid).is_file()


def save(staging: Path, pid: str, g: HeteroData) -> None:
    torch.save(g, path(staging, pid))


def load_all(staging: Path, pids: list[str]) -> list[HeteroData]:
    out: list[HeteroData] = []
    for pid in pids:
        p = path(staging, pid)
        if p.is_file():
            out.append(torch.load(p, weights_only=False))
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
