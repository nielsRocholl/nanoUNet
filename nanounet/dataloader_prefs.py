"""Fixed DataLoader presets: worker count and prefetch per train/val.

Slurm RAM is safer with ``s``; desktop defaults often ``m``. No CPU inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DataloaderBucket:
    nw_train: int
    nw_val: int
    prefetch_train: int
    prefetch_val: int


def dataloader_bucket(name: Literal["s", "m", "l"]) -> DataloaderBucket:
    if name == "s":
        return DataloaderBucket(2, 1, 2, 2)
    elif name == "m":
        return DataloaderBucket(4, 2, 4, 2)
    elif name == "l":
        return DataloaderBucket(8, 4, 4, 2)
    raise ValueError(f"unknown dataloader bucket {name!r}")
