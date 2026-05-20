"""Fixed DataLoader presets: worker count and prefetch per train/val.

Slurm RAM is safer with ``s``; desktop defaults often ``m``. MAE defaults to no workers
to avoid DataLoader IPC shmem growth; set NANOUNET_MAE_KEEP_WORKERS=1 to use s/m/l."""
from __future__ import annotations

import os
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


_MAE_ZERO = DataloaderBucket(0, 0, 0, 0)


def mae_keep_workers() -> bool:
    return os.environ.get("NANOUNET_MAE_KEEP_WORKERS", "").strip() in ("1", "true", "yes")


def mae_dataloader_bucket(name: Literal["s", "m", "l"]) -> DataloaderBucket:
    if mae_keep_workers():
        return dataloader_bucket(name)
    return _MAE_ZERO
