"""Fixed DataLoader presets: worker count and prefetch per train/val.

Bucket ``s`` defaults to zero workers (Slurm cgroup-safe). Set NANOUNET_DL_KEEP_WORKERS=1
(or legacy NANOUNET_MAE_KEEP_WORKERS=1) to restore 2/1 workers for ``s``."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DataloaderBucket:
    nw_train: int
    nw_val: int
    prefetch_train: int
    prefetch_val: int


def dl_keep_workers() -> bool:
    for k in ("NANOUNET_DL_KEEP_WORKERS", "NANOUNET_MAE_KEEP_WORKERS"):
        if os.environ.get(k, "").strip() in ("1", "true", "yes"):
            return True
    return False


def _bucket_workers(name: Literal["s", "m", "l"]) -> DataloaderBucket:
    if name == "s":
        return DataloaderBucket(2, 1, 2, 2)
    elif name == "m":
        return DataloaderBucket(4, 2, 4, 2)
    elif name == "l":
        return DataloaderBucket(8, 4, 4, 2)
    raise ValueError(f"unknown dataloader bucket {name!r}")


def dataloader_bucket(name: Literal["s", "m", "l"]) -> DataloaderBucket:
    if name == "s" and not dl_keep_workers():
        return DataloaderBucket(0, 0, 0, 0)
    return _bucket_workers(name)


def mae_dataloader_bucket(name: Literal["s", "m", "l"]) -> DataloaderBucket:
    return dataloader_bucket(name)


def mae_keep_workers() -> bool:
    return dl_keep_workers()


def init_dataloader_ipc() -> None:
    if not dl_keep_workers():
        return
    import torch.multiprocessing as tmp

    if tmp.get_sharing_strategy() != "file_system":
        tmp.set_sharing_strategy("file_system")


def build_iter_dataloader(
    dataset,
    *,
    batch_size: int,
    bucket: DataloaderBucket,
    nw: int,
    prefetch: int,
    collate_fn: Callable,
    pin_memory: bool,
    worker_init_fn: Callable | None,
) -> DataLoader:
    pin = False if nw == 0 else pin_memory
    kw: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": nw,
        "pin_memory": pin,
        "collate_fn": collate_fn,
    }
    if nw:
        kw["persistent_workers"] = False
        kw["prefetch_factor"] = prefetch
        kw["worker_init_fn"] = worker_init_fn
    return DataLoader(dataset, **kw)
