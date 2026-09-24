"""Import shim for every harness subprocess: make `nanounet` resolve to EQUIV_SRC (a git worktree
of the base SHA, or the working tree) instead of the editable install, and seed/pin everything.

The editable install registers a sys.meta_path finder that maps `nanounet` to the repo checkout
and wins over sys.path, so it is removed here. Spawned DataLoader workers re-run the parent's
main script as __mp_main__, and that script imports this module first, so workers resolve the
same source tree."""

from __future__ import annotations

import os
import sys

SRC = os.environ["EQUIV_SRC"]
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in repr(f).lower()]
sys.path_hooks[:] = [h for h in sys.path_hooks if "editable" not in repr(h).lower()]
sys.path_importer_cache.clear()
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def seed_all(seed: int = 0) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pin() -> None:
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def check_src() -> None:
    import nanounet

    got = os.path.realpath(os.path.dirname(os.path.dirname(nanounet.__file__)))
    assert got == os.path.realpath(SRC), f"nanounet imported from {got}, expected {SRC}"


def mod(name: str):
    """Import `name`, following renames.json at HEAD (e.g. dice_helpers -> dice_metrics)."""
    import importlib
    import json

    here = os.path.dirname(os.path.abspath(__file__))
    ren = json.load(open(os.path.join(here, "renames.json"), encoding="utf-8"))["modules"]
    for cand in (ren.get(name), name):
        if cand is None:
            continue
        try:
            return importlib.import_module(cand)
        except ModuleNotFoundError as e:
            if e.name != cand:
                raise
    raise ModuleNotFoundError(name)


def _plain_tensor_ipc() -> None:
    """macOS only, harness only: DataLoader workers hand batches back through torch's
    `file_system` shared memory, which needs torch_shm_manager; forking that manager from the
    worker's (threaded) queue-feeder intermittently deadlocks here ("no response from
    torch_shm_manager"). Send tensors BY VALUE (numpy) instead. Transport
    only: the received batch is bit-identical. Linux/production IPC is untouched."""
    import platform

    # applied in every harness process (parent_process() is still None while a spawned child
    # imports __mp_main__); only DataLoader workers ever pickle tensors through ForkingPickler
    if platform.system() == "Darwin":
        import torch
        import torch.multiprocessing  # noqa: F401  (registers torch's reducers first)
        from multiprocessing.reduction import ForkingPickler

        ForkingPickler.register(torch.Tensor, lambda t: (torch.from_numpy, (t.detach().numpy().copy(),)))


_plain_tensor_ipc()


if os.environ.get("EQUIV_HANG_DUMP"):
    import faulthandler

    faulthandler.dump_traceback_later(int(os.environ["EQUIV_HANG_DUMP"]), exit=True)
