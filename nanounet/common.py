"""NANOUNET_* path env, Rich rank-0 UI (`print0`, headers, progress), logging.

`quiet_lightning_runtime`: call once before importing pytorch_lightning — warning filters,
rank_zero_info shim (litlogger noise), CUDA matmul precision high.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

_LOG = logging.getLogger("nanounet")
_CONSOLE = Console(stderr=True)

_LIGHTNING_QUIET = False

ANISO_THRESHOLD = 3
DEFAULT_NUM_PROCESSES = 8 if "NANOUNET_DEF_N_PROC" not in os.environ else int(os.environ["NANOUNET_DEF_N_PROC"])
_REPO_ROOT = Path(__file__).resolve().parent.parent


def quiet_lightning_runtime() -> None:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*pin_memory.*not supported on MPS.*",
        category=UserWarning,
        module=r"torch.utils.data.dataloader",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Precision 16-mixed is not supported by the model summary.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", message=r".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=r".*anonymous setting has no effect.*", category=UserWarning)
    warnings.filterwarnings(
        "ignore",
        message=r".*IterableDataset.*__len__.*multi-process data loading.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*DataLoader will create.*worker processes in total.*",
        category=UserWarning,
        module=r"torch.utils.data.dataloader",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*set_float32_matmul_precision.*",
        category=UserWarning,
    )
    global _LIGHTNING_QUIET
    if _LIGHTNING_QUIET:
        return
    import pytorch_lightning.utilities.rank_zero as _rz

    _orig = _rz.rank_zero_info

    def _no_litlogger_tip(*a: object, **k: object) -> None:
        if a and "litlogger" in str(a[0]).lower():
            return
        _orig(*a, **k)

    _rz.rank_zero_info = _no_litlogger_tip
    _LIGHTNING_QUIET = True

    import torch

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def resolve_user_config_path(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(path_str)
        return str(p.resolve())
    for base in (Path.cwd(), _REPO_ROOT):
        cand = (base / p).resolve()
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(path_str)


def _rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _env_path(name: str) -> str:
    d = os.environ.get(name)
    if not d:
        raise EnvironmentError(f"Set {name}")
    return d


def raw_dir() -> str:
    return _env_path("NANOUNET_RAW")


def preprocessed_dir() -> str:
    return _env_path("NANOUNET_PREPROCESSED")


def results_dir() -> str:
    return _env_path("NANOUNET_RESULTS")


def cprint(msg: str, **kw: Any) -> None:
    if _rank0():
        _CONSOLE.print(msg, **kw)


print0 = cprint


def nano_rule() -> None:
    if _rank0():
        _CONSOLE.print(Rule(style="dim"))


def nano_header(title: str, color: str = "cyan") -> None:
    if _rank0():
        _CONSOLE.print(Panel(f"[bold {color}]{title}[/bold {color}]", border_style=color))


@contextmanager
def nano_progress(total: int, desc: str) -> Iterator[Callable[[int], None]]:
    if not _rank0():
        yield lambda n=1: None
        return
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=_CONSOLE,
        transient=True,
    ) as prog:
        tid = prog.add_task(desc, total=total)

        def advance(n: int = 1) -> None:
            prog.advance(tid, n)

        yield advance


def setup_logging() -> None:
    if not _LOG.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        _LOG.addHandler(h)
        _LOG.setLevel(logging.INFO)
