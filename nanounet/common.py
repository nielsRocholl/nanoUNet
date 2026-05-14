"""Paths, logging, Rich CLI on stderr, rank-0 output. Sync NANOUNET_* env to nnUNet_*."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

_LOG = logging.getLogger("nanounet")
_CONSOLE = Console(stderr=True)

ANISO_THRESHOLD = 3
DEFAULT_NUM_PROCESSES = 8 if "nnUNet_def_n_proc" not in os.environ else int(os.environ["nnUNet_def_n_proc"])


def _rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def sync_nnunet_env() -> None:
    for a, b in (
        ("NANOUNET_RAW", "nnUNet_raw"),
        ("NANOUNET_PREPROCESSED", "nnUNet_preprocessed"),
        ("NANOUNET_RESULTS", "nnUNet_results"),
    ):
        v = os.environ.get(a) or os.environ.get(b)
        if v:
            os.environ[b] = v


def raw_dir() -> str:
    sync_nnunet_env()
    d = os.environ.get("nnUNet_raw")
    if not d:
        raise EnvironmentError("Set nnUNet_raw or NANOUNET_RAW")
    return d


def preprocessed_dir() -> str:
    sync_nnunet_env()
    d = os.environ.get("nnUNet_preprocessed")
    if not d:
        raise EnvironmentError("Set nnUNet_preprocessed or NANOUNET_PREPROCESSED")
    return d


def results_dir() -> str:
    sync_nnunet_env()
    d = os.environ.get("nnUNet_results")
    if not d:
        raise EnvironmentError("Set nnUNet_results or NANOUNET_RESULTS")
    return d


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
