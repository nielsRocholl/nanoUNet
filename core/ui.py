"""Shared terminal UI for every project: one stderr rich Console (U1), headers, config tables, progress.

Projects import these instead of owning a Console, so output from chained steps (segtrack runs
nanounet then lesionglue) interleaves on one stream. Nothing here knows about a project (R21).
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

_CONSOLE = Console(stderr=True)


def cprint(msg: str, **kw: Any) -> None:
    _CONSOLE.print(msg, **kw)


def nano_rule() -> None:
    _CONSOLE.print(Rule(style="dim"))


def nano_header(title: str, color: str = "cyan") -> None:
    _CONSOLE.print(Panel(f"[bold {color}]{title}[/bold {color}]", border_style=color))


def nano_banner(title: str, subtitle: str, color: str = "cyan") -> None:
    body = Align.center(f"[bold {color}]{title}[/bold {color}]\n[dim]{subtitle}[/dim]")
    _CONSOLE.print(Panel(body, border_style=color, padding=(1, 4)))


def console() -> Console:
    return _CONSOLE


def config_table(rows: list[tuple[str, Any, str]], title: str = "config") -> None:
    """Render resolved config as a rich Table: (argument, value, source: cli/config/default)."""
    from rich.table import Table

    t = Table(title=title, box=None, padding=(0, 2))
    t.add_column("argument", style="cyan")
    t.add_column("value")
    t.add_column("source", style="dim")
    for name, value, source in rows:
        t.add_row(str(name), str(value), source)
    _CONSOLE.print(t)


def arg_rows(ap: ArgumentParser, args: Namespace) -> list[tuple[str, Any, str]]:
    """config_table rows for every parsed flag; source is "default" when the value equals the parser default."""
    return [(k.replace("_", "-"), v, "default" if v == ap.get_default(k) else "cli") for k, v in vars(args).items()]


@contextmanager
def nano_progress(total: int, desc: str) -> Iterator[Callable[[int], None]]:
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
