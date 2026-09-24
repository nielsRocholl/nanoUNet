"""Host-RAM diagnostics switch and logs: the --mem-diag flag, JSONL snapshot writer, worker logs.

The readers (RSS, cgroup, GPU, snapshot row) live in mem_probe.py; this module owns the process-
global `_MEM_DIAG` flag (inherited by DataLoader workers under fork only) and where rows go."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nanounet.diag.mem_probe import snapshot

_MEM_DIAG = False
_WORKER_LOG_DIR: str | None = None


def set_mem_diag(enabled: bool) -> None:
    global _MEM_DIAG
    _MEM_DIAG = enabled


def mem_diag_enabled() -> bool:
    return _MEM_DIAG or os.environ.get("NANOUNET_MEM_DIAG", "").strip() in ("1", "true", "yes")


def mem_log_every() -> int:
    v = os.environ.get("NANOUNET_MEM_LOG_EVERY", "").strip()
    return int(v) if v.isdigit() and int(v) > 0 else 0


def set_worker_log_dir(path: str | None) -> None:
    global _WORKER_LOG_DIR
    _WORKER_LOG_DIR = path


def worker_log_dir() -> str | None:
    return _WORKER_LOG_DIR

def append_jsonl(path: str, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def log_snapshot(
    tag: str,
    out_dir: str,
    extra: dict[str, Any] | None = None,
    filename: str = "mem_diag.jsonl",
) -> dict[str, Any]:
    if not mem_diag_enabled():
        return {}
    row = snapshot(tag, extra)
    append_jsonl(str(Path(out_dir) / filename), row)
    return row


def worker_diag_init(wid: int, out_dir: str) -> None:
    if not mem_diag_enabled():
        return
    set_worker_log_dir(out_dir)
    log_snapshot(f"worker_{wid}_start", out_dir, filename=f"mem_diag_worker_{wid}.jsonl")


def worker_diag_tick(wid: int, extra: dict[str, Any]) -> None:
    if not mem_diag_enabled():
        return
    d = worker_log_dir() or "."
    every = mem_log_every() or 100
    opens = extra.get("opens", 0)
    if opens and opens % every != 0:
        return
    log_snapshot(f"worker_{wid}_tick", d, extra=extra, filename=f"mem_diag_worker_{wid}.jsonl")


def worker_diag_iter_end(wid: int, extra: dict[str, Any]) -> None:
    if not mem_diag_enabled():
        return
    d = worker_log_dir() or "."
    log_snapshot(f"worker_{wid}_iter_end", d, extra=extra, filename=f"mem_diag_worker_{wid}.jsonl")
