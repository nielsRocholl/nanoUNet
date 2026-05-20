"""Process runtime: TMPDIR off tmpfs, cgroup scope, startup banner."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from nanounet.mem_diag import cgroup_scope, log_snapshot, mem_diag_enabled, purge_torch_tmp, tmp_fs_type


def _is_tmpfs(path: str) -> bool:
    p = Path(path).resolve()
    while p != p.parent:
        try:
            if tmp_fs_type(str(p)) == "tmpfs":
                return True
        except OSError:
            pass
        p = p.parent
    return tmp_fs_type("/tmp") == "tmpfs"


def _writable_dir(path: str) -> bool:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        f = p / ".nanounet_tmp_probe"
        f.write_text("ok")
        f.unlink()
        return True
    except OSError:
        return False


def set_safe_tmpdir(*, results_tmp: str | None = None) -> str:
    """Point TMPDIR/TMP/TEMP at first writable non-tmpfs path."""
    home = os.environ.get("HOME") or "/tmp"
    cands = [
        os.environ.get("NANOUNET_TMPDIR", "").strip(),
        join_safe(home, ".cache", "nanounet_tmp"),
        results_tmp or "",
    ]
    chosen = ""
    for c in cands:
        if not c or _is_tmpfs(c):
            continue
        if _writable_dir(c):
            chosen = str(Path(c).resolve())
            break
    if not chosen:
        fallback = str(Path(home) / ".cache" / "nanounet_tmp")
        _writable_dir(fallback)
        chosen = str(Path(fallback).resolve())
    for k in ("TMPDIR", "TMP", "TEMP"):
        os.environ[k] = chosen
    tempfile.tempdir = chosen
    return chosen


def join_safe(*parts: str) -> str:
    return str(Path(*parts))


def _git_head() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[1],
        )
        if r.returncode == 0:
            return r.stdout.strip()[:12]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def runtime_banner(out_dir: str | None = None) -> dict[str, Any]:
    import nanounet

    tmp = os.environ.get("TMPDIR", "")
    row: dict[str, Any] = {
        "nanounet_file": getattr(nanounet, "__file__", None),
        "git_head": _git_head(),
        "tmpdir": tmp,
        "tmpdir_fs": tmp_fs_type(tmp) if tmp else None,
        "cgroup_scope": cgroup_scope(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    n = purge_torch_tmp()
    if n:
        row["purged_tmp_files"] = n
    if mem_diag_enabled() and out_dir:
        log_snapshot("runtime_banner", out_dir, extra=row)
    print(
        f"[nanounet] tmpdir={tmp} fs={row['tmpdir_fs']} cgroup={row['cgroup_scope']} "
        f"git={row['git_head'] or '?'}",
        file=sys.stderr,
    )
    return row


def assert_mem_diag_cgroup() -> None:
    if not mem_diag_enabled():
        return
    if os.environ.get("NANOUNET_ALLOW_ROOT_CGROUP", "").strip() in ("1", "true", "yes"):
        return
    if cgroup_scope() == "root" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "mem-diag on root cgroup (0::/) measures node-wide RAM, not this process. "
            "Submit via Slurm or set NANOUNET_ALLOW_ROOT_CGROUP=1 to override."
        )
