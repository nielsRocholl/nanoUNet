"""Host resource readers: cgroup scope, mount fs-type (used by TMPDIR selection + purge)."""

from __future__ import annotations

import os
from pathlib import Path


def _cgroup_dir(pid: int | None = None) -> Path | None:
    pid = pid or os.getpid()
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            _h, _uid, path = line.split(":", 2)
            if "memory" in _h or path.startswith("/"):
                for base in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
                    cand = base / path.lstrip("/")
                    if (cand / "memory.current").is_file():
                        return cand
                unified = Path("/sys/fs/cgroup") / path.lstrip("/")
                if (unified / "memory.current").is_file():
                    return unified
    except OSError:
        pass
    return None


def cgroup_scope(pid: int | None = None) -> str:
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    cg = _cgroup_dir(pid)
    if cg is None:
        return "other"
    if cg == Path("/sys/fs/cgroup"):
        return "root"
    return "other"


def tmp_fs_type(path: str) -> str | None:
    try:
        target = str(Path(path).resolve())
        best_mp, best_fst = "", None
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mp, fst = parts[1], parts[2]
            if target == mp or target.startswith(mp + "/"):
                if len(mp) >= len(best_mp):
                    best_mp, best_fst = mp, fst
        return best_fst
    except OSError:
        return None
