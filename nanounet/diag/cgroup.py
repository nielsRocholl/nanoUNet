"""Host resource readers: cgroup scope, mount fs-type, preprocess OOM diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


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
    except OSError:  # nanochat-style: allow E4 (no cgroup fs; mem-diag degrades)
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


def _cgroup_mem_limit_gb() -> Optional[float]:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p, encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError:  # nanochat-style: allow E4 (cgroup file gone mid-read)
            continue
        if raw == "max":
            continue
        try:
            v = int(raw)
        except ValueError:  # nanochat-style: allow E4 (malformed cgroup int)
            continue
        if v < (1 << 40) * 1024:  # exclude the "no limit" sentinel some kernels report as a huge int
            return v / 1e9
    return None


def _cgroup_oom_kills() -> Optional[int]:
    try:
        with open("/sys/fs/cgroup/memory.events", encoding="utf-8") as f:
            for line in f:
                if line.startswith("oom_kill "):
                    return int(line.split()[1])
    except OSError:  # nanochat-style: allow E4 (no memory.events)
        return None
    return None


def _dead_worker_error(num_processes: int, resume_flag: str) -> RuntimeError:
    lim_gb = _cgroup_mem_limit_gb()
    oom_kills = _cgroup_oom_kills()
    lines = [
        "A preprocess worker was killed with no Python exception (SIGKILL) -- almost always an "
        "out-of-memory (OOM) kill by the cgroup, not a bug in the preprocessing code.",
    ]
    if lim_gb is not None:
        per_worker = lim_gb / num_processes
        lines.append(
            f"cgroup memory limit: {lim_gb:.1f} GB across {num_processes} workers = "
            f"{per_worker:.1f} GB/worker."
        )
    else:
        lines.append(f"{num_processes} workers configured (cgroup memory limit unreadable).")
    if oom_kills:
        lines.append(f"cgroup memory.events reports {oom_kills} oom_kill event(s) so far.")
    lines.append(
        "Large volumes can peak at ~50 GB/worker during resampling, so a worker count sized for "
        "average cases can still get OOM-killed on the largest ones."
    )
    lines.append(f"Fix: rerun with fewer workers: nanounet_preprocess ... -np {max(1, num_processes // 2)}{resume_flag}")
    return RuntimeError("\n".join(lines))


