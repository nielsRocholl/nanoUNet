"""Host-RAM diagnostics: /proc RSS, cgroup memory, GPU bytes, JSONL + W&B."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import threading
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any

_MEM_DIAG = False
_B2_CLOSES = threading.local()
_WORKER_LOG_DIR: str | None = None
_B2_GLOBAL: mp.sharedctypes.Synchronized | None = None
_FADV_GLOBAL: mp.sharedctypes.Synchronized | None = None
_PK_MAGIC = b"PK\x03\x04"
_TMPFILE_RE = re.compile(r"^tmp[a-zA-Z0-9_]{8}$")


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


def _b2_global() -> mp.sharedctypes.Synchronized:
    global _B2_GLOBAL
    if _B2_GLOBAL is None:
        _B2_GLOBAL = mp.Value("Q", 0)
    return _B2_GLOBAL


def _fadv_global() -> mp.sharedctypes.Synchronized:
    global _FADV_GLOBAL
    if _FADV_GLOBAL is None:
        _FADV_GLOBAL = mp.Value("Q", 0)
    return _FADV_GLOBAL


def b2_close_inc() -> int:
    v = _b2_global()
    with v.get_lock():
        v.value += 1
        return int(v.value)


def b2_close_count() -> int:
    return int(_b2_global().value)


def fadvise_inc() -> int:
    v = _fadv_global()
    with v.get_lock():
        v.value += 1
        return int(v.value)


def fadvise_count() -> int:
    return int(_fadv_global().value)


def _read_kv(path: Path, key: str) -> int | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split(":", 1)[1].strip().split()[0])
    except OSError:
        pass
    return None


def proc_rss_kb(pid: int | None = None) -> int | None:
    pid = pid or os.getpid()
    return _read_kv(Path(f"/proc/{pid}/status"), "VmRSS")


def proc_fds(pid: int | None = None) -> int | None:
    pid = pid or os.getpid()
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return None


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


def cgroup_path(pid: int | None = None) -> str | None:
    cg = _cgroup_dir(pid)
    return str(cg) if cg else None


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


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError):
        return None


def cgroup_mem_bytes(pid: int | None = None) -> dict[str, int | None]:
    cg = _cgroup_dir(pid)
    if cg is None:
        return {"current": None, "max": None, "anon": None, "file": None, "shmem": None}
    out: dict[str, int | None] = {
        "current": _read_int(cg / "memory.current"),
        "max": _read_int(cg / "memory.max"),
        "anon": None,
        "file": None,
        "shmem": None,
    }
    try:
        for line in (cg / "memory.stat").read_text().splitlines():
            k, v = line.split(maxsplit=1)
            if k in ("anon", "file", "shmem"):
                out[k] = int(v)
    except OSError:
        pass
    return out


def gpu_mem_bytes() -> dict[str, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated": None, "reserved": None, "max_allocated": None}
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
        }
    except Exception:
        return {"allocated": None, "reserved": None, "max_allocated": None}


def snapshot(tag: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cg = cgroup_mem_bytes()
    gpu = gpu_mem_bytes()
    row: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cgroup_path": cgroup_path(),
        "cgroup_scope": cgroup_scope(),
        "tmpdir": os.environ.get("TMPDIR"),
        "proc_rss_kb": proc_rss_kb(),
        "proc_fds": proc_fds(),
        "cgroup_current_bytes": cg["current"],
        "cgroup_max_bytes": cg["max"],
        "cgroup_anon_bytes": cg["anon"],
        "cgroup_file_bytes": cg["file"],
        "cgroup_shmem_bytes": cg["shmem"],
        "gpu_allocated_bytes": gpu["allocated"],
        "gpu_reserved_bytes": gpu["reserved"],
        "gpu_max_allocated_bytes": gpu["max_allocated"],
        "b2_closes": b2_close_count(),
        "fadvise_calls": fadvise_count(),
    }
    if extra:
        row.update(extra)
    return row


def _wandb_scalars(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for src, dst in (
        ("cgroup_current_bytes", "mem/cgroup_current_gb"),
        ("cgroup_anon_bytes", "mem/cgroup_anon_gb"),
        ("cgroup_file_bytes", "mem/cgroup_file_gb"),
        ("cgroup_shmem_bytes", "mem/cgroup_shmem_gb"),
        ("cgroup_file_delta_bytes", "mem/cgroup_file_delta_gb"),
        ("cgroup_shmem_delta_bytes", "mem/cgroup_shmem_delta_gb"),
        ("gpu_allocated_bytes", "mem/gpu_allocated_gb"),
        ("gpu_max_allocated_bytes", "mem/gpu_max_allocated_gb"),
    ):
        v = row.get(src)
        if v is not None:
            out[dst] = float(v) / (1024**3)
    if row.get("proc_rss_kb") is not None:
        out["mem/proc_rss_gb"] = float(row["proc_rss_kb"]) / (1024**2)
    if row.get("proc_fds") is not None:
        out["mem/proc_fds"] = float(row["proc_fds"])
    return out


def cgroup_epoch_deltas(
    prev_file: int | None, prev_shmem: int | None
) -> tuple[int | None, int | None]:
    cg = cgroup_mem_bytes()
    f, s = cg.get("file"), cg.get("shmem")
    fd = (f - prev_file) if f is not None and prev_file is not None else None
    sd = (s - prev_shmem) if s is not None and prev_shmem is not None else None
    return fd, sd


def _is_pytorch_zip_stage(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == _PK_MAGIC
    except OSError:
        return False


def _purge_ckpt_stage_files(directory: str, max_age_sec: float) -> int:
    if directory == "/tmp" and tmp_fs_type("/tmp") != "tmpfs":
        return 0
    now = time.time()
    uid = os.getuid()
    n = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    for name in names:
        if not _TMPFILE_RE.fullmatch(name):
            continue
        p = os.path.join(directory, name)
        try:
            st = os.stat(p, follow_symlinks=False)
        except OSError:
            continue
        if not os.path.isfile(p) or st.st_uid != uid:
            continue
        if st.st_size < 64 * 1024 * 1024 or now - st.st_mtime < max_age_sec:
            continue
        if not _is_pytorch_zip_stage(p):
            continue
        try:
            os.unlink(p)
            n += 1
        except OSError:
            pass
    return n


def _purge_ipc_files(directory: str, prefixes: tuple[str, ...], max_age_sec: float) -> int:
    now = time.time()
    uid = os.getuid()
    n = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    for name in names:
        if not any(name.startswith(p) for p in prefixes):
            continue
        p = os.path.join(directory, name)
        try:
            st = os.stat(p, follow_symlinks=False)
        except OSError:
            continue
        if st.st_uid != uid or now - st.st_mtime < max_age_sec:
            continue
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.unlink(p)
            elif os.path.isdir(p):
                os.rmdir(p)
            n += 1
        except OSError:
            pass
    return n


def _purge_dev_shm_torch(max_age_sec: float) -> int:
    return _purge_ipc_files("/dev/shm", ("torch_", "pymp-"), max_age_sec)


def purge_torch_tmp(max_age_sec: float = 60.0) -> int:
    n = _purge_ckpt_stage_files("/tmp", max_age_sec)
    tmpdir = os.environ.get("TMPDIR", "").strip()
    if tmpdir and tmpdir != "/tmp":
        n += _purge_ckpt_stage_files(tmpdir, max_age_sec)
    n += _purge_ipc_files("/tmp", ("torch_", "pymp-"), max_age_sec)
    n += _purge_dev_shm_torch(max_age_sec)
    return n


def log_wandb_scalars(module: Any, row: dict[str, Any]) -> None:
    if not row:
        return
    for k, v in _wandb_scalars(row).items():
        module.log(k, v, prog_bar=False)


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
