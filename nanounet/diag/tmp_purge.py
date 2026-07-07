"""Purge orphaned torch checkpoint-stage + IPC temp files from /tmp, TMPDIR, /dev/shm."""

from __future__ import annotations

import os
import re
import time

from nanounet.diag.cgroup import tmp_fs_type

_PK_MAGIC = b"PK\x03\x04"
_TMPFILE_RE = re.compile(r"^tmp[a-zA-Z0-9_]{8}$")


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
