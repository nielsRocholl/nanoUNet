# Host RAM / cgroup OOM (MAE & supervised training)

Dataset999 MAE on `dlc-slowpoke` was killed by Linux **cgroup OOM** around epoch 100 (`oom_kill`, no Python traceback). GPU memory was fine (~33 GB). Process RSS stayed ~5–10 GB while **cgroup memory** kept climbing.

This doc summarizes what we saw, what actually caused it, and what the code does now.

---

## Symptoms

| Signal | Healthy | Broken (pre-fix) |
|--------|---------|------------------|
| `cgroup_anon` | Flat ~2–4 GB | Flat (not a heap leak) |
| `cgroup_shmem` | Stable | **+~1.6 GB/epoch** |
| `cgroup_file` | Stable or reclaimable | **+~1.5 GB/epoch** (often lockstep with shmem) |
| `/tmp` on node | Small | **Dozens of ~779 MB files** (`tmpXXXXXXXX`) |
| Training speed (`--dl-bucket s`) | ~100 s/epoch (2 workers) | ~3–4 min/epoch (`num_workers=0` debug mode) |

Slurm limit was 250G; projected OOM ~epoch 100–500 depending on slope.

---

## Root causes (two layers)

### 1. Checkpoint temp files on RAM-backed `/tmp` (main OOM)

Lightning saves checkpoints atomically: write a full temp file, then rename to `last.ckpt`.

- Default `TMPDIR` was `/tmp`, which on this host is **tmpfs** (counts as `cgroup_shmem`, not reclaimable under the job limit).
- Each MAE `last.ckpt` is **~779 MB**.
- Ctrl+C, OOM, or failed saves leave orphaned temp files (`tmpXXXXXXXX`, zip header `PK\x03\x04`).
- We found **~26 GB** of these on `/tmp` after interrupted runs.

This explains **shmem growing ~1.6 GB/epoch** in lockstep with file metrics: it was checkpoint staging on tmpfs, not the training heap.

### 2. Page cache from Blosc2 reads (secondary)

Random patch I/O opens many `.b2nd` files per epoch. Buffered reads fill the kernel **page cache** (`cgroup_file`). Closing blosc2 handles does not evict cache pages.

Mitigations already in code:

- No mmap on training reads (`mmap=False`).
- `posix_fadvise(DONTNEED)` after every `.b2nd` close.
- Case-sticky sampling (K patches per open).

Under a real Slurm step cgroup, page cache is usually reclaimable; tmpfs checkpoint temps are not.

### 3. Misleading metrics on the interactive node

On VS Code / interactive sessions, `/proc/self/cgroup` is often `0::/` (root cgroup). `--mem-diag` then reports **node-wide** memory, not just your process. Use Slurm for authoritative cgroup deltas, or set `NANOUNET_ALLOW_ROOT_CGROUP=1` knowing the numbers are noisy.

---

## What we tried (short history)

1. **Handle hygiene** — context managers, case-sticky I/O, small dl bucket. OOM continued.
2. **`--mem-diag`** — confirmed growth in `cgroup_file` + `cgroup_shmem`, flat `cgroup_anon`.
3. **Disable mmap** — helped micro-benchmarks; full training still grew.
4. **`posix_fadvise` + `num_workers=0`** — shmem still climbed until we moved TMPDIR off tmpfs.
5. **TMPDIR redirect** — shmem flat; workers re-enabled when TMPDIR is on disk.

Rejected as fixes: raising `--mem`, splitting MAE into multiple jobs, job-only workarounds without stopping growth.

---

## Fixes in code

| Component | What it does |
|-----------|----------------|
| [`nanounet/runtime.py`](../nanounet/runtime.py) | `set_safe_tmpdir()` — TMPDIR on disk, not tmpfs; prefers `$NANOUNET_TMPDIR`, `<run>/.tmp`, `$NANOUNET_RESULTS/.nanounet_tmp`; warns if under `$HOME` |
| [`nanounet/mem_diag.py`](../nanounet/mem_diag.py) | Per-epoch cgroup snapshots; purge stale checkpoint temps on `/tmp` + `TMPDIR`, `/dev/shm` torch IPC (uid + age); `cgroup_scope` |
| [`nanounet/data/blosc2_dataset.py`](../nanounet/data/blosc2_dataset.py) | `fadvise` after close; `POSIX_FADV_RANDOM` on open |
| [`nanounet/dataloader_prefs.py`](../nanounet/dataloader_prefs.py) | `--dl-bucket` s/m/l (2/1, 4/2, 8/4 train/val when TMPDIR off tmpfs); `--dl-persistent-workers`; `file_system` IPC; `NANOUNET_DL_FORCE_NO_WORKERS=1` escape hatch |

Each epoch: `purge_torch_tmp()` removes stale checkpoint stage files under `/tmp` (if tmpfs), `TMPDIR`, and stale `/dev/shm/torch_*` owned by your uid (belt-and-suspenders).

---

## Residual leak after primary fix (epochs 431+)

The **~1.6 GB/epoch tmpfs checkpoint leak is fixed** (`/tmp` clean, TMPDIR on local zfs). A smaller steady-state growth appeared once workers were re-enabled.

### What we still see (with DataLoader workers)

| Metric | Rate | Cause |
|--------|------|--------|
| `cgroup_shmem_delta` | **~98 MB/epoch** (constant) | One leaked prefetched batch in `/dev/shm` (~93.75 MiB for batch 10 × `96×160×160`); PyTorch worker IPC uses tmpfs despite `file_system` strategy |
| `cgroup_file_delta` | **~80–110 MB/epoch** | Residual Blosc2 page cache; `fadvise` active (~300 calls/epoch) but not zero |
| TMPDIR disk | **~1.6 GB/epoch** | Orphan ~816 MB checkpoint stage files (two `ModelCheckpoint` callbacks); old purge only scanned `/tmp` |

**Slurm projection (with workers):** ~98 MB/ep × 570 remaining epochs ≈ **56 GB** shmem — much safer than the old ~900 GB path, but not zero.

**With `num_workers=0`:** `cgroup_shmem_delta` ≈ 0 (proven in logs). Tradeoff: ~3–4 min/epoch vs ~100 s with workers.

### Recommendations

| Priority | Action | Effect |
|----------|--------|--------|
| 1 | **`NANOUNET_TMPDIR` on local disk** + `--dl-bucket m`/`l` + **`--dl-persistent-workers`** | Fast MAE; avoids per-epoch worker respawn; ~98 MB/ep shmem (monitor) |
| 2 | **Extended purge** (TMPDIR + `/dev/shm`) | Stops disk orphans; trims IPC debris |
| 3 | Optional: single MAE `ModelCheckpoint` | Halves stage-file churn (future) |
| 4 | Slurm validation (`cgroup_scope=slurm`) | Authoritative metrics vs interactive root cgroup |
| — | Escape: `NANOUNET_DL_FORCE_NO_WORKERS=1` | Shmem delta → ~0; ~3–4 min/epoch |

### Purge safety (Docker / shared node)

Purge only deletes files that match **all** of:

- **Your uid** (`st_uid == os.getuid()`)
- **Age** > 60 s (not in active use)
- **Specific patterns** (checkpoint stage `tmpXXXXXXXX` with zip header; `/dev/shm/torch_*`, `pymp-*`)

| Target | Safe for other users? |
|--------|------------------------|
| `TMPDIR` (e.g. `/root/.cache/nanounet_tmp`) | Yes — other uids skipped |
| `/dev/shm` | Yes — other uids skipped; does not wipe all of `/dev/shm` |
| `/tmp` (tmpfs) | Yes — same uid + pattern guards |

**Caveat:** Two jobs under the **same uid** in one container could delete each other's **stale** (>60 s) torch IPC files. Run one training job at a time.

**Not purged:** Final checkpoints under `$NANOUNET_RESULTS/.../checkpoints/`.

---

## Required environment (interactive & Slurm)

```bash
export NANOUNET_ALLOW_ROOT_CGROUP=1   # only for --mem-diag on interactive (root cgroup)
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp   # local disk; CIFS/NFS breaks DataLoader workers
```

Also set the usual paths:

```bash
export NANOUNET_RAW=/nnunet_data/nnUNet_raw
export NANOUNET_PREPROCESSED=/path/to/NanoUNet_preprocessed
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results
```

On Slurm you typically **do not** need `NANOUNET_ALLOW_ROOT_CGROUP` (step cgroup is correct). Still set `NANOUNET_TMPDIR` to a path with space (not `$HOME` if quota is 10 GB).

---

## What goes in `NANOUNET_TMPDIR`

Scratch only — **not** final checkpoints or W&B artifacts.

| Content | Size | Notes |
|---------|------|--------|
| Checkpoint staging | ~779 MB per save | Temp copy before rename to `<run>/checkpoints/` on NFS |
| DataLoader IPC files | Small–medium | When `num_workers > 0`; uses `file_system` strategy on disk |
| Other Python/torch temps | Small | Short-lived |

Final outputs stay under `$NANOUNET_RESULTS/...` (checkpoints, `mem_diag.jsonl`, configs).

---

## Monitoring (`--mem-diag`)

Logs: `<run>/mae_pretrain/mem_diag.jsonl` (and W&B `mem/*` if enabled).

**Fix is holding if:**

- Startup banner: `tmpdir=...` on **local zfs** (e.g. `/root/.cache/nanounet_tmp`), not `/tmp` or CIFS
- `fadvise_calls` increases each epoch
- `du -sh /tmp` stays small; TMPDIR stage orphans not accumulating without bound
- For MAE with workers + `--dl-persistent-workers`: expect ~98 MB/ep shmem (acceptable on 250G jobs); use `NANOUNET_DL_FORCE_NO_WORKERS=1` only if shmem must stay flat

**Optional env vars:**

| Variable | Purpose |
|----------|---------|
| `NANOUNET_MEM_DIAG=1` | Same as `--mem-diag` |
| `NANOUNET_DL_FORCE_NO_WORKERS=1` | Escape hatch: force `num_workers=0` (slow, ~0 shmem/ep) |
| `NANOUNET_DL_KEEP_WORKERS=1` | Legacy: keep workers even on tmpfs (not recommended) |

Validation script: [`scripts/interactive_validate_mem.sh`](../scripts/interactive_validate_mem.sh).

---

## Home disk quota

If `TMPDIR` points at `$HOME/.cache`, each checkpoint save needs **~800 MB free**. A 10 GB home quota can fail with `OSError: [Errno 122] Disk quota exceeded` even when NFS results have space. Use `/root/.cache/nanounet_tmp` (local zfs). **Do not** use CIFS/NFS for TMPDIR — breaks DataLoader workers (`torch_shm_manager: Operation not permitted`).
