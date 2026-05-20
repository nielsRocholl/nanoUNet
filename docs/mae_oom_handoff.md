# Handoff: MAE / training cgroup OOM (Dataset999, Slurm 250G)

**Do not treat job-splitting, raising `--mem`, or Slurm-only workarounds as acceptable fixes.** Need a single long run (MAE 1000 epochs, then supervised) without unbounded cgroup growth.

---

## 1. Problem

**Symptom:** Integrated MAE pretrain on `Dataset999_Merged` (fold 0, ResEnc-L) is killed by Linux **cgroup OOM** (~epoch 100 historically; projected ~490 with latest slope). Slurm reports `oom_kill`; **no Python/CUDA traceback**.

| Item | Value |
|------|-------|
| Dataset | `Dataset999_Merged` (~545 GB preprocessed; 4267 train / 1044 val cases after patch filter) |
| Patch / batch | `[96, 160, 160]`, batch 10 |
| Slurm RAM | 250G target |
| Phase in flight | **MAE only** (`--mae-pretrain`); supervised not started |
| Data paths | NFS `/nnunet_data/NanoUNet_preprocessed`; local `/root/NanoUNet_preprocessed` (latest runs) |
| Results | `/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/` |

**Failure mode:** `cgroup_file` (page cache) + `cgroup_shmem` grow ~**3 GB/epoch combined** while process RSS stays ~5–10 GB. GPU fine (~33 GB peak on H200).

**Mechanism:** Random patch I/O — `PretrainPatchIterable` / `_PatchIterable` → random `case_id` → `Blosc2Folder.open_case` → crop → close. Over epochs, touched `.b2nd` pages accumulate in cgroup page cache. Separately, DataLoader multiprocessing used to leak **tmpfs shmem** (`/tmp`, `/dev/shm` are 1 TB tmpfs on `dlc-slowpoke`).

**Ruled out:** Python heap leak (`cgroup_anon` flat ~2–4 GB), unclosed blosc2 FDs, GPU OOM, wrong batch size, `persistent_workers`, supervised loss (still MAE).

---

## 2. Two independent contributors (both must be fixed)

| # | Metric | Cause |
|---|--------|--------|
| **A** | `cgroup_file` | Kernel page cache from buffered blosc2 reads (mmap or not) |
| **B** | `cgroup_shmem` | PyTorch DataLoader IPC (`memfd` / `/tmp/torch_*`) when `num_workers > 0` |

Startup scan (`_keys_fit_patch`) was fixed earlier (header-only shape, no mmap); no longer adds measurable cache.

---

## 3. What was tried (chronological)

### Wave 1 — Handle lifecycle + dataloader hygiene
- `open_case` context manager, explicit `_close_b2`
- Case-sticky sampling (K = `batch_size` patches per open)
- `persistent_workers=False`
- `--dl-bucket s` (2 train / 1 val workers)
- **Result:** OOM ~epoch 100; growth continued

### Wave 2 — `--mem-diag`
- [`nanounet/mem_diag.py`](../nanounet/mem_diag.py): cgroup file/anon/shmem, RSS, GPU → `mae_pretrain/mem_diag.jsonl` + W&B `mem/*`
- **Finding:** ~+0.8–1.5 GB/epoch `cgroup_file`; `cgroup_anon` flat

### Wave 3 — Disable mmap
- `mmap=False` default, `case_spatial_shape()` header-only, MAE `need_seg=False`, `pin_memory=False`
- **Micro-benchmark (200 cases, 1 process, local disk):** mmap +1.33 GB file; no-mmap +0.00 GB
- **Multi-epoch training:** growth **continued** (~1.5 GB/epoch file **and** shmem)

### Wave 4 — Two-target fix (current repo `/root/nanounet`)

**Fix A — page cache:** `posix_fadvise(POSIX_FADV_DONTNEED)` after every `.b2nd` close in [`nanounet/data/blosc2_dataset.py`](../nanounet/data/blosc2_dataset.py) (`_fadvise_dontneed`, called from `open_case` + `case_spatial_shape`). Applies to **MAE and supervised** (shared `Blosc2Folder`).

**Fix B — shmem / DataLoader:**
- `--dl-bucket s` → **`num_workers=0`** train+val unless `NANOUNET_DL_KEEP_WORKERS=1` (or legacy `NANOUNET_MAE_KEEP_WORKERS=1`)
- Fallback when keep-workers: `torch.multiprocessing` **`file_system`** strategy + `purge_torch_tmp()` each epoch ([`nanounet/dataloader_prefs.py`](../nanounet/dataloader_prefs.py), [`NanoMAELM`](../nanounet/pretrain/module.py))
- Shared [`build_iter_dataloader()`](../nanounet/dataloader_prefs.py) used by MAE + supervised [`data_module.py`](../nanounet/train/data_module.py)
- Per-epoch `cgroup_file_delta_bytes` / `cgroup_shmem_delta_bytes` in W&B ([`NanoMAELM`](../nanounet/pretrain/module.py), [`NanoUNetLM`](../nanounet/train/lightning_module.py) — also **fixed** `configure_optimizers` return that had been accidentally inside `on_exception`)

**Offline validation (local `/root/NanoUNet_preprocessed`, 5 epochs):**
- 5×250 `open_case` + fadvise: **0.000 GB/epoch** file
- DataLoader `nw=0`: **0.000 GB/epoch** file + shmem
- Fallback workers + fadvise: first epoch +0.09 GB, then flat

**Not acceptable:** Split MAE into multiple Slurm jobs; `--mem` bump without fixing slope.

---

## 4. Latest training observations

### Command (post-fix restart, interrupted ~epoch 428)
```bash
nanounet_train -d 999 -f 0 \
  --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --mae-pretrain --mae-epochs 1000 --epochs 2000 \
  --iters-per-epoch 250 --val-iters 50 \
  --dl-bucket s --accelerator cuda --precision 16-mixed \
  --mae-resume .../mae_pretrain/checkpoints/last.ckpt \
  --mem-diag
```
- W&B run: `5n3nc9or` (2026-05-20 ~10:17)
- Lightning warns **train/val dataloader have no workers** → Fix B (`nw=0`) **is active** in this process
- `mae_config` in mem_diag: `nw_train=0`, `nw_val=0`, `preprocessed_dir=/root/NanoUNet_preprocessed`

### mem_diag.jsonl (same file, multiple runs)

| Phase | Epochs | nw | file slope | shmem slope | Notes |
|-------|--------|-----|------------|-------------|-------|
| Pre-fix (mmap off, workers) | 414–423 | 2/1 | ~+1.5 GB/ep | ~+1.5 GB/ep | Deltas logged from ep 424+ only on newer code |
| Post-fix restart | 424–427 | **0** | **~+1.63 GB/ep** | **~+1.63 GB/ep** | After cache dip at ep 423 (21 GB file vs 34 GB prior) |

Example post-fix rows:
```
mae_epoch_424: file=22.29 shmem=20.42  fd=+0.82  sd=+1.63 GB
mae_epoch_425: file=23.92 shmem=22.05  fd=+1.64  sd=+1.63 GB
mae_epoch_426: file=25.56 shmem=23.68  fd=+1.63  sd=+1.63 GB
mae_epoch_427: file=27.19 shmem=25.31  fd=+1.63  sd=+1.63 GB
```

**Interpretation:** With **`nw=0`**, shmem should not grow from DataLoader IPC — yet it still climbs ~1.6 GB/epoch lockstep with file. **Fix A (fadvise) is not winning on this live path** despite passing on local offline tests. Possible causes:
1. `posix_fadvise(DONTNEED)` ineffective on this FS / cgroup (NFS behavior, immediate re-read, or pages not clean)
2. Non-DataLoader shmem (CUDA driver, W&B, other tmpfs) — needs isolation
3. Cgroup `0::/` on interactive node mixes workloads (directional only; still validate on real Slurm step cgroup)
4. Stale install: confirm `pip install -e /root/nanounet` and `_fadvise_dontneed` in loaded module before long run

**Projection:** ~3 GB/epoch total → OOM ~70 epochs after ep 427 → roughly **epoch 500** at 250G limit.

### Training health
- Speed ~1.2–2.4 it/s (slower with `nw=0` vs ~2.3 with workers; acceptable if RAM stable)
- Loss normal; checkpoints under `mae_pretrain/checkpoints/`

---

## 5. Key files

| File | Role |
|------|------|
| [`nanounet/data/blosc2_dataset.py`](../nanounet/data/blosc2_dataset.py) | `_fadvise_dontneed`, `open_case`, `case_spatial_shape` — **Fix A** |
| [`nanounet/dataloader_prefs.py`](../nanounet/dataloader_prefs.py) | `dataloader_bucket`, `build_iter_dataloader`, `dl_keep_workers`, `purge_torch_tmp` — **Fix B** |
| [`nanounet/pretrain/dataset.py`](../nanounet/pretrain/dataset.py) | `PretrainPatchIterable`, `build_pretrain_dataloaders` |
| [`nanounet/train/data_module.py`](../nanounet/train/data_module.py) | Supervised iterable (same I/O + Fix A/B) |
| [`nanounet/cli/train.py`](../nanounet/cli/train.py) | Integrated MAE+supervised; `mae_dl_b` for MAE |
| [`nanounet/cli/pretrain.py`](../nanounet/cli/pretrain.py) | Standalone MAE — still uses `dataloader_bucket` only (not `mae_dataloader_bucket` alias; same `s`→0 behavior now) |
| [`nanounet/mem_diag.py`](../nanounet/mem_diag.py) | Diagnostics |
| [`nanounet/pretrain/module.py`](../nanounet/pretrain/module.py) | `NanoMAELM` epoch mem deltas |
| `.../mae_pretrain/mem_diag.jsonl` | Ground truth for slopes |

---

## 6. Success criteria

Over **≥10 MAE epochs** under `--mem-diag` on **Slurm `#SBATCH --mem=250G`**:
- `cgroup_file_delta` **< 0.1 GB/epoch**
- `cgroup_shmem_delta` **< 0.1 GB/epoch**
- `cgroup_anon` ~2–4 GB flat
- No OOM through **epoch 200+** single job
- Speed within ~20% of baseline (~2.3 it/s with workers)

---

## 7. Suggested next steps (priority)

1. **Prove code path on live PID:** log once per epoch that `_fadvise_dontneed` ran (counter) and `nw_train` from config; verify editable install.
2. **Isolate shmem with `nw=0`:** one epoch with `num_workers=0`, no W&B, minimal Lightning — if shmem still +1.6 GB/ep, source is not DataLoader.
3. **Fix A escalation if fadvise fails on production FS:**
   - `sync_file_range` / range fadvise on bytes actually read
   - Read subset via `O_DIRECT` wrapper (heavy)
   - Stage fold to local NVMe and train from node-local path (I/O model change, not Slurm hack)
4. **Re-benchmark on same path + cgroup as training** (not 200-case single-process micro-bench only).
5. **Slurm validation** as acceptance gate (interactive root cgroup is misleading).

---

## 8. Environment notes

- Host: `dlc-slowpoke`, H200, `/tmp` and `/dev/shm` are **tmpfs ~1 TB** → DataLoader IPC counts as `cgroup_shmem`
- Raw data: `NANOUNET_RAW=/nnunet_data/nnUNet_raw` for dataloader build
- `b2_closes=0` in main-process epoch logs is **expected** (counter is thread-local in workers)

---

## 9. One-line summary

**Cgroup OOM is page-cache + shmem growth (~3 GB/epoch), not heap; fadvise + `nw=0` are implemented for all stages but the live MAE run still shows ~1.6 GB/epoch for both metrics — next agent must make Fix A work on the production filesystem and find the non-DataLoader shmem source (or prove fadvise is not executing).**
