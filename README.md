# nanoUNet

Minimal prompt-aware 3D ResEnc nnU-Net with PyTorch Lightning and optional MAE pretraining. Layout and style follow [nanochat](https://github.com/karpathy/nanochat): small modules, no framework sprawl.

## Getting started

```bash
python -m pip install -e .
```

Environment (same as nnU-Net; `NANOUNET_*` aliases are synced to `nnUNet_*`):

```bash
export NANOUNET_RAW="/path/to/nnUNet_raw"
export NANOUNET_PREPROCESSED="/path/to/nnUNet_preprocessed"
export NANOUNET_RESULTS="/path/to/nnUNet_results"

# Host-RAM / checkpoint staging (see docs/cgroup_memory.md)
export NANOUNET_TMPDIR=/nnunet_data/.nanounet_tmp   # avoids /tmp tmpfs OOM and $HOME quota
export NANOUNET_ALLOW_ROOT_CGROUP=1                 # only if using --mem-diag on interactive node
```

Quote paths that contain spaces.

---

## CLI reference

### `nanounet_preprocess`

Fingerprint → ResEnc plan → resample to blosc2 (`3d_fullres`). One dataset id or multiple ids (merge → `Dataset<merged-id>_<merged-name>`).

| Argument | Default | Description |
|----------|---------|-------------|
| `-d`, `--dataset_id` | (required) | One or more integers, e.g. `-d 001` or `-d 1 2 3` for merge |
| `--merged-id` | `999` | Output dataset id when merging multiple `-d` |
| `--merged-name` | `Merged` | Name segment for merged folder `DatasetNNN_<name>` |
| `--planner` | `nnUNetPlannerResEncL` | Planner class (e.g. `nnUNetPlannerResEncTiny`, `…ResEncL`) |
| `-np`, `--num_processes` | `8` | Parallel workers for fingerprint / preprocess |
| `--resume` | off (flag) | Keep existing preprocess output; do not wipe `3d_fullres` folder |
| `--gpu-memory-gb` | none | VRAM budget (GB) for planner patch shrink loop |
| `--patch-vol` | `large` | `small` (128) \| `medium` (192) \| `large` (256) \| `xlarge` (320) isotropic edge before aniso handling |
| `--plans-name` | none | Basename of plans JSON (no `.json`) when using `--skip-plan` |
| `--config-path` | none | Optional path forwarded into preprocessing |
| `--skip-fingerprint` | off (flag) | Skip fingerprint; use existing `dataset_fingerprint.json` |
| `--skip-plan` | off (flag) | Skip planning; requires `--plans-name` |

Artifacts: `dataset_fingerprint.json`, `<plans>.json`, `<plans>/3d_fullres/*.b2nd`, `gt_segmentations` under the raw dataset folder.

---

### `nanounet_train`

Supervised training on one fold; optional integrated MAE then supervised. Default run dir: `$NANOUNET_RESULTS/nanounet/<DatasetFolder>_<plans>_f<fold>/`.

| Argument | Default | Description |
|----------|---------|-------------|
| `-d`, `--dataset_id` | (required) | Dataset id (folder must exist under raw/preprocessed as `DatasetXXX_*`) |
| `-f`, `--fold` | `0` | Fold index (0–4 with default 5-fold split) |
| `--plans` | (required) | Plans basename without `.json` (must exist in preprocessed dataset dir) |
| `--config` | `configs/default.json` | ROI / prompt JSON (relative resolves from cwd, then repo root) |
| `--epochs` | `1000` | Supervised epoch budget |
| `--lr` | `0.01` | Supervised initial learning rate |
| `--wd` | `3e-5` | Weight decay |
| `--batch-size` | from plans | If omitted, `3d_fullres.batch_size` from plans |
| `--iters-per-epoch` | `250` | Training batches per epoch (iterable dataloaders) |
| `--val-iters` | `50` | Validation batches per epoch |
| `--out` | see above | Override run directory |
| `--lr-schedule` | `poly` | `poly` \| `stretched_tail_poly` |
| `--stretched-k` | `750` | Used when `stretched_tail_poly` |
| `--stretched-ref` | `1000` | Used when `stretched_tail_poly` |
| `--stretched-exp` | `0.9` | Used when `stretched_tail_poly` |
| `--no-wandb` | off (flag) | Disable W&B logger |
| `--wandb-project` | `nanounet` | W&B project name |
| `--wandb-name` | auto | W&B run name |
| `--loss`, `-loss` | `dc_ce` | `dc_ce` \| `cc_dc_ce` (CC-DiceCE; see [docs/losses.md](docs/losses.md)) |
| `--resume` | none | Supervised Lightning checkpoint path; must exist. No auto `last.ckpt`. If set and ckpt already finished `epochs`, exits cleanly after checks |
| `--precision` | `16-mixed` | Passed to Lightning (e.g. `32-true`) |
| `--accelerator` | `auto` | `auto` \| `cpu` \| `cuda` \| `gpu` \| `mps` (`gpu` normalized to `cuda`) |
| `--mae-ckpt` | none | Load encoder weights only (no integrated MAE run) |
| `--mae-pretrain` | off (flag) | Run MAE under `<run>/mae_pretrain/` then supervised |
| `--mae-resume` | none | With `--mae-pretrain` only: MAE ckpt to resume or to detect MAE already done; must match `--mae-epochs`. Conflicts with `--mae-ckpt` |
| `--mae-epochs` | `1000` | MAE epoch budget when using `--mae-pretrain` |
| `--mae-lr` | `1e-2` | MAE initial LR |
| `--mae-mask-ratio` | `0.75` | MAE mask ratio |
| `--mae-iters-per-epoch` | same as `--iters-per-epoch` | MAE batches per epoch |
| `--dl-bucket` | `m` | `s`: 2/1 workers when TMPDIR is off tmpfs, else 0. `m`/`l`: more workers. Case-sticky Blosc2 I/O, no mmap, non-persistent workers. |
| `--mem-diag` | off (flag) | Log cgroup/process RAM to `<run>/mem_diag.jsonl` and W&B `mem/*`. Requires `NANOUNET_ALLOW_ROOT_CGROUP=1` on interactive nodes. See [docs/cgroup_memory.md](docs/cgroup_memory.md). |

Checkpoints: `<run>/checkpoints/last.ckpt` (supervised); `<run>/mae_pretrain/checkpoints/` (integrated MAE). Re-running without `--resume` / `--mae-resume` trains from scratch and may overwrite those paths.

#### Host RAM / cgroup OOM

Long MAE runs were killed by cgroup OOM from **checkpoint temp files on RAM-backed `/tmp`**, not GPU or Python heap. Full write-up: **[docs/cgroup_memory.md](docs/cgroup_memory.md)**.

Quick checks with `--mem-diag`: `cgroup_shmem_delta` ≈ 0 per epoch, `fadvise_calls` rising, `/tmp` not accumulating ~779 MB files. Set `NANOUNET_TMPDIR` to NFS or local disk (not `$HOME` if quota is small).

Slurm example: [`scripts/slurm_nanounet_train_mae_999.sh`](scripts/slurm_nanounet_train_mae_999.sh).

---

### `nanounet_pretrain`

Standalone MAE only (no prompts). Default out: `$NANOUNET_RESULTS/nanounet/<DatasetFolder>_<plans>_mae_pretrain_f<fold>/`.

| Argument | Default | Description |
|----------|---------|-------------|
| `-d`, `--dataset_id` | (required) | Dataset id |
| `-f`, `--fold` | `0` | Fold |
| `--plans` | (required) | Plans basename (no `.json`) |
| `--epochs` | `1000` | MAE epoch budget |
| `--lr` | `1e-2` | Initial LR |
| `--wd` | `3e-5` | Weight decay |
| `--mask-ratio` | `0.75` | Mask ratio |
| `--batch-size` | from plans | Optional override |
| `--iters-per-epoch` | `250` | Batches per epoch |
| `--val-iters` | `50` | Val batches per epoch |
| `--out` | auto | Output directory |
| `--no-wandb` | off (flag) | Disable W&B |
| `--wandb-project` | `nanounet-mae` | W&B project |
| `--wandb-name` | auto | W&B run name |
| `--dl-bucket` | `m` | `s` \| `m` \| `l` |
| `--resume` | none | MAE Lightning ckpt; must exist. No auto `last.ckpt`. Epoch target must match `--epochs` |
| `--precision` | `16-mixed` | Lightning precision |
| `--accelerator` | `auto` | `auto` \| `cpu` \| `cuda` \| `gpu` \| `mps` |

---

### `nanounet_predict`

Segment from preprocessed-space point prompt; TTA on by default.

| Argument | Default | Description |
|----------|---------|-------------|
| `-i`, `--images` | (required) | One or more input image paths (channels) |
| `-o`, `--output` | (required) | Output `.nii.gz` or base path (suffix from `dataset.json`) |
| `-m`, `--model-dir` | (required) | Training run directory (with `plans.json`, `dataset.json`, `checkpoints/`) |
| `--ckpt` | auto | Checkpoint name/path; default picks from `checkpoints/` via [predictor](nanounet/infer/predictor.py) |
| `--point-zyx` | (required) | Comma-separated `z,y,x` in **preprocessed** voxels |
| `--seg` | none | Optional segmentation input for predictor |
| `--no-prompt-encode` | off (flag) | Skip prompt encoding |
| `--border-expand` | off (flag) | BFS hull-shell merge (see [infer/border_expand.py](nanounet/infer/border_expand.py)) |
| `--max-border-extra` | library default | Max extra voxels for border expand |
| `--disable-tta` | off (flag) | Disable test-time augmentation |
| `--device` | `cuda` | Torch device string for inference |

---

## Pipeline overview

**1 — Preprocess**

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncL -np 8
```

Planner **Tiny** / **L** / **XL** set network scale (~1–5M vs 8–40+ GB VRAM targets). The `<plans>.json` sets patch size and topology—not [configs/default.json](configs/default.json). For small local models use Tiny everywhere (preprocess + train).

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncTiny --patch-vol small -np 4
```

Cluster example (CPU preprocess, large GPU train): add `--gpu-memory-gb 80 --patch-vol medium`. See CLI table for `--skip-fingerprint`, `--skip-plan`, `--resume`.

Set `NANOUNET_TMPDIR` (see Environment). Training reads Blosc2 without mmap; MAE opens data only (not seg). Use `--dl-bucket s` and `--mem-diag` on long runs; see [docs/cgroup_memory.md](docs/cgroup_memory.md).

**2 — Train**

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --config configs/default.json
```

First run writes `splits_final.json` (5-fold, fixed seed). MAE options: `--mae-pretrain` / `--mae-ckpt`, standalone `nanounet_pretrain`, encoder transfer ([Wald et al.](https://arxiv.org/abs/2410.23132)). Resume flags: **CLI reference** (`--resume`, `--mae-resume`). On Slurm, prefer `--dl-bucket s` if host RAM is tight.

Tiny laptop smoke:

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetTinyPlans --config configs/default.json \
  --epochs 2 --iters-per-epoch 50 --accelerator cpu --precision 32-true --batch-size 1 --no-wandb
```

**3 — Predict**

```bash
nanounet_predict -i case_0000.nii.gz -o seg.nii.gz -m /path/to/run --point-zyx 48,120,95
```

`--point-zyx` is in **preprocessed** space, not original scanner space.

---

## Configuration and docs

| Resource | Role |
|----------|------|
| [configs/default.json](configs/default.json) | ROI sampling, prompts: `click_modes` must sum to 1 (`pos` = jittered centroid prompt probability, `drop` = omit prompt). |
| [docs/cgroup_memory.md](docs/cgroup_memory.md) | Host RAM OOM: causes, fixes, `NANOUNET_TMPDIR`, `--mem-diag`. |
| [docs/losses.md](docs/losses.md) | DC+CE vs CC-DiceCE (`--loss cc_dc_ce`). |
| [docs/patch_size.md](docs/patch_size.md) | Patch FOV vs lesion scale. |

---

## Repository layout

```
nanounet/
├── common.py           # paths, Rich UI, logging shims
├── config.py           # RoiPromptConfig
├── dataloader_prefs.py # DataLoader s/m/l buckets
├── lightning_ckpt.py   # resume epoch / num_epochs peek
├── data/ plan/ model/ pretrain/ train/ infer/ cli/
└── …
```

Entry points: `nanounet_preprocess`, `nanounet_train`, `nanounet_pretrain`, `nanounet_predict` (see [pyproject.toml](pyproject.toml)).

---

## Smoke test

```bash
python -c "import sys; import nanounet.cli.preprocess, nanounet.cli.train, nanounet.cli.predict; assert 'nnunetv2' not in sys.modules; print('ok')"
```
