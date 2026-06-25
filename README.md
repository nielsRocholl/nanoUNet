# nanoUNet

Minimal prompt-aware 3D ResEnc nnU-Net with PyTorch Lightning and optional MAE pretraining. Layout and style follow [nanochat](https://github.com/karpathy/nanochat): small modules, no framework sprawl.

## Getting started

```bash
python -m pip install -e .
```

Environment:

```bash
export NANOUNET_RAW="/path/to/NanoUNet_raw"
export NANOUNET_PREPROCESSED="/path/to/NanoUNet_preprocessed"
export NANOUNET_RESULTS="/path/to/NanoUNet_results"

# Host-RAM / checkpoint staging (see docs/cgroup_memory.md)
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp   # local disk; not /tmp (tmpfs) or CIFS (breaks workers)
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
| `-f`, `--fold` | `0` | Fold index 0–4 (default 5-fold split), or `all` for full-data training (val set = train set, in-sample metrics) |
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
| `--loss`, `-loss` | `dc_ce` | `dc_ce` \| `cc_dc_ce` (CC-DiceCE is much slower; see [docs/losses.md](docs/losses.md)) |
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
| `--dl-bucket` | `m` | `s`: 2/1 workers when TMPDIR is off tmpfs, else 0. `m`: 4/2. `l`: 8/4. Case-sticky Blosc2 I/O, no mmap. |
| `--dl-persistent-workers` | off (flag) | Keep DataLoader workers between epochs (recommended for long MAE with workers). |
| `--mem-diag` | off (flag) | Log cgroup/process RAM to `<run>/mem_diag.jsonl` and W&B `mem/*`. Requires `NANOUNET_ALLOW_ROOT_CGROUP=1` on interactive nodes. See [docs/cgroup_memory.md](docs/cgroup_memory.md). |

Checkpoints: `<run>/checkpoints/last.ckpt` (supervised); `<run>/mae_pretrain/checkpoints/` (integrated MAE). Re-running without `--resume` / `--mae-resume` trains from scratch and may overwrite those paths.

#### Loss throughput warning

Use `--loss dc_ce` for normal long supervised training. `--loss cc_dc_ce` runs CPU connected components plus SciPy Euclidean-distance Voronoi inside the training loss and can make epochs roughly 4x slower on A100/H200 nodes. Treat CC-DiceCE as an opt-in experiment or short fine-tuning objective after validating the speed/quality trade-off.

#### Host RAM / cgroup OOM

Long MAE runs were killed by cgroup OOM from **checkpoint temp files on RAM-backed `/tmp`**, not GPU or Python heap. Full write-up: **[docs/cgroup_memory.md](docs/cgroup_memory.md)**.

**MAE long runs (recommended):** Set `NANOUNET_TMPDIR` to local disk (not tmpfs/CIFS), use `--dl-bucket m` or `l` with workers, and add `--dl-persistent-workers` to avoid respawning workers each epoch. Expect ~98 MB/epoch cgroup shmem growth with workers; monitor with `--mem-diag`. Escape hatch: `NANOUNET_DL_FORCE_NO_WORKERS=1` forces `num_workers=0` (~0 shmem/ep, ~3–4 min/epoch).

Quick checks with `--mem-diag`: `cgroup_shmem_delta` ≈ 0 per epoch, `fadvise_calls` rising, `/tmp` and TMPDIR not accumulating orphan ~779 MB files. Set `NANOUNET_TMPDIR=/root/.cache/nanounet_tmp` (local zfs; not `$HOME` or CIFS).

Slurm example: [`scripts/slurm_nanounet_train_mae_999.sh`](scripts/slurm_nanounet_train_mae_999.sh).

---

### `nanounet_pretrain`

Standalone MAE only (no prompts). Default out: `$NANOUNET_RESULTS/nanounet/<DatasetFolder>_<plans>_mae_pretrain_f<fold>/`.

| Argument | Default | Description |
|----------|---------|-------------|
| `-d`, `--dataset_id` | (required) | Dataset id |
| `-f`, `--fold` | `0` | Fold index 0–4, or `all` for full-data training (val set = train set, in-sample metrics) |
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
| `--dl-bucket` | `m` | `s`: 2/1 workers when TMPDIR is off tmpfs, else 0. `m`: 4/2. `l`: 8/4. |
| `--dl-persistent-workers` | off (flag) | Keep DataLoader workers between epochs (recommended for long MAE with workers). |
| `--resume` | none | MAE Lightning ckpt; must exist. No auto `last.ckpt`. Epoch target must match `--epochs` |
| `--precision` | `16-mixed` | Lightning precision |
| `--accelerator` | `auto` | `auto` \| `cpu` \| `cuda` \| `gpu` \| `mps` |

---

### `nanounet_predict`

Prompt-driven GPU-batched inference over a **dataset folder** or a **single case**. Points come
from JSON (native scanner voxel `(x,y,z)`). All of a case's points are clustered ("infer all") and
each cluster is one batched seed forward. TTA on by default (from `nano_config.json`).

Per-patch predictions are combined with `--merge max` (default): each voxel takes the logits of the
patch most confident it is foreground (a **union** of the per-patch segmentations). This matches
running the model once per click and unioning the results. The legacy `--merge average` (gaussian
mean) silently erased lesions when many patches overlapped — the patch that owned a lesion was
outvoted by the dozens of neighbouring patches that (correctly) predict background there.

**Dataset mode** (`-i` is a folder): segments every `*.nii.gz`, each paired with a sibling
`<name>.json` (same basename). Writes `<name>.nii.gz` to `-o`. Missing JSON for any scan → error.

**Single mode** (`-i` is a `.nii.gz`): requires `--points`; writes the single `-o`.

Points JSON ("Points of interest" format): `{"points": [{"name": "1", "point": [x, y, z]}, ...]}`,
voxel `(x,y,z)` in the scan's own grid. Empty `points` → all-background output.

| Argument | Default | Description |
|----------|---------|-------------|
| `-i`, `--input` | (required) | Folder (dataset) **or** single `.nii.gz` |
| `-o`, `--output` | (required) | Output folder (dataset) **or** single `.nii.gz` |
| `-m`, `--model-dir` | (required) | Run dir with `plans.json`, `dataset.json`, `nano_config.json`, checkpoint |
| `--ckpt` | auto | Checkpoint name/path; `auto` → `checkpoints/last.ckpt`. Pass `last.ckpt` if it sits in the run-dir root |
| `--points` | none | Points JSON (**single mode only**) |
| `--no-prompt-encode` | off (flag) | Zero the 2 prompt channels (point encoding off) |
| `--border-expand` | off (flag) | Large-lesion extra border patches (per-cluster BFS, [infer/border_expand.py](nanounet/infer/border_expand.py)) |
| `--max-border-extra` | `16` | Max extra patches per cluster for border expand |
| `--tta` / `--disable-tta` | from `nano_config` | Force test-time augmentation on / off |
| `--batch-size` | `8` | GPU seed mini-batch (patches per forward) |
| `--num-workers` | `4` | CPU preprocess prefetch threads (dataset mode) |
| `--cluster-margin-frac` | `0.1` | Cluster bbox margin as fraction of patch size |
| `--inference-mode` | `clustered` | `clustered` \| `centered` — patch placement (one patch per click in centered mode) |
| `--merge` | `max` | Cross-patch merge. `max` = union: each voxel takes the logits of the patch most confident it is foreground, so a lesion segmented by its own patch is never overvoted by neighbouring background-predicting patches (reproduces interactive single-click per point). `average` = legacy gaussian-weighted mean (suffers cross-patch washout when many patches overlap) |
| `--device` | `cuda` | `cuda` \| `cpu` \| `mps` (falls back to `cpu` if unavailable) |
| `--no-amp` | off (flag) | Disable autocast (exact fp32; CUDA AMP is on by default) |
| `--overwrite` | off (flag) | Re-run cases whose output exists (default: skip = resume) |

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

Set `NANOUNET_TMPDIR` (see Environment). For long MAE runs use `--dl-persistent-workers` and `--mem-diag`; see [docs/cgroup_memory.md](docs/cgroup_memory.md).

**2 — Train**

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --config configs/default.json
```

First run writes `splits_final.json` (5-fold, fixed seed). Use `-f all` to train on every case (no holdout); val metrics are in-sample, so use `checkpoints/last.ckpt` for deployment, not `best-*`. MAE options: `--mae-pretrain` / `--mae-ckpt`, standalone `nanounet_pretrain`, encoder transfer ([Wald et al.](https://arxiv.org/abs/2410.23132)). Resume flags: **CLI reference** (`--resume`, `--mae-resume`). On Slurm, prefer `--dl-bucket s` if host RAM is tight.

Tiny laptop smoke:

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetTinyPlans --config configs/default.json \
  --epochs 2 --iters-per-epoch 50 --accelerator cpu --precision 32-true --batch-size 1 --no-wandb
```

**3 — Predict**

```bash
# Dataset: folder of *.nii.gz + sibling *.json -> one seg per case
nanounet_predict -i /path/to/scans -o /path/to/out -m /path/to/run \
  --ckpt last.ckpt --border-expand --batch-size 8 --device cuda

# Single case
nanounet_predict -i case.nii.gz -o seg.nii.gz --points case.json -m /path/to/run --ckpt last.ckpt

# centered: one patch per click, each prompt = its own click (lesion-centered)
nanounet_predict -i case.nii.gz -o seg.nii.gz --points case.json \
  -m /path/to/run --ckpt last.ckpt --inference-mode centered --border-expand
```

Points are native scanner voxels `(x,y,z)`; mapping to preprocessed space is automatic.

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
