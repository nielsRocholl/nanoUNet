# Lesion tracking

Graph neural network that matches lesions between a baseline CT and a follow-up CT: dense bipartite BL↔FU edges, pair logits, and dust (no-match) heads. PyTorch Geometric + Lightning; layout follows [nanochat](https://github.com/karpathy/nanochat) style.

**Model:** r9_base — L0 descriptors (`DESC_DIM=1372`), RowMatchability dustbin, bilinear matcher. One canonical config in `configs/base.json`.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Weights & Biases (optional): `wandb login` once.

**Dataset:** `/nnunet_data/Longitudinal-CT/` — `meta/{patient}.csv`, `inputsTrBL/FU`, `targetsTrBL/FU`, `data_split.json`. Tracking split is `configs/split.json` (240 train/val, 60 holdout). Override root with `--root`.

**Cache:** `{CACHE_ROOT}/processed/{split}_v6_h60.pt` (default `/nnunet_data/lesion_tracking/cache`).

**Training vs deployment:** preprocess/train need the CSV (supervision + `cog_propagated`). Deployment: `lesion_track` from CT + instance masks + propagated centroids. `predict.py` is cached-graph benchmark only.

---

## Configuration

All training knobs live in JSON, loaded by `tracking/config.py` (`Config` dataclass, `load_config()`, `dump_config()`). Canonical r9_base values: `configs/base.json`.

| Config key | Role |
|------------|------|
| `max_steps`, `val_check_steps`, `warmup_steps`, `early_stop_patience` | Step-clock training; checkpoint monitors `val_match_score_ema` |
| `lr`, `weight_decay`, `batch_size`, `val_batch_size`, `num_workers`, `seed` | Optimizer / data loading |
| `d`, `layers`, `heads`, `dropout` | GNN architecture |
| `sinkhorn_w`, `pair_w`, `nce_w`, `dust_w`, `dust_pos_w`, `nce_tau`, `sinkhorn_iters` | Loss weights |
| `fu_jitter`, `p_drop_fu`, `p_drop_bl`, `k_intra` | Augmentation + graph kNN (also used by preprocess default) |
| `ema_decay`, `ema_start_step`, `val_score_ema_beta` | Weight EMA + smoothed val score for early stop |
| `dust_tau` | Default decode threshold (sweep at eval time) |
| `n_folds`, `cv_seed` | Patient-level k-fold CV |

**Config-driven CLIs** (`train`, `cv`, `report`): pass `--config configs/base.json`. Training writes a copy to `{out}/config.json`.

**CLI-only overrides:** paths (`--root`, `--cache`, `--out`), W&B flags, fold index (`--fold`), early-stop disable, eval/report device and batch settings. `lesion_track` decode flags: `--decode`, `--thresh`, `--sinkhorn-tau`, `--sinkhorn-iters`.

Copy and edit `configs/base.json` for experiments; unknown keys raise on load.

---

## Pipeline

**1 — Preprocess** (once per dataset; L0 only)

```bash
lesion_track_split
lesion_track_preprocess --split all --jobs 16
lesion_track_train --config configs/base.json --out /nnunet_data/lesion_tracking/runs/h60_r9 --wandb
lesion_track_eval --ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt --split test
```

**Cross-validation** (optional)

```bash
python3 tracking/cli/cv.py --config configs/base.json --out runs/cv --wandb --wandb-run-name r9_base
```

Writes `fold_*/` subdirs + `cv_summary.json` (mean±std over folds on `val_match_score_ema`).

**4 — Round 9 script** (CV → optional final retrain + tau sweep + test gate)

```bash
export RUNS=runs/round9          # optional; default runs/round9
export CONFIG=configs/base.json  # optional
bash scripts/round9.sh
RUN_FINAL=1 bash scripts/round9.sh   # retrain on full train+val, sweep dust_tau on val, eval test once
```

Cluster: `scripts/lesion-round9-cv.sh` (SLURM; sets `RUNS` on `/nnunet_data`).

**Cached-graph eval / predict** (benchmark only)

```bash
lesion_track_eval --ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt --split val
python3 tracking/cli/predict.py --ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt --split val --out preds
```

**Deploy**

```bash
lesion_track \
  --bl-img bl.nii.gz --bl-mask bl_instances.nii.gz \
  --fu-img fu.nii.gz --fu-mask fu_instances.nii.gz \
  --propagated propagated_centroids.csv \
  --ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt \
  --decode dense --out matches.csv
```

---

## CLI reference

After `pip install -e .`, commands are `lesion_track_*`. `--decode` omitted → interactive table (non-TTY must pass `--decode`).

### `lesion_track`

CSV-free inference from CT, instance masks, propagated BL centroids.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-img` `--bl-mask` `--fu-img` `--fu-mask` | path | required | NIfTI |
| `--propagated` | path | required | CSV `lesion_id,z,y,x` (+ optional `lesion_type`) |
| `--ckpt` | path | required | Lightning ckpt |
| `--out` | path | required | matches CSV |
| `--decode` | choice | unset | dense / sinkhorn / hungarian (see help) |
| `--thresh` | float | 0.5 | dense pair cutoff |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` |
| `--k-intra` | int | 8 | intra-graph kNN |
| `--sinkhorn-iters` | int | 20 | |
| `--sinkhorn-tau` | float | 0.2 | |
| `--default-lesion-type` | str | `unclear` | |
| `--no-ema` | flag | off | |
| `--pairs-out` | path | `""` | optional full N×M dump |

Output columns: `bl_lesion_id, fu_lesion_id, pair_prob, decode`.

### `preprocess.py`

Materialize cached L0 PyG graphs from NIfTIs + CSV.

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | (required) | `train` \| `val` \| `test` \| `all` |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--cache` | `CACHE_ROOT` | Graph cache root |
| `--k-intra` | `8` | kNN degree for intra-BL / intra-FU edges |
| `--jobs` | `1` | Parallel patients (`ProcessPoolExecutor`) |
| `--resume` | off | Skip patients already in staging; merge all at end |

### `train.py`

Train matcher on cached train/val graphs (step clock from config).

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | JSON config path |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--cache` | `CACHE_ROOT` | Graph cache root |
| `--out` | `lightning_logs` | Checkpoint dir (`best.ckpt`, `last.ckpt`) |
| `--fold` | none | CV fold index `0 … n_folds-1` |
| `--no-early-stop` | off | Run full `max_steps` |
| `--wandb` | off | Log to W&B |
| `--wandb-project` | `lesion-tracking` | W&B project |
| `--wandb-run-name` | `""` | W&B run name |

Device: MPS when available on Apple Silicon, else Lightning `accelerator=auto`.

### `cv.py`

Patient-level k-fold CV; trains each fold via `train.py`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | JSON config path |
| `--out` | (required) | CV root (`fold_*/`, `cv_summary.json`) |
| `--start-fold` | `0` | First fold (inclusive) |
| `--end-fold` | `n_folds` | Exclusive upper bound |
| `--wandb` … | | Forwarded to per-fold train |

### `eval.py`

Same metrics as training validation, on val or test graphs.

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | (required) | Lightning checkpoint |
| `--split` | `test` | `val` \| `test` |
| `--cache` | `CACHE_ROOT` | Graph cache |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--batch-size` | `8` | Batch size |
| `--num-workers` | `2` | DataLoader workers |
| `--dust-tau` | ckpt value | Override decode threshold |

### `predict.py` (benchmark only)

Cached val/test graphs → CSV. Deployment uses `lesion_track`.

### `report.py`

End-to-end benchmark: train (or load checkpoint), eval GNN + nearest-mask baseline, write `report.json`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | JSON config path |
| `--out` | (required) | Run directory |
| `--checkpoint` | `""` | Skip training; eval from this `.ckpt` |
| `--root`, `--cache` | defaults | Dataset / cache paths |
| `--no-early-stop`, `--no-ema` | off | Training / eval toggles |
| `--wandb` … | | W&B during training |
| `--eval-batch-size` | `1` | GNN eval batch (low RAM) |
| `--eval-num-workers` | `0` | GNN eval workers |
| `--eval-device` | `auto` | `auto` \| `cuda` \| `cpu` \| `mps` |
| `--quiet` | off | No Rich progress during eval |
| `--baseline-full-mask-cache` | off | Keep all FU masks in RAM for baseline |

### `baseline_distance.py`

Distance-only baseline: score = `-dist_mm` on cross edges → AP / AUROC.

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | `val` | `val` \| `test` |
| `--cache`, `--root` | defaults | Paths |
| `--batch-size` | `8` | Batch size |

### `qc.py`

Dash cytoscape viewer for one cached patient graph. Prints URL via `print0`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--case` | (required) | Patient id |
| `--split` | `val` | `train` \| `val` \| `test` |
| `--cache`, `--root` | defaults | Paths |
| `--port` | `8050` | HTTP port |

---

## Further reading

| Doc | Role |
|-----|------|
| [technical.md](technical.md) | Graph construction, model, losses, metrics, deployment checklist |
| [blueprint.md](blueprint.md) | Older design notes — background only |

---

## Repository layout

```
configs/
  base.json             # canonical r9_base training config
tracking/
  config.py             # Config dataclass + JSON load/save
  common.py             # paths, constants, print0, seed
  matcher.py            # encoder + heterogeneous GNN + edge head
  data/                 # graph build, dataset, staging, L0 features, augment
  train/                # Lightning DataModule + Module (module_from_config)
  cli/                  # preprocess, train, cv, predict, eval, qc, report, …
scripts/
  round9.sh             # CV → optional final + tau sweep + test gate
  lesion-round9-cv.sh   # SLURM wrapper for round9.sh
```

Common failures: missing NIfTI/CSV under `--root`, empty BL/FU side, missing `configs/split.json` (`lesion_track_split`), or looking at stale `{split}_v5_l0.pt` (ignored; cache tag is `v6_h60`).
