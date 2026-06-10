# Lesion tracking

Graph neural network that matches lesions between a baseline CT and a follow-up CT: dense bipartite BL↔FU edges, pair logits, and dust (no-match) heads. PyTorch Geometric + Lightning; layout follows [nanochat](https://github.com/karpathy/nanochat) style.

**Model:** r9_base — L0 descriptors (`DESC_DIM=1372`), RowMatchability dustbin, bilinear matcher. One canonical config in `configs/base.json`.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

Weights & Biases (optional): `wandb login` once.

**Dataset:** Longitudinal CT v2 layout — `meta/{patient}.csv`, `inputsTrBL/FU`, `targetsTrBL/FU`, `data_split.json`. Default root is `DATASET_ROOT` in `tracking/common.py`; override with `--root` on every CLI.

**Cache:** preprocessed graphs under `{CACHE_ROOT}/processed/` (default `{DATASET_ROOT}/tracking/processed/`). Files: `{split}_v5_l0.pt`, `{split}_v5_l0_meta.pt`. While a split builds, per-patient staging lives in `processed/staging/{split}_v5_l0/{patient_id}.pt` and is removed after merge.

**Training vs deployment:** benchmark preprocess/train needs the CSV (supervision + `cog_propagated`). At inference you need the same geometry from masks + CT + registration-warped baseline centroids — see [technical.md](technical.md). `predict_masks.py` covers the CSV-free path; `predict.py` uses cached benchmark graphs.

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

**CLI-only overrides:** paths (`--root`, `--cache`, `--out`), W&B flags, fold index (`--fold`), early-stop disable, eval/report device and batch settings. Inference CLIs (`predict`, `predict_masks`, `eval`) keep operational decode flags (`--dust-tau`, `--sinkhorn-tau`, `--sinkhorn-iters`, etc.).

Copy and edit `configs/base.json` for experiments; unknown keys raise on load.

---

## Pipeline

**1 — Preprocess** (once per dataset; L0 only)

```bash
PYTHONPATH=. python3 tracking/cli/preprocess.py --split all --jobs 4
# or per split:
PYTHONPATH=. python3 tracking/cli/preprocess.py --split train --jobs 4
PYTHONPATH=. python3 tracking/cli/preprocess.py --split val --jobs 1
PYTHONPATH=. python3 tracking/cli/preprocess.py --split test --jobs 1
```

Resume after interrupt:

```bash
PYTHONPATH=. python3 tracking/cli/preprocess.py --split train --jobs 4 --resume
```

Delete `{split}_v5_l0.pt` before re-preprocessing — PyG skips `process()` if the final file exists.

**2 — Train**

```bash
PYTHONPATH=. python3 tracking/cli/train.py \
  --config configs/base.json \
  --out runs/my_run \
  --wandb --wandb-project lesion-tracking --wandb-run-name my-run
```

**3 — Cross-validation**

```bash
PYTHONPATH=. python3 tracking/cli/cv.py \
  --config configs/base.json \
  --out runs/cv \
  --wandb --wandb-run-name r9_base
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

**5 — Eval / predict**

```bash
PYTHONPATH=. python3 tracking/cli/eval.py --ckpt runs/my_run/best.ckpt --split val
PYTHONPATH=. python3 tracking/cli/eval.py --ckpt runs/my_run/best.ckpt --split test --dust-tau 0.20

PYTHONPATH=. python3 tracking/cli/predict.py --ckpt runs/my_run/best.ckpt --split val --out preds
```

**6 — Report** (train or load ckpt → GNN + distance baseline → `report.json`)

```bash
PYTHONPATH=. python3 tracking/cli/report.py \
  --config configs/base.json \
  --out runs/report
# or skip training:
PYTHONPATH=. python3 tracking/cli/report.py \
  --config configs/base.json \
  --out runs/report \
  --checkpoint runs/my_run/best.ckpt
```

**7 — Predict (masks + registration, no CSV labels)**

```bash
PYTHONPATH=. python3 tracking/cli/predict_masks.py \
  --bl-img bl.nii.gz --bl-mask bl_instances.nii.gz \
  --fu-img fu.nii.gz --fu-mask fu_instances.nii.gz \
  --propagated propagated_centroids.csv \
  --default-lesion-type unclear \
  --ckpt runs/my_run/best.ckpt --out best_matches.csv
```

---

## CLI reference

Run from repo root: `PYTHONPATH=. python3 tracking/cli/<script>.py …`.

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

### `predict.py`

Batch inference on cached val/test graphs → one CSV per patient.

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | (required) | Lightning checkpoint |
| `--split` | `val` | `val` \| `test` |
| `--out` | `preds` | Output directory |
| `--cache` | `CACHE_ROOT` | Graph cache |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--thresh` | `0.5` | Min prob to write a row (unless `--dump-all`) |
| `--sinkhorn-iters` | `20` | Decode Sinkhorn iterations |
| `--sinkhorn-tau` | `0.2` | Decode temperature |
| `--no-ema` | off | Use training weights instead of EMA |
| `--dump-all` | off | Write every cross edge |
| `--strict` | off | Hungarian 1:1 decode for `decoded` column |

Output columns: `bl_lesion_id`, `fu_lesion_id`, `prob`, `decoded`.

### `predict_masks.py`

CSV-free inference from CT volumes, instance masks, and propagated baseline centroids.

| Argument | Default | Description |
|----------|---------|-------------|
| `--bl-img`, `--bl-mask`, `--fu-img`, `--fu-mask` | (required) | NIfTI paths |
| `--propagated` | (required) | CSV: `lesion_id,z,y,x` (+ optional `lesion_type`) |
| `--ckpt` | (required) | Lightning checkpoint |
| `--out` | (required) | Best-match CSV (Hungarian decode) |
| `--pairs-out` | `""` | Optional dense pair-prob CSV |
| `--default-lesion-type` | none | Fallback anatomy label |
| `--k-intra` | `8` | Intra-graph kNN |
| `--sinkhorn-iters` | `20` | Decode iterations |
| `--sinkhorn-tau` | `0.2` | Decode temperature |
| `--no-ema` | off | Use training weights |

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

Common failures: missing NIfTI/CSV under `--root`, patient skipped (empty BL or FU side), forgot `PYTHONPATH=.`, or stale `{split}_v5_l0.pt` blocking re-preprocess.
