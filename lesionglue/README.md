# Lesion tracking

Graph neural network that matches lesions between a baseline CT and a follow-up CT: dense bipartite BL↔FU edges, pair logits, and dust (no-match) heads. PyTorch Geometric + Lightning; layout follows [nanochat](https://github.com/karpathy/nanochat) style.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

Weights & Biases (optional): `wandb login` once.

**Dataset:** Longitudinal CT v2 layout — `meta/{patient}.csv`, `inputsTrBL/FU`, `targetsTrBL/FU`, `data_split.json`. Default root is `DATASET_ROOT` in `tracking/common.py`; override with `--root` on every CLI.

**Cache:** preprocessed graphs under `{CACHE_ROOT}/processed/` (default `{DATASET_ROOT}/tracking/processed/`). Files: `{split}_v5_{feat}.pt`, `{split}_v5_{feat}_meta.pt`. While a split builds, per-patient staging lives in `processed/staging/{split}_v5_{feat}/{patient_id}.pt` and is removed after merge.

**Training vs deployment:** benchmark preprocess/train needs the CSV (supervision + `cog_propagated`). At inference you need the same geometry from masks + CT + registration-warped baseline centroids — see [technical.md](technical.md). `predict_masks.py` covers the CSV-free path; `predict.py` uses cached benchmark graphs.

---

## CLI reference

Run from repo root with `PYTHONPATH=. python tracking/cli/<script>.py …`.

### Shared feature flags

Used by any script that calls `add_feat_args` (`preprocess`, `train`, `predict`, `eval`, `qc`, `baseline_distance`, `report`, `predict_masks`).

| Argument | Default | Description |
|----------|---------|-------------|
| `--feat` | `l0` | `l0` \| `mae` \| `yerebakan` — node descriptor mode; cache tag is `v5_{feat}` |
| `--mae-ckpt` | see `features.py` | MAE encoder checkpoint (required when `--feat mae`) |
| `--mae-plans` | see `features.py` | nnUNet plans JSON for MAE ROI geometry |
| `--mae-skip` | `4` | MAE encoder stride |
| `--mae-batch` | `2` | MAE ROI batch size (`--jobs 1` required for MAE preprocess) |

---

### `preprocess.py`

Materialize cached PyG graphs from NIfTIs + CSV.

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | (required) | `train` \| `val` \| `test` \| `all` |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--cache` | `CACHE_ROOT` | Graph cache root (`…/processed/` under here) |
| `--k-intra` | `8` | kNN degree for intra-BL / intra-FU edges |
| `--jobs` | `1` | Parallel patients (`ProcessPoolExecutor`); MAE requires `1` |
| `--resume` | off (flag) | Skip patients already in staging; merge all at end. Fresh run clears staging first |
| `--feat` … | see above | Feature mode |

Delete `{split}_v5_{feat}.pt` before re-preprocessing a split — PyG skips `process()` if the final file exists.

---

### `train.py`

Train the matcher on cached train/val graphs.

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | `DATASET_ROOT` | Dataset root |
| `--cache` | `CACHE_ROOT` | Graph cache root |
| `--out` | `lightning_logs` | Checkpoint dir (`best.ckpt`, `last.ckpt`; monitors `val_match_score`) |
| `--epochs` | `400` | Max epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `1e-2` | Weight decay |
| `--batch-size` | `8` | Patients per batch |
| `--num-workers` | `2` | DataLoader workers |
| `--seed` | `0` | Random seed |
| `--d` | `128` | Hidden dim |
| `--layers` | `4` | GNN layers |
| `--heads` | `4` | Attention heads |
| `--dropout` | `0.2` | Dropout |
| `--sinkhorn-w` | `1.0` | Sinkhorn loss weight |
| `--pair-w` | `0.1` | Pair BCE weight |
| `--nce-w` | `0.3` | Contrastive loss weight |
| `--dust-w` | `0.30` | Dust (no-match) loss weight |
| `--dust-pos-w` | `1.0` | Dust positive class weight |
| `--dust-tau` | `0.2` | Hungarian decode row-normalized threshold |
| `--nce-tau` | `0.1` | NCE temperature |
| `--sinkhorn-iters` | `20` | Sinkhorn iterations |
| `--fu-jitter` | `0.3` | FU position jitter scale (`0` disables) |
| `--p-drop-fu` | `0.10` | Train-time FU node drop prob |
| `--p-drop-bl` | `0.10` | Train-time BL node drop prob |
| `--desc-jitter-frac` | `0.0` | Descriptor jitter fraction |
| `--set-attn-blocks` | `0` | Set-attention blocks (`0` = off) |
| `--ema-decay` | `0.999` | EMA decay (`0` disables) |
| `--ema-start` | `5` | Epoch to start EMA |
| `--tta-n` | `0` | Val TTA passes (`0` = off) |
| `--single-pos-fu` | off (flag) | Legacy one-argmax FU Sinkhorn target (ablation) |
| `--early-stop-patience` | `60` | Early stop on `val_match_score` plateau |
| `--no-early-stop` | off (flag) | Run all `--epochs` |
| `--dust-no-pair-summary` | off (flag) | Ablation: no pair-logit summaries in DustHead |
| `--dust-legacy-linear` | off (flag) | Round-5 linear dust head |
| `--wandb` | off (flag) | Log to W&B (also on if `--wandb-run-name` is set) |
| `--wandb-project` | `lesion-tracking` | W&B project |
| `--wandb-run-name` | `""` | W&B run name (auto if empty) |
| `--feat` … | see above | Must match preprocessed cache |

Device: MPS when available on Apple Silicon, else Lightning `accelerator=auto` (CUDA/CPU). No CLI flag.

---

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
| `--tta-n` | `5` | TTA jitter passes (`0` disables) |
| `--no-ema` | off (flag) | Use training weights instead of EMA |
| `--dump-all` | off (flag) | Write every cross edge |
| `--strict` | off (flag) | Hungarian 1:1 decode for `decoded` column |
| `--feat` … | see above | Must match checkpoint |

Output columns: `bl_lesion_id`, `fu_lesion_id`, `prob`, `decoded`.

---

### `predict_masks.py`

CSV-free inference from CT volumes, instance masks, and propagated baseline centroids.

| Argument | Default | Description |
|----------|---------|-------------|
| `--bl-img` | (required) | Baseline CT NIfTI |
| `--bl-mask` | (required) | Baseline instance mask |
| `--fu-img` | (required) | Follow-up CT NIfTI |
| `--fu-mask` | (required) | Follow-up instance mask |
| `--propagated` | (required) | CSV: `lesion_id,z,y,x` (+ optional `lesion_type`) |
| `--ckpt` | (required) | Lightning checkpoint |
| `--out` | (required) | Best-match CSV (Hungarian decode) |
| `--pairs-out` | `""` | Optional dense pair-prob CSV |
| `--default-lesion-type` | none | Fallback anatomy label |
| `--k-intra` | `8` | Intra-graph kNN |
| `--sinkhorn-iters` | `20` | Decode iterations |
| `--sinkhorn-tau` | `0.2` | Decode temperature |
| `--tta-n` | `5` | TTA passes (`0` disables) |
| `--no-ema` | off (flag) | Use training weights |
| `--feat` … | see above | Must match checkpoint |

---

### `eval.py`

Same validation metrics as training, on val or test graphs (not predict CSVs).

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | (required) | Lightning checkpoint |
| `--split` | `test` | `val` \| `test` |
| `--cache` | `CACHE_ROOT` | Graph cache |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--batch-size` | `8` | Batch size |
| `--num-workers` | `2` | DataLoader workers |
| `--dust-ramp-epoch` | `1000` | Dust ramp epoch for val loss (`1000` ≈ full weight; `-1` = leave unpatched) |
| `--feat` … | see above | Must match checkpoint |

---

### `baseline_distance.py`

Distance-only baseline: score = `-dist_mm` on cross edges → AP / AUROC.

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | `val` | `val` \| `test` |
| `--cache` | `CACHE_ROOT` | Graph cache |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--batch-size` | `8` | Batch size |
| `--feat` … | see above | Cached graph feature mode |

---

### `qc.py`

Dash cytoscape viewer for one cached patient graph. Prints URL via `print0`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--case` | (required) | Patient id (strips trailing `_<digit>`) |
| `--split` | `val` | `train` \| `val` \| `test` |
| `--cache` | `CACHE_ROOT` | Graph cache |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--port` | `8050` | HTTP port |
| `--feat` … | see above | Cached graph feature mode |

---

### `report.py`

End-to-end benchmark: train (or load checkpoint), eval GNN + nearest-mask baseline, write `report.json`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--out` | (required) | Run directory |
| `--checkpoint` | `""` | Skip training; eval from this `.ckpt` |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--cache` | `CACHE_ROOT` | Graph cache (train/val/test `.pt` must exist) |
| `--epochs` | `400` | Training epochs (if not `--checkpoint`) |
| `--batch-size` | `8` | Train batch size |
| `--num-workers` | `2` | Train DataLoader workers |
| `--seed` | `0` | Random seed |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `1e-2` | Weight decay |
| `--d` | `128` | Hidden dim |
| `--layers` | `4` | GNN layers |
| `--heads` | `4` | Attention heads |
| `--dropout` | `0.2` | Dropout |
| `--sinkhorn-w` | `1.0` | Sinkhorn loss weight |
| `--pair-w` | `0.1` | Pair BCE weight |
| `--nce-w` | `0.3` | NCE weight |
| `--dust-w` | `0.30` | Dust loss weight |
| `--dust-pos-w` | `1.0` | Dust positive weight |
| `--nce-tau` | `0.1` | NCE temperature |
| `--sinkhorn-iters` | `20` | Sinkhorn iterations |
| `--fu-jitter` | `0.3` | FU jitter scale |
| `--p-drop-fu` | `0.10` | FU node drop |
| `--p-drop-bl` | `0.10` | BL node drop |
| `--desc-jitter-frac` | `0.0` | Descriptor jitter |
| `--dust-tau` | `0.2` | Decode threshold |
| `--ema-decay` | `0.999` | EMA decay |
| `--ema-start` | `5` | EMA start epoch |
| `--tta-n` | `0` | Val TTA |
| `--early-stop-patience` | `60` | Early stop patience |
| `--no-early-stop` | off (flag) | Disable early stop |
| `--no-ema` | off (flag) | Eval without EMA weights |
| `--wandb` | off (flag) | W&B during training |
| `--wandb-project` | `lesion-tracking` | W&B project |
| `--wandb-run-name` | `""` | W&B run name |
| `--eval-batch-size` | `1` | GNN eval batch (low RAM) |
| `--eval-num-workers` | `0` | GNN eval workers |
| `--eval-device` | `auto` | `auto` \| `cuda` \| `cpu` \| `mps` for GNN eval |
| `--quiet` | off (flag) | No Rich progress during eval |
| `--baseline-full-mask-cache` | off (flag) | Keep all FU masks in RAM for baseline (faster, more RAM) |
| `--feat` … | see above | Feature mode |

---

## Pipeline overview

**1 — Preprocess**

```bash
PYTHONPATH=. python tracking/cli/preprocess.py --split train --feat l0 --jobs 4
PYTHONPATH=. python tracking/cli/preprocess.py --split val --feat l0 --jobs 1
PYTHONPATH=. python tracking/cli/preprocess.py --split test --feat l0 --jobs 1
```

Resume after interrupt:

```bash
PYTHONPATH=. python tracking/cli/preprocess.py --split train --feat l0 --jobs 4 --resume
```

MAE features (requires `--jobs 1`):

```bash
PYTHONPATH=. python tracking/cli/preprocess.py --split all --feat mae --jobs 1 --mae-batch 2
```

**2 — Train**

```bash
PYTHONPATH=. python tracking/cli/train.py \
  --wandb --wandb-project lesion-tracking --wandb-run-name my-run \
  --epochs 200 --out lightning_logs
```

**3 — Predict (benchmark graphs)**

```bash
PYTHONPATH=. python tracking/cli/predict.py --ckpt lightning_logs/best.ckpt --split val --out preds
```

**4 — Predict (masks + registration, no CSV labels)**

```bash
PYTHONPATH=. python tracking/cli/predict_masks.py \
  --bl-img bl.nii.gz --bl-mask bl_instances.nii.gz \
  --fu-img fu.nii.gz --fu-mask fu_instances.nii.gz \
  --propagated propagated_centroids.csv \
  --default-lesion-type unclear \
  --ckpt lightning_logs/best.ckpt --out best_matches.csv
```

---

## Further reading

| Doc | Role |
|-----|------|
| [technical.md](technical.md) | Graph construction, model, losses, metrics, deployment checklist |
| [blueprint.md](blueprint.md) | Older design notes — background only; code is edge classification + PyG + Lightning |

---

## Repository layout

```
tracking/
├── common.py           # paths, constants, print0, seed
├── matcher.py          # encoder + heterogeneous GNN + edge head
├── data/               # graph build, dataset, staging, features, augment
├── train/              # Lightning DataModule + Module
└── cli/                # preprocess, train, predict, eval, qc, report, …
```

Common failures: missing NIfTI/CSV under `--root`, patient skipped (empty BL or FU side), forgot `PYTHONPATH=.`, or stale `{split}_v5_{feat}.pt` blocking re-preprocess.
