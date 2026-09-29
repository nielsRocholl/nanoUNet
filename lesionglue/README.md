# Lesion tracking

Graph neural network that matches lesions between a baseline CT and a follow-up CT: dense bipartite BL↔FU edges, pair logits, and dust (no-match) heads. PyTorch Geometric + Lightning; layout follows [nanochat](https://github.com/karpathy/nanochat) style.

**Deployed matcher** — local weights, nothing to download. `lesionglue_track` / `lesionglue_eval` / `nanounet_segtrack` use this unless overridden:

| Knob | Value |
|------|-------|
| checkpoint | `/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` |
| weights | EMA (`--no-ema` to disable) |
| decode | hungarian, `dust_tau=0.125` (`--sinkhorn-tau`) |
| graph | `drop_dp=false`, `intra=complete`, `type_mask=false` (ckpt hparams) |
| holdout | cache_v7 test match **0.9701** (57 graphs) |

L0 descriptors (`DESC_DIM=1372`). R13 v8-region retrain lost the common-set gate (0.9688) and was not deployed. `h60_r9/best.ckpt` was 0.9453. Train experiments still use `lesionglue/configs/*.json`.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Weights & Biases (optional): `wandb login` once.

**Dataset:** `/nnunet_data/Longitudinal-CT/` — `meta/{patient}.csv`, `inputsTrBL/FU`, `targetsTrBL/FU`, `data_split.json`. Tracking split is `lesionglue/configs/split.json` (240 train/val, 60 holdout). Override root with `--root`.

**Cache:** `{CACHE_ROOT}/processed/{split}_v7_native.pt` (default `/nnunet_data/lesion_tracking/cache`). Multi-region graphs live in `/nnunet_data/lesion_tracking/cache_v8_regions` (does not overwrite v7).

**Encoding (keep L0):** pid `16a5cdae36`, 90 lesions. L0+mask_stats 33341 ms CPU; `build_mask_graph` 42107 ms wall; `track()` GPU fwd 19.2 ms. Encoder GAP skipped (hook >40 LOC). GPU is not the bottleneck.

**Training vs deployment:** preprocess/train need the CSV (supervision + `cog_propagated`). Deployment: `lesionglue_track` from CT + instance masks + propagated centroids. `predict.py` is cached-graph benchmark only.

---

## Configuration

All training knobs live in JSON, loaded by `lesionglue/config.py` (`Config` dataclass, `load_config()`, `dump_config()`). Inference defaults live in `lesionglue/common.py` (`DEPLOYED_CKPT`, `DEPLOYED_DUST_TAU`). Train recipes: `lesionglue/configs/base.json` (kNN) and `lesionglue/configs/complete.json` (deployed graph recipe).

| Config key | Role |
|------------|------|
| `max_steps`, `val_check_steps`, `warmup_steps`, `early_stop_patience` | Step-clock training; checkpoint monitors `val_match_score_ema` |
| `lr`, `weight_decay`, `batch_size`, `val_batch_size`, `num_workers`, `seed` | Optimizer / data loading |
| `d`, `layers`, `heads`, `dropout` | GNN architecture |
| `sinkhorn_w`, `pair_w`, `nce_w`, `dust_w`, `dust_pos_w`, `nce_tau`, `sinkhorn_iters` | Loss weights |
| `fu_jitter`, `p_drop_fu`, `p_drop_bl`, `k_intra` | Augmentation + graph kNN (ignored when `intra=complete`) |
| `drop_dp` | Zero the 5 registered `dp/dist` channels in `cross_attr`. Retrain required. |
| `intra` | `"knn"` or `"complete"` (deployed: complete) |
| `type_mask` | Restrict intra edges to the same `lesion_type` |
| `ema_decay`, `ema_start_step`, `val_score_ema_beta` | Weight EMA + smoothed val score for early stop |
| `dust_tau` | Decode dustbin threshold. Inference default **0.125**; train JSON may differ |
| `n_folds`, `cv_seed` | Patient-level k-fold CV |

**Config-driven CLIs** (`train`, `cv`, `report`): pass `--config lesionglue/configs/base.json`. Training writes a copy to `{out}/config.json`.

**CLI-only overrides:** paths (`--root`, `--cache`, `--out`), W&B flags, fold index (`--fold`), early-stop disable, eval/report device and batch settings. `lesionglue_track` decode flags: `--decode`, `--thresh`, `--sinkhorn-tau`, `--sinkhorn-iters`.

Copy and edit `lesionglue/configs/base.json` for experiments; unknown keys raise on load.

---

## Pipeline

**1 — Preprocess** (once per dataset; L0 only)

```bash
lesionglue_split
lesionglue_preprocess --split all --jobs 16
lesionglue_eval --split test
```

**Cross-validation** (optional)

```bash
python3 lesionglue/cli/cv.py --config lesionglue/configs/base.json --out runs/cv --wandb --wandb-run-name r9_base
```

Writes `fold_*/` subdirs + `cv_summary.json` (mean±std over folds on `val_match_score_ema`).

**4 — Round 9 script** (CV → optional final retrain + tau sweep + test gate)

```bash
export RUNS=runs/round9          # optional; default runs/round9
export CONFIG=lesionglue/configs/base.json  # optional
bash lesionglue/scripts/round9.sh
RUN_FINAL=1 bash lesionglue/scripts/round9.sh   # retrain on full train+val, sweep dust_tau on val, eval test once
```

Cluster: `lesionglue/scripts/lesion-round9-cv.sh` (SLURM; sets `RUNS` on `/nnunet_data`).

**Cached-graph eval / predict** (benchmark only)

```bash
lesionglue_eval --split val
python3 lesionglue/cli/predict.py --split val --out preds
```

**Deploy**

`--propagated` is BL lesion_id → centroid in the **FU voxel grid**. Accepts `meta/{pid}.csv` (`cog_propagated`), slim CSV `lesion_id,z,y,x`, or nanoUNet JSON in the FU frame. Not `inputsTrBL/*.json` (BL-native). Omit it when the checkpoint has `drop_dp`. `--bl-clicks` only instance-labels binary FG.

Single case:

```bash
lesionglue_track \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/0a09c8844b_00.nii.gz \
  --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/0a09c8844b_00.nii.gz \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/0a09c8844b_00.nii.gz \
  --fu-mask /nnunet_data/Longitudinal-CT/targetsTrFU/0a09c8844b_00.nii.gz \
  --propagated /nnunet_data/Longitudinal-CT/meta/0a09c8844b.csv \
  --out matches.csv
```

Holdout folder:

```bash
lesionglue_track \
  --root /nnunet_data/Longitudinal-CT --split test \
  --out /tmp/track_test
```

---

## CLI reference

After `pip install -e .`, commands are `lesionglue_*`. Decode defaults to hungarian (the holdout gate). `--decode dense` and `--decode sinkhorn` keep merges/splits (sinkhorn up to about `1/--sinkhorn-tau` lesions per merge; see `lesionglue/model/decode.py`). `lesionglue/cli/report.py --decode` scores any of the three.

### `lesionglue_track`

CSV-free inference from CT and instance masks. Geo checkpoints also need propagated BL centroids. Single case or `--root` dataset.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-img` `--bl-mask` `--fu-img` `--fu-mask` | path | required in single | NIfTI |
| `--propagated` | path | required in single unless `drop_dp` ckpt | meta CSV / slim CSV / FU-frame JSON (not inputsTrBL clicks) |
| `--types-csv` | path | unset | `lesion_id,lesion_type`; needed for `type_mask` unless `--default-lesion-type` is set |
| `--root` | path | unset | Longitudinal-CT root → dataset mode |
| `--split` | choice | unset | `train` \| `val` \| `test` from `lesionglue/configs/split.json` |
| `--patients-csv` | path | unset | CSV column `patient`; xor with `--split` |
| `--bl-mask-dir` `--fu-mask-dir` | path | `targetsTr*` | instance-mask override |
| `--prop-dir` | path | `meta/` | `{pid}.csv` or `{pid}_{idx}.json` |
| `--ckpt` | path | `v7_complete/last.ckpt` | Lightning ckpt |
| `--out` | path | required | file (single) or dir `{pid}.csv` (dataset) |
| `--decode` | choice | `hungarian` | hungarian / dense / sinkhorn (see help) |
| `--thresh` | float | 0.5 | dense pair cutoff only |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` |
| `--k-intra` | int | 8 | intra-graph kNN |
| `--sinkhorn-iters` | int | 20 | |
| `--sinkhorn-tau` | float | 0.125 | dustbin tau (deployed) |
| `--default-lesion-type` | str | `unclear` | used when propagated JSON has no type |
| `--no-ema` | flag | off | |
| `--pairs-out` | path | `""` | optional full N×M dump (single) |
| `--bl-clicks` `--fu-clicks` | path | unset | instance JSON; treat that side's mask as binary FG |

Output columns: `bl_lesion_id, fu_lesion_id, pair_prob, decode`.

### `preprocess.py`

Materialize cached L0 PyG graphs from NIfTIs + CSV.

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | (required) | `train` \| `val` \| `test` \| `all` |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--cache` | `CACHE_ROOT` | Graph cache root |
| `--k-intra` | `8` | kNN degree for intra-BL / intra-FU edges |
| `--jobs` | `1` | Parallel patients (`ProcessPoolExecutor`); each worker pins BLAS/OpenMP to 1 thread |
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
| `--seed` | config | Override `Config.seed` before `seed_all` |
| `--max-steps` | config | Override `Config.max_steps` |
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
| `--ckpt` | `v7_complete/last.ckpt` | Lightning checkpoint |
| `--split` | `test` | `val` \| `test` |
| `--cache` | `CACHE_ROOT` | Graph cache |
| `--root` | `DATASET_ROOT` | Dataset root |
| `--batch-size` | `8` | Batch size |
| `--num-workers` | `2` | DataLoader workers |
| `--dust-tau` | `0.125` | Repeatable decode threshold; JSON records every value plus the selected tau |
| `--no-ema` | off | Use raw `matcher` weights (default is EMA) |
| `--out` | unset | Optional JSON path for counts + selected tau |

### `predict.py` (benchmark only)

Cached val/test graphs → CSV. Deployment uses `lesionglue_track`.

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
| [technical.md](docs/technical.md) | Graph construction, deployed matcher, losses, metrics |
| [blueprint.md](blueprint.md) | Older design notes — background only |

---

## Repository layout

```
lesionglue/configs/
  base.json             # kNN train recipe
  complete.json         # geo + complete intra (deployed graph recipe)
lesionglue/
  config.py             # Config dataclass + JSON load/save
  common.py             # paths, DEPLOYED_CKPT / DUST_TAU, print0, seed
  matcher.py            # encoder + heterogeneous GNN + edge head
  data/                 # graph build, dataset, staging, L0 features, augment
  train/                # Lightning DataModule + Module (module_from_config)
  cli/                  # preprocess, train, cv, predict, eval, qc, report, …
scripts/
  round9.sh             # CV → optional final + tau sweep + test gate
  lesion-round9-cv.sh   # SLURM wrapper for round9.sh
```

Common failures: missing NIfTI/CSV under `--root`, empty BL/FU side, missing `lesionglue/configs/split.json` (`lesionglue_split`), or looking at stale `{split}_v6_h60.pt` (ignored; cache tag is `v7_native`).
