# Train

A config JSON ([`lesionglue/config.py`](../../config.py), examples in `lesionglue/configs/`) drives the matcher. `lesionglue_cv` trains one model per patient-level fold, `lesionglue_oof` scores each fold checkpoint on its own held-out patients, and `lesionglue_pool` concatenates those per-patient counts into one out-of-fold estimate with a patient-bootstrap 95% CI.
`lesionglue_train` without `--fold` is the final fit on train+val (no validation, `last.ckpt` only).
All four need the cached lesion graphs (`lesionglue_preprocess`) and `lesionglue/configs/split.json` (`lesionglue_split`).

Config fields (`max_steps`, `n_folds`, `cv_seed`, `ema_decay`, `intra`, ...) are defined once in [`lesionglue/config.py`](../../config.py); unknown keys are rejected. The selection metric is `val_match_score_ema`.

## `lesionglue_train`

### Command

```bash
lesionglue_train --config lesionglue/configs/base.json --out runs/base --fold 0
```

Final fit on train+val, no validation:

```bash
lesionglue_train --config lesionglue/configs/complete.json --out runs/final_seed0 --seed 0
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | required | JSON config (`lesionglue/configs/base.json`) |
| `--root` | str | `/nnunet_data/Longitudinal-CT` | Dataset root read by the datamodule |
| `--cache` | str | `/nnunet_data/lesion_tracking/cache` | Root of the cached lesion graphs |
| `--out` | str | `lightning_logs` | Run dir for checkpoints, `config.json`, `fold_metrics.json`. Refuses to run if it already holds `*.ckpt` |
| `--fold` | int | none | CV fold in `[0, n_folds)` held out for validation. Omitted: no-val fit on train+val, saves `last.ckpt` only |
| `--seed` | int | none | Override the config seed; none uses the seed from `--config` |
| `--max-steps` | int | none | Override `max_steps` (optimizer steps); none uses the config value |
| `--no-early-stop` | flag | off | Disable EarlyStopping on `val_match_score_ema` (only applies with `--fold`) |
| `--wandb` | flag | off | Log to Weights & Biases (needs the `wandb` package; also on when `--wandb-run-name` is set) |
| `--wandb-project` | str | `lesion-tracking` | W&B project name |
| `--wandb-run-name` | str | `""` | W&B run name; a non-blank value also turns on W&B logging |

Device: MPS when available, else Lightning `accelerator=auto`, one device. The holdout `test_patients.csv` is never used for selection.

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `lesionglue/configs/base.json` (`--config`) | JSON | you |
| `$CACHE/processed/train_v7_native.pt`, `val_v7_native.pt` (`--cache`) | torch graph cache | `lesionglue_preprocess` |
| `lesionglue/configs/split.json` | JSON | `lesionglue_split` |
| `runs/base/config.json` | JSON (resolved config, after `--seed`/`--max-steps`) | this step |
| `runs/base/best.ckpt` | Lightning ckpt, best `val_match_score_ema` (fold runs) | this step |
| `runs/base/best_raw.ckpt` | Lightning ckpt, best raw `val_match_score` (fold runs) | this step |
| `runs/base/last.ckpt` | Lightning ckpt, final weights (fold and no-val runs) | this step |
| `runs/base/swa_plateau.ckpt` | Lightning ckpt, SWA weights (written at train end when SWA ran) | this step |
| `runs/base/fold_metrics.json` | JSON (best scores, `selector_ckpts`) | this step |

### Common errors

| Message starts with | Fix |
|---|---|
| `Refusing to overwrite checkpoints in` | Point `--out` at an empty run dir |
| `AssertionError` (no text, from `assert args.fold in range(cfg.n_folds)`) | `--fold` must be in `[0, n_folds)` of `--config` (default 5 folds: 0-4) |
| `unknown config keys:` | Remove the listed keys from the JSON; fields are in `lesionglue/config.py` |
| `intra must be 'knn' or 'complete'` | Set `"intra": "knn"` or `"intra": "complete"` |
| `No tracking split at` | `lesionglue_split` first |
| `No holdout CSV at` | Create `/nnunet_data/Longitudinal-CT/test_patients.csv` (`patient` column) |
| `No graph cache at` | `lesionglue_preprocess --split train`, then `--split val` |
| `wandb requested but not installed.` | `pip install 'wandb>=0.12.10'` or drop `--wandb` / `--wandb-run-name` |
| `No last.ckpt at` | No-val fit ended without weights; rerun into a fresh `--out` |

## `lesionglue_cv`

### Command

```bash
lesionglue_cv --config lesionglue/configs/base.json --out runs/cv --wandb --wandb-run-name r9_base
```

Resume or split folds across jobs (folds with an existing `fold_metrics.json` are reused):

```bash
lesionglue_cv --config lesionglue/configs/base.json --out runs/cv --start-fold 2 --end-fold 4
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | required | Config JSON; sets `n_folds` and seed, passed to each fold's `lesionglue_train` run |
| `--out` | str | required | CV output root; writes `fold_*/` subdirs |
| `--start-fold` | int | 0 | First fold index to run (inclusive) |
| `--end-fold` | int | none (`n_folds`) | Exclusive upper bound |
| `--wandb` | flag | off | Log each fold's training run to W&B |
| `--wandb-project` | str | `lesion-tracking` | W&B project name (used only with `--wandb`) |
| `--wandb-run-name` | str | `""` | Run-name base, fold index appended (`r9_base_fold0`); blank uses `cv` |

Each fold is a `lesionglue_train --fold N` subprocess, run sequentially.

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `lesionglue/configs/base.json` (`--config`) | JSON | you |
| `runs/cv/fold_0/` ... `fold_4/` (`best.ckpt`, `last.ckpt`, ...) | run dirs, as in `lesionglue_train` | `lesionglue_train` |
| `runs/cv/fold_N/fold_metrics.json` | JSON | `lesionglue_train` |
| `runs/cv/cv_summary.json` | JSON (aggregate over folds, `monitor`, `n_folds`) | this step |

### Common errors

| Message starts with | Fix |
|---|---|
| `AssertionError` (no text, from `0 <= start_fold < end <= n_folds`) | Keep `--start-fold` < `--end-fold` <= `n_folds` of `--config` |
| `missing runs/cv/fold_N/fold_metrics.json after fold N` | Fold's train run exited without metrics; rerun that fold and read its log |
| `CalledProcessError` (from a fold subprocess) | Read the `lesionglue_train` error above it (table for `lesionglue_train`) |
| `unknown config keys:` | Same as `lesionglue_train` |

## `lesionglue_oof`

### Command

```bash
lesionglue_oof --ckpt runs/cv/fold_0/best.ckpt --fold 0 --config lesionglue/configs/base.json --out runs/cv/fold_0/oof_best
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | str | required | Checkpoint to validate (`best.ckpt`, `best_raw.ckpt`, `swa_plateau.ckpt`) |
| `--fold` | int | required | CV fold whose held-out patients are scored; in `[0, n_folds)` of `--config` |
| `--config` | str | required | JSON config the checkpoint was trained with (gives `n_folds`, `cv_seed`, graph settings) |
| `--root` | str | `/nnunet_data/Longitudinal-CT` | Dataset root for the datamodule and fold split |
| `--cache` | str | `/nnunet_data/lesion_tracking/cache` | Root of the cached lesion graphs |
| `--out` | str | required | Output dir; `val_per_patient.json` is written here |
| `--dust-tau` | float | 0.125 | Decode threshold, overrides the checkpoint's |
| `--no-ema` | flag | off | Score the raw training weights instead of EMA |

Runs one validation pass on CPU. Name the output dir `oof_<selector>` (`oof_best`, `oof_best_raw`, `oof_swa_plateau`): `lesionglue_pool` reads the selector from that name.

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `runs/cv/fold_0/best.ckpt` (`--ckpt`) | Lightning ckpt | `lesionglue_train` |
| `lesionglue/configs/base.json` (`--config`) | JSON | you |
| `$CACHE/processed/*_v7_native.pt` | torch graph cache | `lesionglue_preprocess` |
| `runs/cv/fold_0/oof_best/val_per_patient.json` | JSON (`match_score`, per-subtype acc, `per_patient` counts) | this step |

### Common errors

| Message starts with | Fix |
|---|---|
| `EMA evaluation requested but this checkpoint has no ema_matcher.` | Add `--no-ema` |
| `AssertionError` (no text, from `assert args.fold in range(cfg.n_folds)`) | `--fold` must be in `[0, n_folds)` of `--config` |
| `fold N val loader leaked non-fold patients` | `--config` (`n_folds`, `cv_seed`) differs from the training run; use the config the checkpoint was trained with |
| `evaluated patient set != fold N val loader` | Same cause; rerun with the matching `--config` and `--fold` |
| `fold N: empty train or val after patient split` | `--fold` has no cached patients; check `--cache` and `--root` |
| `No graph cache at` | `lesionglue_preprocess --split train`, then `--split val` |

## `lesionglue_pool`

### Command

```bash
lesionglue_pool --runs runs/cv --out runs/cv/pool
```

With the registration-flagged vs clean split:

```bash
lesionglue_pool --runs runs/cv --out runs/cv/pool --stratify-registration
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--runs` | str | required | Dir containing `fold_*/oof_<selector>/val_per_patient.json` |
| `--out` | str | required | Output dir; `pool_summary.json` is written here |
| `--root` | str | `/nnunet_data/Longitudinal-CT` | Dataset root, read only for `--stratify-registration` |
| `--stratify-registration` | flag | off | Also report pooled scores split into registration-flagged vs clean patients (reads `--root`) |

Pools per selector across folds; each patient must appear in exactly one fold.

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `runs/cv/fold_*/oof_*/val_per_patient.json` | JSON | `lesionglue_oof` |
| `$ROOT/derivatives/registration_error_table.json` (`--stratify-registration`) | JSON | dataset derivatives |
| `runs/cv/pool/pool_summary.json` | JSON (per selector: `n_patients`, `match_score`, `ci95`, `acc_uc`/`acc_dis`/`acc_new`; `flagged`/`clean` with `--stratify-registration`) | this step |

### Common errors

| Message starts with | Fix |
|---|---|
| `no fold_*/oof_*/val_per_patient.json under` | Run `lesionglue_oof` per fold; `--runs` is the dir holding `fold_*/` |
| `patient '...' appears in both` | Fold leakage across the pooled files; do not mix runs with different `cv_seed` or `n_folds` under one `--runs` |
