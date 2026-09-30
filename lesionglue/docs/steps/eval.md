# Evaluate

Score a matcher checkpoint on the cached val/test lesion graphs: training-time metrics (`lesionglue_eval`), an end-to-end
report against the nearest-mask baseline (`lesionglue_report`), per-patient edge CSVs (`lesionglue_predict`), and a
distance-only floor (`lesionglue_baseline_distance`). Needs `lesionglue_preprocess` graphs; deployment is `lesionglue_track`.

The deployed matcher is `DEPLOYED_CKPT` (`/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt`, EMA, hungarian,
`dust_tau=0.125`): holdout cache_v7 test match **0.9701** (57 graphs).

## lesionglue_eval

Same `val_match_score` counts as training validation, one pass per `--dust-tau`; picks the best tau.

### Command

```bash
lesionglue_eval --split test --root /nnunet_data/Longitudinal-CT --dust-tau 0.10 --dust-tau 0.125 --dust-tau 0.15 --out runs/eval_test.json
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | path | `DEPLOYED_CKPT` | Matcher Lightning checkpoint |
| `--split` | choice | `test` | `val` or `test` cached split |
| `--cache` | path | `CACHE_ROOT` (`/nnunet_data/lesion_tracking/cache`) | Root of the cached lesion graphs |
| `--root` | path | `DATASET_ROOT` (`/nnunet_data/Longitudinal-CT`) | Dataset root passed to `LesionDataset` |
| `--batch-size` | int | 8 | Graphs per validation batch |
| `--num-workers` | int | 2 | DataLoader workers; 0 loads in the main process |
| `--dust-tau` | float, repeatable | `0.125` (`DEPLOYED_DUST_TAU`) | Decode threshold; one validate pass per value |
| `--no-ema` | flag | off | Score the raw training weights instead of EMA |
| `--out` | path | `""` | JSON path for counts and the selected tau; empty prints only |

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/lesion_tracking/cache/processed/test_v7_native.pt` | PyG cache | `lesionglue_preprocess` |
| `--ckpt` | Lightning `.ckpt` | `lesionglue_train` |
| `--out` (`runs/eval_test.json`) | JSON: `ckpt`, `split`, `weights`, `rows[]` (per tau: counts, `match_score`, `metrics`), `selected` | this step |

The selected tau is the highest `match_score`, ties broken toward 0.10. The last line prints a `next:` `lesionglue_track` command.

### Common errors

| Message starts with | Fix |
|---|---|
| `No checkpoint at` | `--ckpt /nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` |
| `No tracking split at` | `lesionglue_split --root /nnunet_data/Longitudinal-CT --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out lesionglue/configs/split.json` |
| `zero graphs for split=` | Split has no patients: rerun `lesionglue_preprocess` for `--split test` |

## lesionglue_report

Trains (or loads a checkpoint), evaluates the matcher on val and test, scores the nearest-mask baseline, writes `report.json`.

### Command

```bash
lesionglue_report --config lesionglue/configs/base.json --root /nnunet_data/Longitudinal-CT --out runs/report_v7 --checkpoint /nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt --decode hungarian
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | path | required | Training config JSON (`lesionglue/configs/*.json`); builds model and trainer |
| `--root` | path | `DATASET_ROOT` | Longitudinal-CT dataset root |
| `--cache` | path | `CACHE_ROOT` | Cached graph dir; needs `processed/{train,val,test}_v7_native.pt` |
| `--out` | path | required | Output dir for `report.json` and `checkpoints/` (created if missing) |
| `--checkpoint` | str | `""` | Skip training, only evaluate this `.ckpt` |
| `--no-early-stop` | flag | off | Train to `max_steps` without EarlyStopping |
| `--no-ema` | flag | off | Evaluate the raw weights instead of EMA |
| `--wandb` | flag | off | Log training to Weights & Biases (also on with `--wandb-run-name`) |
| `--wandb-project` | str | `lesion-tracking` | W&B project; used only when W&B logging is on |
| `--wandb-run-name` | str | `""` | W&B run name; non-empty turns W&B on |
| `--eval-batch-size` | int | 1 | Graphs per val/test evaluation batch |
| `--eval-num-workers` | int | 0 | Val/test DataLoader workers; 0 = main process |
| `--eval-device` | choice | `auto` | `auto`, `cuda`, `cpu` or `mps`; auto picks cuda, then mps, then cpu |
| `--quiet` | flag | off | Hide the evaluation progress bars |
| `--baseline-full-mask-cache` | flag | off | Keep every FU mask index of the baseline in memory (default keeps one) |
| `--decode` | choice | `hungarian` | `dense`, `sinkhorn` or `hungarian` (see [track.md](track.md)) |
| `--thresh` | float | 0.5 | Pair probability cutoff for `--decode dense` |

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/lesion_tracking/cache/processed/{train,val,test}_v7_native.pt` | PyG cache | `lesionglue_preprocess` |
| `runs/report_v7/checkpoints/*.ckpt` | Lightning `.ckpt` (only without `--checkpoint`) | this step |
| `runs/report_v7/report.json` | JSON: `best_checkpoint`, `config`, `decode`, `gnn.{val,test}`, `baseline.{val,test}` | this step |

Each of the four blocks holds `merge_acc`, `split_acc`, `newly_appeared_acc`, `disappeared_acc`, `row_acc_micro`,
`edge_acc_micro`, `positive_edge_{recall,precision,f1}`, and `tp/fp/tn/fn`. A metric with zero support is `null`.

### Common errors

| Message starts with | Fix |
|---|---|
| `missing cached` | `lesionglue_preprocess` for the named split, or point `--cache` at a dir with all three splits |
| `checkpoint not found:` | `--checkpoint /nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` |
| `wandb requested but not installed` | `pip install wandb`, or drop `--wandb` / `--wandb-run-name` |
| `unknown config keys:` | Remove the listed keys from `--config`, or use `lesionglue/configs/base.json` |
| `--device cuda but CUDA not available.` | `--eval-device cpu` (the message names `--device`, it comes from `--eval-device cuda`) |

## lesionglue_predict

Cached-graph benchmark only: one CSV per patient with every edge above `--thresh`. Not deployment: use `lesionglue_track`.

### Command

```bash
lesionglue_predict --split val --root /nnunet_data/Longitudinal-CT --out runs/preds_val --strict
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | path | `DEPLOYED_CKPT` | Matcher Lightning checkpoint |
| `--cache` | path | `CACHE_ROOT` | Root of the cached lesion graphs |
| `--root` | path | `DATASET_ROOT` | Dataset root passed to `LesionDataset` |
| `--split` | choice | `val` | `val` or `test` cached split |
| `--out` | path | `preds` | Output dir, one `<patient>.csv` per patient |
| `--thresh` | float | 0.5 | Edge probability cutoff in [0, 1]; lower edges are not written (unless `--dump-all`) |
| `--sinkhorn-iters` | int | 20 | Sinkhorn iterations for `--strict` |
| `--sinkhorn-tau` | float | `0.125` (`DEPLOYED_DUST_TAU`) | Min row-normalised transport mass to accept a match in `--strict` |
| `--no-ema` | flag | off | Use training weights instead of the EMA shadow |
| `--dump-all` | flag | off | Write every candidate edge, ignoring `--thresh` |
| `--strict` | flag | off | `decoded` from a 1-to-1 Sinkhorn + Hungarian assignment instead of `prob >= --thresh` |

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/lesion_tracking/cache/processed/val_v7_native.pt` | PyG cache | `lesionglue_preprocess` |
| `runs/preds_val/<patient>.csv` | CSV: `bl_lesion_id,fu_lesion_id,prob,decoded` | this step |

### Common errors

| Message starts with | Fix |
|---|---|
| `No checkpoint at` | `--ckpt /nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` |
| `No tracking split at` | `lesionglue_split --root /nnunet_data/Longitudinal-CT --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out lesionglue/configs/split.json` |
| `zero graphs for split=` | Split has no patients: rerun `lesionglue_preprocess` |

## lesionglue_baseline_distance

Distance-only floor: score every cross edge by `-dist_mm`, report AP and AUROC.

### Command

```bash
lesionglue_baseline_distance --split test --root /nnunet_data/Longitudinal-CT
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cache` | path | `CACHE_ROOT` | Cached graph root (output of `lesionglue_preprocess`) |
| `--root` | path | `DATASET_ROOT` | Longitudinal-CT root passed to the graph dataset |
| `--split` | choice | `val` | `val` or `test` cached split |
| `--batch-size` | int | 8 | Graphs per batch while scoring |

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/lesion_tracking/cache/processed/test_v7_native.pt` | PyG cache | `lesionglue_preprocess` |
| stdout line `distance_baseline split=test AP=... AUROC=...` | text | this step (no file is written) |

### Common errors

| Message starts with | Fix |
|---|---|
| `No tracking split at` | `lesionglue_split --root /nnunet_data/Longitudinal-CT --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out lesionglue/configs/split.json` |
| `zero graphs for split=` | Split has no patients: rerun `lesionglue_preprocess` |
