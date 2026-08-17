# Supervised training

Prompt-aware supervised training on one fold. Optional integrated MAE pretrain, longitudinal two-stream finetune, and MAE encoder transfer.

Default run dir: `$NANOUNET_RESULTS/nanounet/<DatasetFolder>_<plans>_f<fold>/`.

## Command

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --config configs/default.json
```

Integrated MAE then supervised:

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --config configs/default.json \
  --mae-pretrain --dl-persistent-workers
```

Reuse an existing self-supervised checkpoint instead of rerunning MAE (`--mae-ckpt` alone, no
`--mae-pretrain`, skips the MAE stage entirely — see
[`scripts/slurm_nanounet_pretrain_train_999.sh`](../../scripts/slurm_nanounet_pretrain_train_999.sh)):

```bash
nanounet_train -d 999 -f 0 --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --mae-ckpt /nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt \
  --dl-persistent-workers
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int | (required) | Dataset id (`DatasetXXX_*` under raw/preprocessed) |
| `-f`, `--fold` | int \| `all` | `0` | Fold 0–4, or `all` for full-data (val = train, in-sample metrics) |
| `--plans` | str | (required) | Plans basename without `.json` |
| `--config` | str | `configs/default.json` | ROI / prompt JSON; relative path tries cwd then repo root |
| `--val-manifest` | str | none | Fixed validation manifest from `nanounet_build_valset`; omit for legacy per-epoch random val sampling. See [valset.md](valset.md) |
| `--val-every-n-epochs` | int | 1 | Validate every N epochs. With a fixed `--val-manifest` the per-epoch resampling noise is gone, so `2` costs ~3% of run time instead of ~18% and still gives 600 points over 1200 epochs |
| `--epochs` | int | `1000` | Supervised epoch budget |
| `--lr` | float | `0.01` | Supervised initial learning rate |
| `--wd` | float | `3e-5` | Weight decay |
| `--optimizer` | choice | `sgd` | `sgd` \| `adamw` |
| `--grad-clip` | float | `0.0` | Max grad norm; 0 disables |
| `--batch-size` | int | from plans | Override `3d_fullres.batch_size` |
| `--iters-per-epoch` | int | `250` | Training batches per epoch |
| `--val-iters` | int | `50` | Validation batches per epoch |
| `--out` | str | auto | Override run directory |
| `--lr-schedule` | choice | `poly` | `poly` \| `stretched_tail_poly` |
| `--stretched-k` | int | `750` | For `stretched_tail_poly` |
| `--stretched-ref` | int | `1000` | For `stretched_tail_poly` |
| `--stretched-exp` | float | `0.9` | For `stretched_tail_poly` |
| `--no-wandb` | flag | off | Disable W&B |
| `--wandb-project` | str | `nanounet` | W&B project |
| `--wandb-name` | str | auto | W&B run name |
| `--loss`, `-loss` | choice | `dc_ce` | `dc_ce` \| `cc_dc_ce` — see [losses.md](../reference/losses.md) |
| `--resume` | str | none | Supervised Lightning ckpt; no auto `last.ckpt` |
| `--init-weights` | str | none | Load full net from supervised ckpt; fresh optimizer |
| `--only-prefix` | str | none | Train/val only case keys with prefix, e.g. `d013_` |
| `--longi` | flag | off | Two-stream BL+FU encoder (requires `--init-weights`) |
| `--longi-null` | flag | off | Ablation: duplicate-FU baseline (requires `--longi`) |
| `--precision` | str | `16-mixed` | Lightning precision |
| `--accelerator` | choice | `auto` | `auto` \| `cpu` \| `cuda` \| `gpu` \| `mps` |
| `--mae-ckpt` | str | none | Load encoder weights only (no integrated MAE run) |
| `--mae-pretrain` | flag | off | Run MAE under `<run>/mae_pretrain/` then supervised |
| `--mae-resume` | str | none | With `--mae-pretrain`: MAE ckpt; conflicts with `--mae-ckpt` |
| `--mae-epochs` | int | `1000` | MAE epoch budget with `--mae-pretrain` |
| `--mae-lr` | float | `1e-2` | MAE initial LR |
| `--mae-lr-schedule` | choice | `cosine_warm_restarts` | MAE LR schedule |
| `--mae-cosine-t0` | int | `250` | MAE cosine T0 |
| `--mae-cosine-t-mult` | int | `1` | MAE cosine T mult |
| `--mae-cosine-eta-min` | float | `0.0` | MAE cosine eta min |
| `--mae-mask-ratio` | float | `0.75` | MAE mask ratio |
| `--mae-iters-per-epoch` | int | same as train | MAE batches per epoch |
| `--dl-bucket` | choice | `m` | DataLoader worker preset: `s` / `m` / `l` / `xl` |
| `--dl-persistent-workers` | flag | off | Keep workers between epochs |
| `--prompts-per-patch` | int | `1` | Independent click draws rendered per patch (same CT crop + augmentation); 2 enables the consistency loss below |
| `--consistency-weight` | float | `0.0` | Lambda max for the two-prompt consistency term; `0` disables it. Requires `--prompts-per-patch >= 2` |
| `--consistency-warmup-epochs` | int | `50` | Epochs to linearly ramp lambda from 0 to `--consistency-weight` |
| `--warmup-epochs` | int | `0` | Linear LR warmup over the first N epochs, applied to either `--lr-schedule`. `0` reproduces the pre-warmup LR curve exactly |
| `--ema-decay` | float | `0.0` | Weight EMA decay (e.g. `0.999`); logs `val_dice_ema` next to `val_dice`. `0` disables EMA. Predict the shadow with `nanounet_predict --ema`. |
| `--monitor` | str | `val_dice` | Metric `ModelCheckpoint` tracks for `best-*.ckpt` |

Checkpoints: `<run>/checkpoints/` (supervised); `<run>/mae_pretrain/checkpoints/` (integrated MAE). Finetune with `--init-weights` writes to `<run>/finetune/`.

## Two-prompt consistency

`--prompts-per-patch 2 --consistency-weight <lambda>` trains each patch with two independently
drawn click sets sharing the same CT crop and augmentation pass, and penalises disagreement
between the two predictions (1 − soft Dice between their foreground-probability maps, finest
resolution only). This directly targets click-position sensitivity: the same lesion clicked in two
places should not swing per-lesion Dice.

`--batch-size` must be divisible by `--prompts-per-patch`. `train_loss_seg` and
`train_loss_consistency` are logged separately. `val_dice_prompt_ablated` (prompt-heatmap channels
zeroed) is logged beside `val_dice` on every validation epoch — if that gap closes, the net (or the
consistency term) is learning to ignore the prompt, and `--consistency-weight` is too high.

## Prompt-robustness validation metrics

Logged every validation epoch alongside `val_dice` / `val_dice_macro` / `val_fp`, always on (no
flag needed) — they measure whether prompt-consistency training is actually working:

| Metric | What it measures | Direction |
|--------|-------------------|-----------|
| `val_prompt_agreement` | Dice between two predictions on the *same* val patch under two independently-drawn clicks (NOT vs. ground truth). The headline number — the project's own sweep measured 0.588–0.605 pre-fix. | higher is better, target → 1.0 |
| `val_dice_click_inside` | Per-lesion Dice on rows where the drawn click landed on foreground (post-augmentation). | — |
| `val_dice_click_outside` | Same, for rows where the click missed (models the 52% deployment case, vs. 88% during old training). | should stay close to `_inside` |
| `val_prompt_gap` | `val_dice − val_dice_prompt_ablated`. | trending to 0 = net has stopped using the click, lower `--consistency-weight` |

Implementation notes:
- `val_prompt_agreement` costs a **3rd forward pass** during validation (normal + prompt-ablated +
  2nd-prompt), computed only because the val dataloader renders one extra prompt variant per patch
  (`emit_prompt2`, its own RNG stream, never perturbing the patch/prompt sequence that produces
  `val_dice`). Validation is 50 iterations, so this is cheap; it does not touch training throughput.
- Foreground = argmax over classes (not a probability threshold). A row where both predictions (or,
  for the click split, the row itself) are empty is **skipped**, not scored 1.0 (rewards agreeing on
  nothing) or 0.0 (penalises an undefined comparison) or NaN-propagated into the mean.
- `val_dice_click_inside` / `_outside` only cover has-foreground rows with at least one positive
  click; rows with no positive click are excluded from both buckets. "Inside" is a strict majority
  vote when a row has multiple positive clicks; a tie counts as outside.
- Sanity-check epochs (`trainer.sanity_checking`) run this same path — no special-casing needed.

## Loss throughput

Use `--loss dc_ce` for normal long supervised training. `--loss cc_dc_ce` runs CPU connected components plus SciPy Euclidean-distance Voronoi inside the training loss and can make epochs roughly **4× slower** on A100/H200 nodes. Treat CC-DiceCE as an opt-in experiment or short fine-tuning objective. Details: [reference/losses.md](../reference/losses.md).

## Host RAM / cgroup OOM

Long MAE runs were killed by cgroup OOM from **checkpoint temp files on RAM-backed `/tmp`**, not GPU or Python heap.

**Recommended for long MAE:** set `NANOUNET_TMPDIR` to local disk (not tmpfs/CIFS), use `--dl-bucket m` or `l` with workers, and `--dl-persistent-workers`. Escape hatch: `NANOUNET_DL_FORCE_NO_WORKERS=1` forces `num_workers=0`.

Full write-up: [dev-notes/cgroup_memory.md](../dev-notes/cgroup_memory.md).

## Inputs / outputs

**Inputs**

- Preprocessed blosc2 + plans JSON
- ROI config (`--config`) copied to run dir as `nano_config.json`
- Optional MAE checkpoint or `--mae-pretrain`

**Outputs**

- `checkpoints/last.ckpt`, `checkpoints/best-*.ckpt`
- `splits_final.json` (created on first run, 5-fold fixed seed)
- `plans.json`, `dataset.json`, `nano_config.json` in run dir

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `--mae-resume requires --mae-pretrain` | MAE resume without integrated flag | Add `--mae-pretrain` or use `--mae-ckpt` |
| `--longi requires --init-weights` | Longi without warm-start | Pass stage-2 supervised ckpt via `--init-weights` |
| Conflicting resume flags | `--init-weights` + `--resume` / `--mae-pretrain` | Pick one init path |
| Cgroup OOM | tmpfs TMPDIR during checkpoint save | Set `NANOUNET_TMPDIR`; see cgroup doc |
| Missing plans / config | Preprocess or path error | Verify `--plans` basename and `--config` path |
| `--consistency-weight ... requires --prompts-per-patch >= 2` | Consistency enabled without 2 prompts | Add `--prompts-per-patch 2` |
| `batch_size ... not divisible by --prompts-per-patch` | Batch size / prompt count mismatch | Pick `--batch-size` as a multiple of `--prompts-per-patch` |

## Cohort-weighted sampling

Control the training mixture explicitly instead of letting it follow dataset sizes. Set in the ROI
config, not on the CLI, because it changes the data distribution rather than the run:

```json
"sampling": {
  "cohorts": { "d013": 0.25, "d025": 0.10 }
}
```

Named prefixes take the stated probability; the remaining mass spreads over all other cases in
proportion to their counts. An absent or empty block reproduces the uniform draw exactly. Names are
bare dataset prefixes (`d013`, no trailing underscore) — `--only-prefix` still uses the underscored
form, they are different flags. Composes with the `*_weights.json` lesion weights: cohorts pick
which case, lesion weights pick where inside it. See `nanounet/data/cohorts.py`.
