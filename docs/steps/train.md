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
  --mae-pretrain --dl-persistent-workers --mem-diag
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int | (required) | Dataset id (`DatasetXXX_*` under raw/preprocessed) |
| `-f`, `--fold` | int \| `all` | `0` | Fold 0–4, or `all` for full-data (val = train, in-sample metrics) |
| `--plans` | str | (required) | Plans basename without `.json` |
| `--config` | str | `configs/default.json` | ROI / prompt JSON; relative path tries cwd then repo root |
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
| `--mem-diag` | flag | off | Log RAM to `<run>/mem_diag.jsonl` and W&B `mem/*` |

Checkpoints: `<run>/checkpoints/` (supervised); `<run>/mae_pretrain/checkpoints/` (integrated MAE). Finetune with `--init-weights` writes to `<run>/finetune/`.

## Loss throughput

Use `--loss dc_ce` for normal long supervised training. `--loss cc_dc_ce` runs CPU connected components plus SciPy Euclidean-distance Voronoi inside the training loss and can make epochs roughly **4× slower** on A100/H200 nodes. Treat CC-DiceCE as an opt-in experiment or short fine-tuning objective. Details: [reference/losses.md](../reference/losses.md).

## Host RAM / cgroup OOM

Long MAE runs were killed by cgroup OOM from **checkpoint temp files on RAM-backed `/tmp`**, not GPU or Python heap.

**Recommended for long MAE:** set `NANOUNET_TMPDIR` to local disk (not tmpfs/CIFS), use `--dl-bucket m` or `l` with workers, and `--dl-persistent-workers`. Monitor with `--mem-diag`. Escape hatch: `NANOUNET_DL_FORCE_NO_WORKERS=1` forces `num_workers=0`.

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
