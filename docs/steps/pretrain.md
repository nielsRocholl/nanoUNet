# MAE pretrain

Standalone masked-autoencoder pretraining on preprocessed blosc2 data. No prompts; produces encoder weights for `--mae-ckpt` or inspection.

Default output: `$NANOUNET_RESULTS/nanounet/<DatasetFolder>_<plans>_mae_pretrain_f<fold>/`.

## Command

```bash
nanounet_pretrain -d 001 -f 0 --plans nnUNetResEncUNetLPlans --epochs 1000
```

Long run with stable workers:

```bash
nanounet_pretrain -d 001 -f 0 --plans nnUNetResEncUNetLPlans \
  --dl-bucket m --dl-persistent-workers --mem-diag
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int | (required) | Dataset id |
| `-f`, `--fold` | int \| `all` | `0` | Fold index 0–4, or `all` for full-data (val = train, in-sample metrics) |
| `--plans` | str | (required) | Plans basename (no `.json`) |
| `--epochs` | int | `1000` | MAE epoch budget |
| `--lr` | float | `1e-2` | Initial learning rate |
| `--lr-schedule` | choice | `cosine_warm_restarts` | `cosine_warm_restarts` \| `poly` |
| `--cosine-t0` | int | `250` | Cosine restart period |
| `--cosine-t-mult` | int | `1` | Cosine period multiplier |
| `--cosine-eta-min` | float | `0.0` | Cosine minimum LR |
| `--wd` | float | `3e-5` | Weight decay |
| `--mask-ratio` | float | `0.75` | Fraction of voxels masked per patch |
| `--batch-size` | int | from plans | Override `3d_fullres.batch_size` from plans |
| `--iters-per-epoch` | int | `250` | Training batches per epoch |
| `--val-iters` | int | `50` | Validation batches per epoch |
| `--out` | str | auto | Output directory |
| `--no-wandb` | flag | off | Disable Weights & Biases |
| `--wandb-project` | str | `nanounet-mae` | W&B project name |
| `--wandb-name` | str | auto | W&B run name |
| `--dl-bucket` | choice | `m` | `s`: 2/1 workers (0 on tmpfs TMPDIR). `m`: 4/2. `l`: 8/4. `xl`: 16/8 |
| `--dl-persistent-workers` | flag | off | Keep DataLoader workers between epochs |
| `--resume` | str | none | MAE Lightning checkpoint; must exist; epoch target must match `--epochs` |
| `--precision` | str | `16-mixed` | Lightning precision (e.g. `32-true`) |
| `--accelerator` | choice | `auto` | `auto` \| `cpu` \| `cuda` \| `gpu` \| `mps` |
| `--mem-diag` | flag | off | Log cgroup/process RAM to `<out>/mem_diag.jsonl` |

## Inputs / outputs

**Inputs**

- Preprocessed `$NANOUNET_PREPROCESSED/DatasetXXX_*/<plans>/3d_fullres/`
- Plans JSON and raw `dataset.json`

**Outputs** (under run dir)

- `checkpoints/last.ckpt`, `checkpoints/best-*.ckpt`
- `plans.json`, `dataset.json` (copied for downstream use)

Encoder weights load into supervised training via `nanounet_train --mae-ckpt <path>`.

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| Cgroup OOM on long runs | Checkpoint temp files on tmpfs `/tmp` | Set `NANOUNET_TMPDIR` to local disk; see [cgroup_memory.md](../dev-notes/cgroup_memory.md) |
| Resume epoch mismatch | `--resume` ckpt trained for different `--epochs` | Match `--epochs` to checkpoint target or start fresh |
| Missing plans | Preprocess not run | Run `nanounet_preprocess` first with matching `--plans` |
| `MAE pretrain already reached num_epochs` | Resume on finished run | Expected clean exit; use ckpt for `--mae-ckpt` |
