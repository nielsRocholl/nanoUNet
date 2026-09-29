# GPU efficiency: compute is the bottleneck, always

Load this when touching dataloaders, sampling, augmentation, patch iterables, losses, `lightning_module.py`,
`infer/`, or anything that runs per step or per patch. Rule IDs refer to SKILL.md.

A starved GPU is a bug of the same severity as a wrong loss. Pretraining and supervised training currently run
near full utilization. Every change must keep it that way, **with a number to show it**.

## Measurement protocol (G4): how to produce the number

Metric: `epoch_wall_time_sec` (wandb/CSV). It is logged in `train/lightning_module.py` and `pretrain/module.py`, and covers
the train epoch plus its validation, from `on_train_epoch_start` to `on_validation_epoch_end`.

1. Keep everything but the change fixed: same node/GPU type, `--dl-bucket`, `--batch-size`, `--iters-per-epoch`,
   `--val-iters`, `--val-every-n-epochs 1`, `--precision`, and the same dataset and fold.
2. Run at least 4 epochs per arm (`--epochs 4`, `--no-wandb` is fine, since CSVLogger still writes). **Discard epoch 0**: it includes
   worker spin-up, cudnn autotune, and page-cache warm-up.
3. Report the median of epochs 1..N plus GPU utilization (`nvidia-smi dmon -s u -d 5`, or wandb system charts).
   `--mem-diag` adds worker/host memory, for when the change touches worker RAM or `/dev/shm`.
4. Put this table in the commit message or report:

```
| arm    | epoch_wall_time_sec (median e1-e3) | GPU util | Δ      |
|--------|------------------------------------|----------|--------|
| before | 212.4                              | 97%      |        |
| after  | 205.1                              | 98%      | -3.4%  |
```

Treat a difference under ~2% as noise unless it is repeated. A regression of 5% or more is rejected unless the change buys accuracy, and
you must state the trade-off explicitly (G5).

## The data path (G1, G3, G6)

- **Workers:** worker count and prefetch come from `dataloader_prefs.py` buckets (`--dl-bucket s|m|l|xl`). Don't hardcode
  `num_workers`. Persistent workers are opt-in (`--dl-persistent-workers`) because of worker RSS / `/dev/shm`
  pressure. Enable them only with a `--mem-diag` run showing that memory is flat.
- **Worker work:** blosc2 decode, crop, augmentation (`batchgeneratorsv2`), resampling, and heatmap rendering. The main
  process only does H2D and the step.
- **H2D:** use pinned host memory and `.to(device, non_blocking=True)`, as already done in `lightning_module.py`, `ema.py`, and
  `pretrain/module.py`. Pre-allocate and reuse buffers when shapes are fixed (nanochat allocates its buffer once and does a single copy per step).
- **Never** do `.cuda()` or `torch.device("cuda")` in `__getitem__`/`__iter__`, and never touch the CUDA context in a worker.

## The step (G2)

- **Sync points** are `.item()`, `.cpu()`, `.tolist()`, `.numpy()`, `print(tensor)`, `if tensor:`, `torch.nonzero` (dynamic
  shape), and host-side boolean masks. None of them belong in `forward`/`training_step`/loss code. Keep metrics as tensors and let
  `self.log(..., on_step=False, on_epoch=True)` reduce them. Known debt: `model/cc_dice_ce.py` calls `.item()` inside
  the loss. It is on the next-touch list.
- **Per-step host work**: no Python loops over batch elements, no per-step `dict` rebuilding of big structures, no
  per-step wandb calls outside Lightning's logger.
- **Timing** uses `torch.cuda.synchronize()` only immediately around a timer, never "to be safe".

## Known wins: candidates, each must pass the protocol above

| Technique | nanochat precedent | nanoUNet status |
|---|---|---|
| `gc.collect(); gc.freeze(); gc.disable()` after step 1, `gc.collect()` every ~5k steps | `base_train.py:586-594`: GC was costing ~500ms spikes | not done. Cheap to test. |
| Prefetch the next batch *while* fwd/bwd runs | `base_train.py:518` | Lightning + workers does this. Keep `prefetch_factor` ≥2. |
| `torch.backends.cudnn.benchmark = True` for a fixed patch size | n/a (transformer) | **Bit-identity breaking** (kernel choice). Never inside a refactor series. Measure only as its own change. |
| `torch.compile(model, dynamic=False)` on the train step, eager model kept for sliding-window inference | `base_train.py:246`, `orig_model` kept for eval | **Bit-identity breaking.** Never inside a refactor series. |
| bf16 autocast on Ampere+ (no GradScaler) vs `16-mixed` | global `COMPUTE_DTYPE` detected once | **Bit-identity breaking** vs `16-mixed`. Never inside a refactor series. |
| `channels_last_3d` memory format | n/a | untested. Wins are cudnn- and arch-dependent. |
| 0-D tensors for scalars that change each step (LR, loss weights) under compile | `optim.py:262` avoids recompiles | relevant only if compile is adopted |
| Meta-device init plus `to_empty` for big models or ckpt loads | `checkpoint_manager.py:99-104` | usually not needed at UNet sizes |

## Inference (G7)

- Every predict path is `@torch.inference_mode()` (not `no_grad`). That is already true in `infer/predict_case.py` and `infer/tta.py`.
- Sliding-window: batch the tiles, keep the Gaussian importance map on the GPU, and accumulate logits on the GPU in fp16/bf16 when
  the volume fits. Move to CPU once per case, not per tile.
- Benchmarks run one warmup pass first (cudnn autotune, allocator, and kernels), then time with `synchronize()` around
  the timer. Report per-case seconds and voxels/s.
