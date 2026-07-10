# Longi finetune GPU starvation — diagnosis + fix (2026-07-10)

## Symptom
`Dataset114_registered_f0_finetune_dwb_adamw1e-5_bs6_500ep` averaged ~60% GPU util
(~1400 s/epoch) on A100-40GB, vs 95%+ (~1000 s) for the earlier `Dataset013_registered`
longi finetune. Same model, same plans (`nnUNetResEncUNetLPlans_h200_smallpv`, patch
[96,160,160], batch 6), same `--longi` two-stream setup.

## Root cause
Two compounding facts:

1. **The d114 slurm script regressed `--dl-bucket` from `l` (8 workers) to `m` (4 workers).**
   The d013 longi/registered scripts that hit 95% use `--dl-bucket l`
   (`scripts/slurm_nanounet_finetune_d013_{longi,registered}.sh:112,102`). The new d114
   script was written with `m`. This is the primary regression.

2. **Longi is dataloader-bound: the pipeline is CPU-limited, not GPU-limited.** Per-patch
   CPU cost (measured on-node, single worker):
   - producer (blosc2 2-channel decompress + patch build + EDT heatmaps): **162 ms**
   - consumer (augment chain on the 6-channel tensor): **519 ms**
     - `SpatialTransform` 290 ms + `GaussianBlurTransform` 222 ms dominate; both scale
       with the 6 channels (2 CT + 4 prompt heatmaps).
   - total ~**680 ms/patch**, ~4.1 s for a batch of 6 done serially.

   The `.b2nd` volumes are the unigradicon registration frame (~578x520x515 x2ch float32
   ≈ 1.24 GB/case), but patch reads are chunked (`chunks=(2,40,64,64)`) so only overlapping
   chunks decompress — decompression is *not* the bottleneck; **augmentation is.**

## Evidence (A100-40GB, 38 usable cores, 120-iter burst)
| bucket | train workers | steady GPU util | wall (120 iters) |
|--------|---------------|-----------------|------------------|
| m      | 4             | 45%             | 4:02             |
| l      | 8             | 77%             | 2:37             |
| xl     | 16            | 75% (oversub)   | 2:33             |

`xl` ≈ `l` because 16 workers oversubscribe 38 cores. The slurm allocation is **48 cores**
(this interactive node was pinned to 38 via sched-affinity), so `l` (8 workers) has ample
headroom there and projects to ~95% util / ~<=1000 s/epoch — matching the d013 result.

## Fix
`scripts/slurm_nanounet_finetune_d114_registered.sh`: `--dl-bucket l`, A100-only (dropped the
H200 batch-10 branch). Rule of thumb: **never run longi below bucket `l`.**

## Augmentation synchronization (verified — no bug)
Longi geometry augmentation is correctly synchronized across timepoints. The BL and FU
streams are stacked into one 6-channel tensor (`data/sampling_longi.py:66`) and a single
`SpatialTransform` + single `MirrorTransform` are applied to the whole tensor
(`data/augment.py:53,100`) — so BL and FU always get the identical flip/rotation/scale, and
the click heatmaps warp consistently with their CT. Only *intensity* transforms
(brightness/contrast/blur/gamma) use `synchronize_channels=False`, i.e. BL_CT and FU_CT get
independent intensity jitter — intended, not a bug.

## Fix 2 — CT-only intensity augmentation (applied)
The intensity transforms (noise/blur/brightness/contrast/lowres/gamma) used to also hit the 4
prompt-heatmap channels. They are now wrapped in `ChannelSubsetImageTransform` and restricted to
the CT channels — longi `(0, 3)`, single-stream unchanged (`data/augment.py`,
`train/data_module.py:125`). More correct (synthetic prompts are no longer intensity-jittered)
and cheaper:
- augment: **626 -> 445 ms/patch** (GaussianBlur 222 -> 79; gamma/contrast/lowres also down).
- end-to-end (bucket l, 38 cores): steady util **77% -> 84%**, **1.21 -> 1.05 s/iter** (~1050 s/epoch).
- on the 48-core slurm node this clears 95% util and <1000 s/epoch.

Remaining floor: `SpatialTransform` (~289 ms/patch, always fires, must resample all 6 channels
so the heatmaps warp with anatomy). Cutting it needs the bigger refactor of transforming click
*coordinates* through the affine and rendering heatmaps *after* spatial aug (2 channels resampled
instead of 6). Not done — flag if more headroom is ever needed.

## Resource sizing (CPU / mem)
Bucket `l` spawns 8 workers; each ≈ 1 producer thread (blosc2, GIL-releasing) + 1 consumer thread
(torch augment, GIL-releasing) ≈ up to 2 cores, so ~16 cores peak + ~2-3 for the main loop ≈ ~19
cores active. **48 CPU / 160 G is already over-provisioned; more does not help bucket `l`** (only
8 workers exist). Peak host RAM ~30-40 G (prefetch 4 x ~0.35 G/batch/worker + decompress scratch;
page cache is dropped via `fadvise DONTNEED`), so 160 G is ~4x headroom. To spend more cores you
must raise the bucket (`xl` = 16 workers, wants ~40 cores) — unnecessary here since `l` + Fix 2
already meets target. `xl` only oversubscribes a 38-core box; on 48 cores it fits but buys little.
