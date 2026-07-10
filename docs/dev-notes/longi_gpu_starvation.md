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

## Headroom option (NOT applied — changes training semantics)
The intensity transforms (blur/gamma/contrast/brightness/noise/lowres) currently also hit the
4 prompt-heatmap channels. Restricting them to the 2 CT channels would cut augment cost ~3x
(and is arguably more correct — heatmaps are synthetic prompts, not intensities). Left for the
user to decide: it changes what the model trains on and breaks comparability with d013.
