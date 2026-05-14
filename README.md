# nanoUNet

nanoUNet is a minimal port of the nnUNet, with promting support, and MAE pretraining. The defining feature of this codebase is its [nanochat](https://github.com/karpathy/nanochat) inspired code/architechture style. 

nanoUNet is wraps all torch in pytorch lightning to remove boilerplate code. 

## Getting started

Install from the repo root into a virtual environment:

```bash
python -m pip install -e .
```

Set the same three environment variables as nnU-Net (or use the `NANOUNET_*` aliases; they are synced automatically):

```bash
export NANOUNET_RAW="/path/to/nnUNet_raw"
export NANOUNET_PREPROCESSED="/path/to/nnUNet_preprocessed"
export NANOUNET_RESULTS="/path/to/nnUNet_results"
```

Paths with spaces must be quoted.

## Full pipeline

**Step 1 — Preprocess.** Reads every training case, extracts a dataset fingerprint (spacings, shapes, intensity statistics), runs the ResEnc planner to determine target spacing / patch size / network topology, then resamples and normalises every case into blosc2 tensors. All three phases run in one command:

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncL -np 8
```

This writes `dataset_fingerprint.json` and `nnUNetResEncUNetLPlans.json` next to the raw data, then a `nnUNetResEncUNetLPlans/3d_fullres/` folder of `*.b2nd` / `*.pkl` pairs and a `gt_segmentations/` copy.

The planner is where the "planning phase" lives. `--planner` selects the ResEnc preset: **Tiny** (~1–5M params, fast local / Mac runs), **M** / **L** / **XL** (8 / 24 / 40 GB VRAM targets). The resulting `<plans>.json` is what training reads; you pass its basename as `--plans`.

For fast local test runs on a laptop (small network, smaller patches), use Tiny with `--patch-vol small`:

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncTiny --patch-vol small -np 4
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetTinyPlans --config configs/default.json
```

Optional flags: `--skip-fingerprint` (reuse existing fingerprint), `--skip-plan --plans-name <ident>` (reuse an existing plan JSON).

`**--gpu-memory-gb**` overrides the VRAM budget the planner shrinks towards. The preset default is 24 GB for ResEncL. On a cluster where you preprocess on a CPU node but train on an H200 (80 GB), pass `--gpu-memory-gb 80` so the planner does not unnecessarily shrink the patch for a GPU it will never see during preprocessing.

`**--patch-vol small|medium|large|xlarge**` sets the initial target volume for the patch before the VRAM shrink loop runs. The four presets map to an isotropic edge length of 128 / 192 / 256 / 320 voxels (volumes of ~2M / 7M / 17M / 33M voxels). The default is `large` (256), which matches historical nnU-Net behaviour. On large GPUs with large anatomy the planner may produce very large patches; passing `--patch-vol medium` caps the starting point so patches stay tractable even if VRAM would technically allow more. The two flags are independent: `--gpu-memory-gb` controls the ceiling the VRAM loop converges to, `--patch-vol` controls the floor it starts from.

A typical cluster workflow where you preprocess on a CPU node and train on an H200:

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncL \
  --gpu-memory-gb 80 --patch-vol medium -np 8
```

**Step 2 — Train.** Lightning fit on one fold. On first run it writes `splits_final.json` (5-fold KFold, seed 12345) next to the preprocessed dataset and reuses it on subsequent runs:

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --config configs/default.json
```

**MAE pretraining (optional).** Masked reconstruction on the same ResEnc backbone and `*.b2nd` tensors (labels unused):

```bash
nanounet_pretrain -d 001 -f 0 --plans nnUNetResEncUNetLPlans --epochs 1000
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --mae-ckpt /path/to/.../checkpoints/last.ckpt --lr 0.001
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --mae-pretrain --mae-epochs 1000 --lr 0.001
```

Use `last.ckpt` for transfer; fine-tune with a lower peak LR (e.g. `1e-3`) than scratch ([Wald et al., arXiv:2410.23132](https://arxiv.org/abs/2410.23132)). Only encoder weights load; prompt channels on the stem are zero-initialized.

**Loss.** Default is nnU-Net DC+CE with deep supervision. For multi-instance lesion data (many small lesions per case) enable **CC-DiceCE** (Bouteille et al., ISBI 2026): `--loss cc_dc_ce` (synonym `-loss`). See [`docs/losses.md`](docs/losses.md) for math, `batch_dice` behaviour, and caveats.

**Patch size** trade-offs (FOV vs resolution for tiny vs huge lesions) are summarised in [`docs/patch_size.md`](docs/patch_size.md).

On Apple Silicon, `auto` picks MPS; add `--accelerator mps` to force it. Prefer `--precision 32-true` only if `16-mixed` causes issues.

Train all five folds by repeating with `-f 1` through `-f 4`. Checkpoints and a copy of the plans/config land in `$NANOUNET_RESULTS/nanounet/<dataset>_<plans>_f<fold>/`.

**Step 3 — Predict.** Pass one or more channel images, a model directory, and a point prompt in preprocessed voxel space (z,y,x, comma-separated). The forward pass runs with TTA by default; add `--border-expand` for the BFS hull-shell post-processing step:

```bash
nanounet_predict \
  -i case_0000.nii.gz \
  -o seg.nii.gz \
  -m /path/to/run \
  --point-zyx 48,120,95
```

The `--point-zyx` coordinates are in the **preprocessed** (cropped + resampled) space, not original image space.

## File structure

```
nanounet/
├── common.py          # env paths, Rich CLI helpers (cprint, nano_progress, …)
├── config.py          # RoiPromptConfig dataclass
├── data/
│   ├── io.py          # SimpleITK read/write, reader resolution
│   ├── resampling.py  # 3D resample (separate-z for anisotropic data)
│   ├── normalization.py
│   ├── crop.py        # nonzero crop / insert
│   ├── augment.py     # batchgeneratorsv2 train/val transforms
│   ├── dataset.py     # Blosc2Case lazy reader
│   └── sampling.py    # 4-mode patch sampling
├── prompt/
│   ├── centroids.py   # cc3d centroids → *_centroids.json
│   ├── encoding.py    # EDT / binary heatmap pair
│   └── propagation.py # baseline→follow-up COG offset
├── plan/
│   ├── fingerprint.py # multiprocess intensity / shape stats
│   ├── planner.py     # orchestration: fingerprint → plans.json
│   ├── planner_resenc.py  # ResEnc VRAM loop + topology
│   ├── planner_topology.py
│   ├── preprocess.py  # spawn-pool case preprocessing
│   ├── case_pp.py     # single case: transpose → crop → norm → resample
│   └── splits.py      # KFold splits_final.json
├── model/
│   ├── network.py     # ResidualEncoderUNet (+ optional extra in / num_classes)
│   ├── mae_transfer.py # encoder-only load from MAE ckpt (+ prompt stem zero-fill)
│   ├── losses.py      # DC + CE + deep supervision
│   ├── cc_dice_ce.py  # optional Voronoi CC-DiceCE (+ global DC+CE)
│   └── lr_schedule.py # poly / stretched_tail_poly
├── pretrain/
│   ├── masking.py     # bottleneck-grid mask → full res
│   ├── dataset.py    # random ROI iterables (no prompts)
│   └── module.py     # NanoMAELM
├── train/
│   ├── lightning_module.py  # training_step, val Dice, configure_optimizers
│   └── data_module.py       # IterableDataset + LightningDataModule
├── infer/
│   ├── predictor.py   # single-tile forward, TTA
│   ├── border_expand.py  # BFS hull-shell Gaussian merge
│   └── export.py      # logits → original space → NIfTI
└── cli/
    ├── preprocess.py
    ├── pretrain.py
    ├── train.py
    └── predict.py
```

## Smoke test

```bash
python -c "import sys; import nanounet.cli.preprocess, nanounet.cli.train, nanounet.cli.predict; assert 'nnunetv2' not in sys.modules; print('ok')"
```

