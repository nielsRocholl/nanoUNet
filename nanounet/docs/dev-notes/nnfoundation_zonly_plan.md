# Plan: nnFoundation weights + z-only resampling as the nanoUNet default

Date: 2026-10-02
Status: **approved for implementation** (owner decisions recorded 2026-10-02). Nothing below is implemented yet.
Audience: a fresh agent with no prior context. Everything you need is in this file. Read all of it before
touching code, and follow the `nanochat-style` skill (`.claude/skills/nanochat-style/SKILL.md`) for every change.

---

## 0. What we are building (one paragraph)

Today nanoUNet resamples every case to the Dataset900 median spacing (2.5 × 0.758 × 0.768 mm, z/y/x) and trains a
planner-derived anisotropic ResEnc-L from scratch or from our own MAE. The new **default** is:

1. **Resample only the slice axis** of each case to a fixed `z_target = 1.0 mm`. The other two axes keep the
   case's **native** spacing and voxel grid (no in-plane interpolation at all).
2. Use the **nnFoundationCNN** self-supervised weights (DKFZ, ResEnc-L) as the encoder init, with **its** fixed
   topology and **its** recommended 192³ patch, instead of the planner's.
3. Normalize with **per-image Z-score** (what nnFoundation was pretrained on), not `CTNormalization`.
4. Fine-tune with the official nnU-Net recipe for nnssl checkpoints (SGD, lr 1e-3, 50-epoch linear warmup, poly,
   no deep supervision), batch size from our VRAM estimator.

The old path (planner topology, median spacing, CTNormalization) stays available behind `--no-foundation`.

### 0.1 Scope: three deliverables, two hand-backs

| # | Deliverable | Done when |
|---|---|---|
| 1 | **nanoUNet code changes** (P1–P6, P9) on branch `feat/nnfoundation-zonly`, pushed. | All acceptance checks pass; nanochat-style checker clean; docs updated. |
| 2 | **Preprocessed Dataset900 at z = 1.0 mm** (`nnFoundationCNN_z1p0` plans, data folder, centroid sidecars, d013 lesion weights, new valset), written **directly** to `/nnunet_data/NanoUNet_preprocessed/Dataset900_Merged/`, produced by a preprocessing SLURM job `nanounet/scripts/slurm_preprocess_foundation_900.sh` (P7). | P7 checks pass; `splits_final.json`/`cohorts.json` sha256 unchanged. |
| 3 | **Training SLURM script** `nanounet/scripts/slurm_foundation_900_h200.sh` for the supervised run + d013 finetune (P10), committed. | Smoke run (P8) passed with the same flags; script reviewed by the owner. |

**Nothing may live only on a node's local disk.** Interactive sessions and SLURM containers (including
`dlc-slowpoke`) are wiped when they end. Code goes to git (pushed), data goes to `/nnunet_data` (CIFS; the network
link was upgraded, so copies are faster than before). The **training** job always stages a copy of the data to the
node's local disk with rclone at job start (as the existing train scripts do); that copy is scratch and dies with
the job — the canonical data stays on `/nnunet_data`.

**Hand-back points (STOP and report to the owner, do not continue on your own):**
- **H1, after Stage A** (code that preprocessing needs, plus the preprocessing SLURM script): the owner reviews and
  submits the preprocessing job.
- **H2, after P10**: the owner reviews and submits the SLURM job. You never run `sbatch`.

No planner run is needed, and the fingerprint is computed once up front (the old one was deleted; see §2.4) and then reused
(`--skip-fingerprint`), and the foundation plans JSON is written directly from fixed constants (P2, seconds). But
the data itself **must** be re-preprocessed (new spacing, Z-score, new patch-sized blosc2 chunks, new sidecar keys),
and that needs Stage A's code first.

### 0.2 Non-negotiable: nanochat-style

**Every code change in this plan must follow the `nanochat-style` skill** (`.claude/skills/nanochat-style/SKILL.md`).
Invoke the skill before writing code, load the references for the area you touch (`code.md` for any `.py`,
`cli.md` for `cli/` and error messages, `gpu.md` for data path/loader/training/inference, `docs.md` for docs),
and follow its workflow on every change: <200 LOC per file (R1), flat and procedural (R3), no fallbacks for
missing data (R12), frozen on-disk names (R18), errors that name the fix (E-rules), `config_table` + `next:` line
in CLIs (U2/U3), measured throughput for data-path changes (G4), docs updated in the same change (D-rules).
Run `python .claude/skills/nanochat-style/scripts/check.py --changed` before every commit and fix every error
and every warn on lines you wrote. Waivers need an inline reason (`# nanochat-style: allow R1 (why)`).
A phase is not done until the checker is clean.

---

## 1. Owner decisions (do not re-litigate)

| ID | Decision |
|---|---|
| D1 | Normalization in foundation mode = `ZScoreNormalization` per image, `use_mask_for_norm = [False]`. |
| D2 | Prompt geometry (registration-error offsets, lesion-size bins) must be physical (mm) and converted **per case**. A separate task is fixing the dataset-level version first (see §3, dependency P0). |
| D3 | Cases whose thick axis is not array axis 0 (KiTS d022, some PanTS d028) are handled by **resampling whichever axis is thickest, per case**. No reorientation. |
| D4 | Official finetune recipe: SGD (Nesterov, momentum 0.99), lr **1e-3**, linear warmup of the whole net, then poly; **deep supervision off**. Batch size is **not** fixed to 2 (nnU-Net hard-codes 2); we derive it from the VRAM budget with our existing estimator. **Units:** nnU-Net's 50 warmup epochs are 50 × 250 iters × batch 2 = 25k patches; our epoch is 1000 iters × batch 12 = 12k patches, so the equivalent default is **`--warmup-epochs 2`**, not 50. |
| D10 | **Budget: roughly 4 days of supervised training** (wall clock, including validation and the rclone staging copy). This is a target, not a hard cap; a few hours either way is fine. A somewhat lower batch size than the estimator's 12 is acceptable if it buys more updates in the budget. Set batch and epoch count from the P8 measurement, not from a target epoch count (see P10). |
| D11 | **Training always reads from node-local disk.** The training SLURM job copies the dataset from `/nnunet_data` to local disk with rclone at job start, exactly like `nanounet/scripts/slurm_final_900_h200.sh` does. Never train directly from CIFS. |
| D5 | **No control run.** Do not plan or launch a same-spacing-without-foundation ablation. |
| D6 | `splits_final.json` stays **exactly** as it is (byte-identical). Click-point sidecars (`*_centroids.json`), the valset manifest and the d013 lesion weights must be **regenerated** in the new voxel grid. |
| D7 | `z_target = 1.0 mm`. |
| D8 | Decoder and segmentation head are trained from scratch. Load **encoder + stem only** (what nnU-Net's `PretrainedTrainer` does). |
| D9 | nnFoundation's topology and 192³ patch are used as-is; the planner does not choose them in foundation mode. |

### Decisions taken by the plan author (owner has not explicitly confirmed; implement them, flag in your report)

| ID | Decision | Why |
|---|---|---|
| A1 | Weights are downloaded **at preprocess time** (not training time), verified, and recorded in the plans JSON (`pretrain_info`). Training reads the path from the plans; it never touches the network. | Mirrors nnssl's `pretrain_info`; no surprise downloads mid-run; one source of truth. |
| A2 | Data resampling along the resampled axis uses **linear** interpolation (`order_z = 1`) instead of nnU-Net's nearest (`order_z = 0`). Seg and probability resampling keep nnU-Net's defaults. | Upsampling 5 mm → 1 mm with nearest repeats each slice 5×, giving plateaus the pretrained filters never saw. |
| A3 | "Thickest axis" rule: `ax = argmax(sp)` if `sp[argmax] > 1.25 × sp[0]`, else `ax = 0`. | Avoids picking an in-plane axis on near-isotropic thin cases (e.g. 0.70/0.78/0.78). |
| A4 | Batch size is rounded **down to an even number** (runs use `--prompts-per-patch 2`, which requires divisibility). | `validate_train_args` would otherwise reject the run. |
| A5 | Valset small-lesion threshold moves from voxels (`SMALL_LESION_MAX_VOX = 500`) to an equivalent-sphere diameter of **10 mm** (the value its comment already claims). | 500 voxels is ~11 mm at the old grid but ~8 mm at the new one. |
| A6 | `--init-weights` takes precedence over the foundation encoder (logged, not an error); `--mae-ckpt`/`--mae-pretrain` with foundation weights is an error unless `--no-foundation`. | Finetune chains warm-start from a full supervised ckpt; mixing MAE and foundation weights is always a mistake. |

---

## 2. Facts already established (measured 2026-10-02; do not redo, but re-verify where a step says so)

### 2.1 nnFoundationCNN checkpoint

- Hugging Face repo `MIC-DKFZ/nnFoundationCNN`, file `checkpoint_final.pth` (409,579,442 bytes), plus
  `adaptation_plan.json`. License **CC-BY-SA-4.0**. Paper: arXiv 2609.26924 ("nnFoundation: 3D Foundation Models
  for Radiology"), pretrained on 2.1 M CT/MRI/PET volumes from 125 datasets. Citation bibtex is inside the ckpt
  under `citations`.
- `huggingface_hub` 0.34.3 is installed: `hf_hub_download("MIC-DKFZ/nnFoundationCNN", "checkpoint_final.pth", cache_dir=...)`.
- `torch.load(path, map_location="cpu", weights_only=True)` returns a dict with keys
  `network_weights`, `citations`, `nnssl_adaptation_plan`.
- `network_weights`: 956 tensors. Prefixes: `encoder.*` (448 tensors), `decoder.encoder.*` (alias of the encoder
  inside the decoder), `decoder.stages.*`, `decoder.transpconvs.*`, `decoder.seg_layers.*`. **No `net.` prefix.**
- The stem conv appears twice: `encoder.stem.convs.0.conv.weight` and `encoder.stem.convs.0.all_modules.0.weight`
  (shape `(32, 1, 3, 3, 3)`). In `dynamic_network_architectures` these are the **same Parameter** registered
  twice; loading both is fine (verified).
- `nnssl_adaptation_plan`: `architecture_plans.arch_class_name = "ResEncL"`, `arch_kwargs = None`;
  pretrain configuration `noresample` with `spacing: None`, `spacing_style: noresample`,
  `normalization_schemes: ["ZScoreNormalization"]`, `use_mask_for_norm: [False]`, `patch_size: [192,192,192]`;
  `original_median_spacing_after_transp: [1,1,1]`; `pretrain_num_input_channels: 1`;
  `recommended_downstream_patchsize: [192,192,192]`; `key_to_encoder: "encoder.stages"`,
  `key_to_stem: "encoder.stem"`, `keys_to_in_proj: ["encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"]`,
  `key_to_lpe: None`.
- In nnU-Net's own tooling (`nnunetv2/experiment_planning/like_nnssl.py`), `-am like_pretrained` with
  `spacing: None` **falls back to the dataset's default spacing**; the `-am` default is `default_nnunet`. The
  tooling forces Z-score normalization and batch size 2. nnU-Net's `PretrainedTrainer` loads encoder + stem only,
  repeats the stem across input channels, uses SGD lr 1e-3 (1e-2 from scratch), 50-epoch linear warmup then poly,
  deep supervision off. `DynamicPretrainedTrainer` adapts kernel mismatches by **averaging** along the shrinking
  axis (`[3,3,3] → [1,3,3]`).

### 2.2 The exact topology to use (matches the checkpoint 448/448)

```python
NET_CLASS = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
ARCH_KWARGS = {
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "conv_op": "torch.nn.modules.conv.Conv3d",
    "kernel_sizes": [[3, 3, 3]] * 6,
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "conv_bias": True,
    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
}
KW_REQUIRES_IMPORT = ["conv_op", "norm_op", "dropout_op", "nonlin"]
PATCH_SIZE = [192, 192, 192]
```

Verified: building this net with 2 input channels (CT + prompt heatmap, `N_PROMPT_CHANNELS = 1` in
`nanounet/prompt/encoding.py`) and loading via the existing `load_mae_encoder` after wrapping the weights as
`{"state_dict": {"net." + k: v}}` loads **447 encoder tensors, 0 missing, 0 unexpected** (the 448th is the
alias), the stem's channel 0 equals the pretrained weights exactly, and the prompt channel is zero. All 50
decoder stage/transpconv tensors also match shape (we deliberately do **not** load them, D8).

For reference, the current anisotropic plan (first kernel `[1,3,3]`) has 6 mismatching encoder tensors (stem +
stage-0 block convs, plus aliases); averaging along z makes it 448/448. That path is only relevant if someone
uses foundation weights with a non-foundation plan, which this plan does not support (D9).

### 2.3 GPU benchmark (H200 143 GB, bf16 autocast, synthetic data, 2 input channels, deep supervision on, SGD)

| Config | s/step | Peak alloc |
|---|---|---|
| Current: 2.5 mm, patch 64×192×192, batch 10 | 0.466 | 38.7 GiB |
| Foundation topology, 192³, batch 2 | 0.310 | 18.3 GiB |
| Foundation topology, 192³, batch 3 | 0.402 | 26.9 GiB |
| Foundation topology, 192³, batch 4 | 0.499 | 35.6 GiB |

Our VRAM estimator (`planner_resenc.py`, ResEncL preset, `REF_BS_3D = 2`, `MIN_BATCH = 2`) gives for this
fixed net at 192³: 24 GB → 2, 40 GB → 4, 80 GB → 7, 141 GB → 13 (→ 12 after A4). Real runs of the current
model take 578–706 s/epoch at 1000 iters/epoch, about 1.3–1.5× the pure GPU step time.

### 2.4 Dataset900_Merged facts

**Update 2026-10-02:** the old preprocessed folder was deleted by accident, so this plan REBUILDS it. Nothing
below that mentions the old `nnUNetPlans_3d_fullres/`, the 5796-case fingerprint, `valset_2000.json` or the old
plans JSON exists any more; the old 2.5 mm data and valset are NOT rebuilt.

- Raw: `/nnunet_data/NanoUNet_raw/Dataset900_Merged/` holds only `dataset.json` (5690 cases, entries point into the
  source datasets `Dataset011..031`) and `merged_sources.json`.
- Preprocessed (canonical, rebuilt by P7): `/nnunet_data/NanoUNet_preprocessed/Dataset900_Merged/`.
  - `splits_final.json` / `cohorts.json`: **regenerated 2026-10-02**, then frozen (D6) and never rewritten.
    Splits = `make_balanced_split(sorted(dataset.json["dataset"]), 0.15, 12345)`: 4833 train / 857 val; all 102
    `val:` cases of `experiments/exp00c_seg_eval_manifest/20260930T184041Z_paper_v1/cases.csv` land in val, none in
    train, which confirms it reproduces the original. sha256 splits `1016c2be2ac003bbe8f98d023c791dec2454c25c4d7f045c4701246d3e823ee7`,
    cohorts `ec46f4d03335cd91c40a90de49625aced2f7137bf33530437ab41464e5a2a1d9`. Per-cohort counts equal
    `exp00a_data_audit.PAPER_COHORTS` (21 cohorts, 5690 total); only d029 is >1 case off 15% (10 of 59 val, patient-grouped draw).
  - `dataset_fingerprint.json`: computed fresh (5690 cases; the old one had 5796). Only nominal values use it.
  - Hard guard (all of split creation, preprocess splits-safety, train startup): the 60 patients in
    `/nnunet_data/Longitudinal-CT/test_patients.csv` must not appear in any case id or image/label path
    (`nanounet/plan/dataset/holdout.py`). Result at creation: 0 matches.
  - New data folder `nnFoundationCNN_z1p0_3d_fullres/`, plans `nnFoundationCNN_z1p0.json`, sidecars,
    `gt_segmentations/`, lesion weights, `valset_2000_nnFoundationCNN_z1p0.json` (all produced by P7).
- **Danger:** `run_preprocess` does `shutil.rmtree(out_dir)` on the plan's `data_identifier` folder when not
  `--resume`; keep `data_identifier` unique per plans variant.
- Splits/cohorts in `nanounet_preprocess`: P6 never rewrites existing files (verify and keep); only absent files are created.
- Native z-spacing (array axis 0): p5/25/50/75/95 = 0.70/1.00/2.50/3.00/5.00 mm. 28% ≤ 1 mm, 21% in (1,2],
  28% in (2,3], 2% in (3,4.5], 21% > 4.5 mm. In-plane spacing p5/50/95 = 0.594/0.758/0.977 mm.
- Total voxels after z-only resampling at 1.0 mm vs current: **2.23×** (median case 83 Mvox, p95 209 Mvox). By
  linear scaling ~740 GB on disk (rough; upsampled data compresses better, so likely less). Local disk `/`
  (maindisk) has 25 TB free; `/tmp` is tmpfs (RAM) — never preprocess or stage there.
- Per-cohort native z (median): d011 5.0, d012 1.0, **d013 3.0** (main finetune cohort, 38% sampling weight,
  537 cases), d014 5.0, d015 1.24, d016 2.5, d017 1.0, d018 1.25, d019 1.25, d020 5.0, d021 3.0, d022 0.78*,
  d023 1.0, d024 1.25, d025 0.80, d026 3.0, d027 5.0, d028 2.5, d029 1.25, d030 3.75, d031 3.0.
- *Thick axis not on array axis 0:* **d022 (KiTS23)**: 387/489 cases have spacing like `(0.98, 0.98, 4.0)`,
  thick axis = **array axis 2**, SimpleITK direction `(0,0,1, 0,1,0, -1,0,0)`. **d028 (PanTS)**: 20 cases with
  thick axis 2 (same direction pattern), 2 with axis 1, plus 40 isotropic 1.5 mm cases (no thick axis). Rule A3
  handles all of these.

### 2.5 Code map (what exists, what reads spacing)

| File | Role | Notes |
|---|---|---|
| `nanounet/cli/preprocess.py` (176 LOC) | Orchestrates fingerprint → `run_plan` → `run_preprocess` → splits/cohorts → optional valset | Calls `run_plan(did, planner, gpu_mem, None, plans_name, patch_edge=...)`; never passes `overwrite_target_spacing`. |
| `nanounet/plan/resenc/planner.py` (176) | `run_plan`: target spacing (`_fullres_spacing`), transpose (`_transpose`, argmax of target spacing), median shape, writes plans JSON (`data_identifier = f"{ident}_3d_fullres"`) | Already has `overwrite_target_spacing`. Writes `dataset.json` copy and plans. |
| `nanounet/plan/resenc/planner_resenc.py` (180) | `resenc_3d_fullres_plan`: topology, VRAM shrink loop, batch size, resampling kwargs | Batch: `round((ref/est) * REF_BS_3D)`, `ref = preset.reference_val_3d * (vram/preset.reference_val_corresp_gb)`, capped by dataset coverage, floor `MIN_BATCH`. Uses `estimate_conv_feature_map_size` from `nanounet/model/network.py`. |
| `nanounet/plan/plans.py` (199) | `Plans`, `Config3d` (properties `spacing`, `patch_size`, `batch_size`, `normalization_schemes`, ...) | Add properties for new keys here only if it stays < 200 LOC. |
| `nanounet/plan/prep/case_pp.py` (197) | `crop_normalize_case` computes `o_sp` (native, transposed), `t_sp = list(cm.spacing)`, `new_sh = compute_new_shape(...)`, normalizes; `run_case_npy` resamples data/seg, samples fg locations | **The single place** to make the target spacing per case. File is at 197 LOC: put new logic in another module and call it. |
| `nanounet/plan/prep/preprocess.py` (111) | `run_preprocess`: rmtree data folder, pool over cases, copy GT, `precompute_folder` (centroid sidecars) | Sidecars are regenerated automatically for the new folder. |
| `nanounet/data/volume/resampling.py` (197) | `resample_data_or_seg_to_shape`; separate-z when spacing ratio > `ANISO_THRESHOLD` (3, `nanounet/common.py`), axis = `get_lowres_axis` | Axis detection is automatic per call, so the per-case thick axis needs no change here. |
| `nanounet/data/volume/normalization.py` | `ZScoreNormalization`, `CTNormalization`, ... | Z-score exists. |
| `nanounet/prompt/centroids.py` | `_one_case` → `{centroids_zyx, bboxes_zyx, seed_zyx, volume_vox}` per `_centroids.json`; `apply_propagation_offset` (gaussian mode, voxels) | Voxel coords in the preprocessed grid; regenerated per data folder. |
| `nanounet/data/patch/error_table.py` | Empirical registration-error offsets; **being changed by the P0 task** to mm with a dataset-level `data_spacing_zyx` bound from `cm.spacing` | We extend it to per-case spacing (P4). |
| `nanounet/config.py` | `RoiPromptConfig`; `PropagatedConfig.data_spacing_zyx` (being added by P0); `point_radius_vox` | |
| `nanounet/train/patches/data_module.py`, `nanounet/cli/build_valset.py` | Call `bind_roi_spacing(cfg, cm.spacing)` (P0 adds this) | Per-case spacing must come from case properties in z-only mode. |
| `nanounet/data/patch/sampling.py` | `select_prompt_points` → `draw_propagated_offset`; reads `properties["volume_vox"]`; `_FALSE_POS_GUARD_VOX = 5` | |
| `nanounet/data/valset/manifest.py` | `SMALL_LESION_MAX_VOX = 500` | A5. |
| `nanounet/cli/lesion_weights.py` + `nanounet/plan/dataset/lesion_types.py` | d013 per-centroid weights `<id>_weights.json`; maps CSV cogs by **shape ratio** (`cog_to_preprocessed`), spacing-agnostic; `--max-match-dist 10` voxels, `--max-median-dist 8` gate | Rerun for the new folder; watch the gate. |
| `nanounet/model/network.py` | `build_net(cm, lm, dj, enable_deep_supervision, n_extra_in=N_PROMPT_CHANNELS)`; `estimate_conv_feature_map_size` | |
| `nanounet/model/mae_transfer.py` (64) | `load_mae_encoder` (encoder-only, zero-pads stem `STEM_WEIGHT` input channels, silently skips shape mismatches), `load_full_net` | Add the foundation loader here (room until 200). |
| `nanounet/train/module/lightning_module.py` (191) | Ctor kwargs incl. `enable_deep_supervision=True`, `mae_ckpt`, `init_weights`, `warmup_epochs`; `build_net` then `load_full_net` / `load_mae_encoder`; SGD is already `momentum=0.99, nesterov=True` | Near the LOC limit. Ctor kwarg names are frozen (R18); adding one is allowed. |
| `nanounet/cli/train_parser.py` (147), `nanounet/cli/train.py` (94), `nanounet/train/fit.py` | Flags, validation (`validate_train_args`), `train_config_rows` | `--lr` default 0.01, `--warmup-epochs` default 0, `--optimizer` default sgd, `--batch-size` default from plans. No deep-supervision flag exists. |
| `nanounet/infer/predict/case.py` | Uses `cm.patch_size`, passes `cm.spacing` to `resolve_pts_pad` | Point mapping (`nanounet/prompt/coords.py`) uses **shape ratios**, so it is spacing-agnostic. Prediction preprocesses through `case_pp.run_case`, so it inherits the per-case spacing automatically. |
| `nanounet/infer/export/volume.py`, `nanounet/infer/export/tiles.py` | Map predictions back: `cur_sp = cm.spacing if len(cm.spacing) == len(sh) else [sp_t[0], *cm.spacing]`, `tgt_sp = native` | Must use the per-case target spacing in z-only mode. |
| `nanounet/score.py` | NSD uses spacing read from the image files (original space) | No change. |
| `nanounet/scripts/slurm_final_900_h200.sh` | Reference launch script for the current model (env vars, staging to `/root/NanoUNet_preprocessed`, `BATCH_SIZE=12`, `PROMPTS_PER_PATCH=2`, `ITERS_PER_EPOCH=1000`) | Template for the P7 preprocessing script and the P10 training script. |

---

## 3. Environment and working rules

- The container is **ephemeral**: unpushed code is lost when the session ends. **Commit and `git push origin
  feat/nnfoundation-zonly` after every significant code change** (each new module, each wired-up feature, each
  docs batch), not only at phase ends. Never leave more than ~1 hour of work unpushed.
  Work on a branch (e.g. `feat/nnfoundation-zonly`), never commit to `main` directly.
- The image sets no `NANOUNET_*` variables. Export them in every shell:
  ```bash
  export NANOUNET_RAW=/nnunet_data/NanoUNet_raw
  export NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed
  export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results
  ```
  Do **not** stage data on this machine's local disk: the session and its disk are wiped when the job ends.
  Write all persistent data under `/nnunet_data`. Historically `/nnunet_data`
  is CIFS and is I/O-bound for training (~0.25 it/s vs ~1 it/s locally).
- SimpleITK pitfall: never write `sitk.GetArrayViewFromImage(sitk.ReadImage(f))` inline; keep the image in a
  variable first (use-after-free returns garbage that looks like a corrupt mount).
- Max 2 parallel `nanounet_predict` processes (40 GB cgroup). Watch RAM with `-np` during preprocessing:
  p95 case is 209 Mvox (≈0.84 GB float32 per array, several copies during order-3 resampling).
- Tests are temporary (R16): put check scripts in your scratchpad, run them, paste results in the report, delete.
- After code changes: `python .claude/skills/nanochat-style/scripts/check.py --changed` (fix all errors and
  warns on your lines) and `graphify update .`.
- Every new user-facing error: what is wrong, what was expected, `Fix: <literal command>` (see existing errors in
  `train_parser.py` for the house style). Trigger each new error once and paste the rendered text in your report.

---

## 4. Implementation phases

Each phase ends with its acceptance checks passing, a commit, and a push.

### P0. Dependency: dataset-level mm prompt offsets (separate task, already running)

A separate session (task "Fix registration-error offsets ignoring data spacing") is changing
`nanounet/data/patch/error_table.py`, `nanounet/config.py`, `nanounet/cli/build_valset.py`,
`nanounet/train/patches/data_module.py` and `nanounet/docs/reference/config.md` so that table offsets
(stored in voxels at the table's `spacing_zyx = [1.25, 0.781, 0.789]`) are converted to mm and back to voxels of
the **dataset's** spacing (`data_spacing_zyx`, bound from `cm.spacing` via `bind_roi_spacing`), and lesion
diameters use the data spacing.

- **Start P4 only after that work is merged** into your branch's base. Before P4: `git log`, read the merged
  diff, and confirm the function names (`bind_roi_spacing`, `sample_offset_mm`, `data_spacing_zyx`) — they may
  differ from what this plan assumes. P1–P3 and P5–P6 do not depend on it.

### P1. Foundation constants, download and verification

New module **`nanounet/model/foundation.py`** (docstring per R6; < 200 LOC):

- Constants (R9): `FOUNDATION_REPO = "MIC-DKFZ/nnFoundationCNN"`, `FOUNDATION_FILE = "checkpoint_final.pth"`,
  `NET_CLASS`, `ARCH_KWARGS`, `KW_REQUIRES_IMPORT`, `PATCH_SIZE` exactly as §2.2,
  `N_ENCODER_TENSORS = 448`.
- `fetch_foundation(cache_dir: str) -> str`: `hf_hub_download(FOUNDATION_REPO, FOUNDATION_FILE, cache_dir=cache_dir)`;
  return the local path. Cache dir: new env var **`NANOUNET_PRETRAINED`** (document it in the README env
  section); if unset raise with `Fix: export NANOUNET_PRETRAINED=/nnunet_data/NanoUNet_pretrained`. Network
  errors propagate with a fix line (`huggingface-cli login` / manual download path).
- `verify_foundation(path: str) -> dict`: `torch.load(..., weights_only=True)`; assert keys `network_weights`,
  `nnssl_adaptation_plan`; assert `arch_class_name == "ResEncL"`, `pretrain_num_input_channels == 1`,
  `recommended_downstream_patchsize == PATCH_SIZE`; build the net from `ARCH_KWARGS` with 1 input channel and
  check every `encoder.*` key of the checkpoint exists in the net with identical shape (expect 448/448). Return
  `{"checkpoint_path", "sha256", "repo", "n_encoder_tensors", "citations"}`. Compute sha256 in streamed chunks.

Acceptance: a scratch script calls both on a fresh cache dir; prints 448/448 and the sha256; a second call is a
cache hit (no download).

### P2. Foundation plans builder

New module **`nanounet/plan/resenc/foundation_plan.py`** (< 200 LOC). It replaces `run_plan` in foundation mode
and reuses planner pieces (`_norm_schemes` is not used; normalization is fixed by D1):

- Inputs: `dataset_id`, `plans_name` (default **`nnFoundationCNN_z1p0`**: `f"nnFoundationCNN_z{z:.1f}".replace(".", "p")`),
  `z_target_mm` (default 1.0), `gpu_mem_gb` (default: the ResEncL preset's `default_vram_gb`), `foundation_info`
  from P1.
- `zonly_target_spacing(native_sp: list[float], z_target: float) -> tuple[list[float], int]`, **this is the one
  rule used everywhere** (preprocessing, plan median shape, docs): `ax = argmax(sp)` if
  `sp[argmax] > THICK_AXIS_RATIO * sp[0]` (`THICK_AXIS_RATIO = 1.25`, A3) else `0`; `t = list(sp); t[ax] = z_target`;
  return `(t, ax)`. Expected:
  - `(2.5, 0.76, 0.77) → (1.0, 0.76, 0.77), ax 0`
  - `(5.0, 0.80, 0.80) → (1.0, 0.80, 0.80), ax 0`
  - `(0.70, 0.78, 0.78) → (1.0, 0.78, 0.78), ax 0` (thin case: 0.70 → 1.0 mm is a downsample in z; intended)
  - `(0.98, 0.98, 4.0) → (0.98, 0.98, 1.0), ax 2` (KiTS)
  - `(1.5, 1.5, 1.5) → (1.0, 1.5, 1.5), ax 0` (PanTS isotropic)
- Plans JSON (same top-level schema `run_plan` writes, see `planner.py` lines ~155–170):
  `transpose_forward = transpose_backward = [0, 1, 2]` (no dataset-level transpose; the thick axis is per case);
  `original_median_spacing_after_transp`, `original_median_shape_after_transp` from the fingerprint as today;
  `foreground_intensity_properties_per_channel` copied (unused by Z-score, kept for schema compatibility);
  plus top-level **`pretrain_info`** = P1's dict.
- `configurations["3d_fullres"]`:
  - `data_identifier = f"{plans_name}_3d_fullres"`; **assert** it differs from every existing
    `configurations.*.data_identifier` of other plans JSONs in the dataset folder (in particular
    `nnUNetPlans_3d_fullres`), raise with a fix if not.
  - `spacing_mode = "z_only"`, `z_target_mm = 1.0`, `thick_axis_ratio = 1.25` (new keys; frozen once written, R18).
  - `spacing = [z_target, median_y, median_x]`: a **nominal** spacing (median of the per-case targets from the
    fingerprint), only for display and for any consumer that needs one number. Real per-case spacing lives in
    each case's properties (P3).
  - `patch_size = PATCH_SIZE`; `architecture = {"network_class_name": NET_CLASS, "arch_kwargs": ARCH_KWARGS,
    "_kw_requires_import": KW_REQUIRES_IMPORT}`.
  - `batch_size`: estimator as in `planner_resenc.py` (`estimate_conv_feature_map_size(PATCH_SIZE, n_in, n_out, ...)`,
    `ref = preset.reference_val_3d * (gpu_mem_gb / preset.reference_val_corresp_gb)`,
    `round(ref / est * REF_BS_3D)`, floor `MIN_BATCH`), then **round down to even** (A4). Reuse the constants
    from `planner_resenc.py`; do not copy them.
  - `median_image_size_in_voxels`: median over fingerprint cases of `compute_new_shape(shape, sp, zonly_target_spacing(sp))`.
  - `normalization_schemes = ["ZScoreNormalization"]`, `use_mask_for_norm = [False]`.
  - Resampling kwargs: data `{"is_seg": False, "order": 3, "order_z": 1, "force_separate_z": None}` (A2);
    seg `{"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None}`;
    probabilities `{"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None}`; function names
    `resample_data_or_seg_to_shape` as today. `batch_dice: False`.
- Write with the existing `_save_plans` (import it; do not duplicate).

Acceptance: build the plans for Dataset900 into a scratch copy of the dataset folder (copy `dataset.json`,
`dataset_fingerprint.json`); check `architecture.arch_kwargs == ARCH_KWARGS`, patch 192³, batch at
`--gpu-memory-gb 141` is 12, `data_identifier == "nnFoundationCNN_z1p0_3d_fullres"`, `spacing_mode == "z_only"`;
`build_net` on it + P1's verify → 448/448.

### P3. Per-case z-only resampling in preprocessing (and therefore prediction)

- `nanounet/plan/plans.py` `Config3d`: add read-only properties `spacing_mode` (default `"median"` when the key
  is absent, so old plans are untouched), `z_target_mm`, `thick_axis_ratio`. Keep the file < 200 LOC.
- `nanounet/plan/prep/case_pp.py` `crop_normalize_case`: replace the `t_sp = list(cm.spacing)` block with a call
  that returns `t_sp` and the thick axis: in `z_only` mode `t_sp, ax = zonly_target_spacing(o_sp, cm.z_target_mm)`
  (import from `foundation_plan.py`; the ratio comes from `cm.thick_axis_ratio`); otherwise today's logic exactly.
  Record **`properties["spacing_after_resampling"] = t_sp`** (transposed axis order, floats) and
  **`properties["resampled_axis"] = ax`** in both modes (new sidecar keys, frozen by R18). The file is at 197
  LOC: the change must be net ≤ +2 lines; move the old 3-line `t_sp` logic into the helper module if needed.
- Nothing in `resampling.py` changes: `determine_do_sep_z_and_axis` picks the low-res axis from the case's
  spacings, so KiTS axis 2 is handled.
- `nanounet/infer/export/volume.py` and `tiles.py`: in `z_only` mode `cur_sp = props["spacing_after_resampling"]`;
  if the key is missing in `z_only` mode raise (R12) with
  `Fix: re-run nanounet_preprocess for this plan (the case was preprocessed before spacing_after_resampling existed)`.
  In other modes keep today's expression unchanged.
- `nanounet/infer/predict/case.py` passes `cm.spacing` into `resolve_pts_pad`; confirm by reading
  `nanounet/prompt/coords.py` that voxel- and world-space click mapping uses only shape ratios (it does at the time
  of writing). If anything uses `spacing` numerically there, switch it to `props["spacing_after_resampling"]`.

Acceptance (scratch script on 2 cases per cohort = 42 cases, written to a scratch output dir, **not** the dataset
folder): for each case, `data.shape[1:] == round(shape_after_crop * o_sp / t_sp)`; KiTS cases resample axis 2;
per-case mean ≈ 0 and std ≈ 1 inside the image; seg label set preserved; round trip (resample seg back with the
export code path to the original grid) gives Dice ≥ 0.95 against the original label for every case with
foreground; prediction preprocessing (`run_case` without seg) gives the same image array as training preprocessing.

### P4. Per-case physical prompt geometry (after P0 is merged)

- Wherever P0 uses the dataset-level `data_spacing_zyx`, use the **case's** `properties["spacing_after_resampling"]`
  when `cm.spacing_mode == "z_only"` (training sampler, valset builder). Keep P0's dataset-level path for old plans.
  The cleanest seam is passing a `spacing_zyx` argument into `draw_propagated_offset` / `sample_offset_*`
  from the caller that already has `properties`; check P0's final signatures first.
- `nanounet/data/valset/manifest.py`: replace `SMALL_LESION_MAX_VOX = 500` with `SMALL_LESION_MAX_DIAM_MM = 10.0`
  (A5) and compare equivalent-sphere diameters computed with the case spacing (reuse the diameter function from
  `error_table.py`). Update the manifest header key accordingly (`small_lesion_max_diam_mm`); valset schema
  version bump if the manifest has one (`SCHEMA_VERSION`).
- Leave `point_radius_vox` (an encoding size, not a measurement), `_FALSE_POS_GUARD_VOX` and the gaussian-mode
  `sigma_per_axis`/`max_vox` in voxels; list them in your report as "voxel-unit by design".

Acceptance: on 5 d013 cases and 5 KiTS cases, draw 10k offsets each and print mean |offset| in **mm** per axis;
it must match the table's mm statistics (convert the table with its own `spacing_zyx`) within sampling noise for
both the 2.5 mm plan and the new plan.

### P5. Foundation encoder loader

In `nanounet/model/mae_transfer.py` add `load_foundation_encoder(seg_net, ckpt_path) -> dict`:

1. `ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)`; `pre = ck["network_weights"]`; keep only
   keys starting with `encoder.` (D8; never `decoder.*`).
2. For each key in `seg_net.state_dict()` that is in `pre`:
   - same shape → copy;
   - 5-D conv weight whose **input-channel** count differs (stem; checkpoint has 1, our net has
     `n_img + N_PROMPT_CHANNELS`): put the pretrained weights in the image channel(s) (if the dataset has more
     than one image channel, repeat as nnU-Net does) and **zeros** in the prompt channel(s);
   - 5-D conv weight whose **kernel** differs only where target size is 1 and source is 3 → `mean` over that axis
     (`DynamicPretrainedTrainer` behaviour); should not happen with the foundation plan but keeps the loader honest;
   - anything else → collect as a mismatch.
3. `seg_net.load_state_dict({**seg_net.state_dict(), **new}, strict=True)`.
4. If loaded count < `N_ENCODER_TENSORS` or any mismatch: raise with the counts and
   `Fix: preprocess with the foundation plan (nanounet_preprocess -d <id>) so the network matches nnFoundationCNN`.
5. Log (logger already used in the file) `[foundation] loaded 448 encoder tensors; decoder/seg head from scratch`.

Wire-up in `nanounet/train/module/lightning_module.py`: new ctor kwarg `foundation_ckpt: str | None = None`;
precedence `init_weights` > `foundation_ckpt` > `mae_ckpt` (A6). Keep the file < 200 LOC (move a few lines into
`mae_transfer.py` if needed, e.g. a `load_init_weights(net, init_weights, foundation_ckpt, mae_ckpt)` helper).

Acceptance: scratch script builds the net from the P2 plans and calls the loader → 448 loaded, prompt-channel
stem weights all zero, image-channel stem weights equal the checkpoint, a forward pass on `(1, 2, 192, 192, 192)`
runs; calling it on the old `nnUNetResEncUNetLPlans_h200_smallpv` plans net succeeds via kernel averaging
(6 adapted, 448 loaded) or fails with the documented error if you chose to forbid it — say which.

### P6. CLI defaults: preprocess and train

`nanounet/cli/preprocess.py` (keep < 200 LOC; push logic into `foundation_plan.py`):

- New flags: `--no-foundation` (restore today's planner path exactly), `--target-z` (float mm, default 1.0;
  only in foundation mode). Foundation mode is the **default**. In foundation mode `--planner` and `--patch-vol`
  are ignored: show that in `config_table` (source "foundation"); `--gpu-memory-gb` still sets the batch size.
- Foundation mode flow: `fetch_foundation` + `verify_foundation` → `foundation_plan` → `run_preprocess`.
  Use `--skip-fingerprint` semantics as today (reusing the existing fingerprint is the expected use for Dataset900).
- Splits safety (D6), **both modes**: if `splits_final.json` already exists, never rewrite it. Instead verify every
  id in it exists in the new data folder (and vice versa for training ids) and print
  `✓ splits kept (sha256 …)`; on mismatch raise with the missing ids and a fix. Same for `cohorts.json`: keep the
  existing file, recompute in memory, and raise if it differs. Only create them when absent.
- Valset output name includes the plans: `valset_<n>_<plans>.json` (never overwrites `valset_2000.json`).
- `next:` line: print the train command for the new plans.

`nanounet/cli/train_parser.py` / `train.py` / `fit.py`:

- New flags: `--no-foundation`, `--deep-supervision {auto,on,off}` (default `auto`).
- If the plans JSON has `pretrain_info` and not `--no-foundation`: pass `pretrain_info["checkpoint_path"]` as
  `foundation_ckpt`; verify the file exists and its sha256 matches, else raise with
  `Fix: nanounet_preprocess -d <id> --skip-fingerprint --plans-name <plans>` (which re-fetches). Plans without
  `pretrain_info` train exactly as today.
- Foundation defaults (D4), applied only when the user did not pass the flag (switch the relevant argparse
  defaults to `None` and resolve after parsing; show source "foundation default" vs "cli" in
  `train_config_rows`): `--optimizer sgd`, `--lr 1e-3`, `--warmup-epochs 2` (D4 units note), deep supervision off
  (`auto` → off with foundation, on otherwise), `--lr-schedule poly`. Non-foundation defaults stay exactly as today
  (lr 0.01, warmup 0, DS on).
- Conflicts (A6): `--mae-ckpt` or `--mae-pretrain` with an active foundation plan → error unless `--no-foundation`;
  `--init-weights` → foundation skipped, logged in the config table.
- `enable_deep_supervision` already flows to `build_net` and `build_loss`; plumb the resolved value through.

Acceptance: `nanounet_preprocess --help` and `nanounet_train --help` read cleanly; trigger each new error once;
`nanounet_train` on the foundation plans shows lr 1e-3 / warmup 2 / DS off / foundation ckpt in the config table;
on the old plans it shows today's defaults.

### P7. Full preprocessing of Dataset900 (SLURM job, writes straight to /nnunet_data)

Target folder: `/nnunet_data/NanoUNet_preprocessed/Dataset900_Merged/` (the same folder the current data lives in;
the new data goes into its own `nnFoundationCNN_z1p0_3d_fullres/` subfolder, P2 guarantees the name).
`NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed` for every step.

1. Record the sha256 of `splits_final.json` and `cohorts.json` (regenerated 2026-10-02, values in §2.4; paste them in the report). They must be unchanged after the job.
2. Measure first (in your session): run the P3 code on a 50-case cohort-stratified subset into a scratch output
   under `/nnunet_data` (e.g. `/nnunet_data/NanoUNet_preprocessed/_probe_nnFoundationCNN_z1p0/`, deleted afterwards)
   to get seconds per case, peak RAM per worker, bytes per case and CIFS write throughput. Extrapolate total wall
   time and size for 5690 cases and pick `-np` from the RAM numbers. Report before H1.
3. Write `nanounet/scripts/slurm_preprocess_foundation_900.sh`, modelled on the header/env/container block of
   `nanounet/scripts/slurm_final_900_h200.sh` (same image and `/nnunet_data` mount; CPU-heavy: many cpus, enough
   `--mem` for `-np` workers from step 2; a GPU is not needed unless the partition requires one). It runs:
   ```bash
   nanounet_preprocess -d 900 --skip-fingerprint --target-z 1.0 --gpu-memory-gb 141 -np <N> --resume   # fingerprint computed beforehand (2026-10-02)
   ```
   `--resume` makes resubmits continue where they stopped. It must **not** create a new `splits_final.json`
   (P6 splits safety) and must print the sha256 check at the end. Commit and push; **STOP at H1** for the owner to
   submit it.
4. After the job: checks — case count equals the old folder (5690 `.pkl`; `.b2nd` ×2; one `_centroids.json` per
   case); sha256 of `splits_final.json` and `cohorts.json` unchanged; every split id present; spot-check 5 KiTS
   cases have `resampled_axis == 2`; centroid count per case equals the number of connected components of the new seg.
5. d013 lesion weights for the new plans:
   ```bash
   nanounet_lesion_weights -d 900 --plans nnFoundationCNN_z1p0 --meta-dir <same meta dir as the current run>
   ```
   (find the meta dir and any extra flags in `nanounet/docs/steps/lesion_weights.md` and the existing
   `<id>_weights.json` provenance); the median match-distance gate must pass. `--max-match-dist` is in voxels;
   report the median distance in mm as well.
6. Valset: rebuild with the same patch count and config as `valset_2000.json` (read its header for config path
   and seed):
   ```bash
   nanounet_build_valset -d 900 --plans nnFoundationCNN_z1p0 --config nanounet/configs/longrun900.json \
     --out /nnunet_data/NanoUNet_preprocessed/Dataset900_Merged/valset_2000_nnFoundationCNN_z1p0.json --n-patches 2000
   ```
   Steps 5–6 are light enough to run in an interactive session or at the start of the training job; either way the
   outputs land in `/nnunet_data`. Never touch an existing file there.

### P8. Throughput and smoke training

- GPU gate (G4, `references/gpu.md` in the skill): on a node-local copy of a cohort-stratified subset (D11), measure
  step time and data-loader throughput at batch 12, 10 and 8. Report it/s, s/epoch at `--iters-per-epoch 1000`, and
  GPU utilization (must be compute-bound, not loader-bound).
- Copy throughput: time an rclone copy of a few GB from `/nnunet_data` to local disk (new network link) and
  extrapolate the full staging time for the new data folder; it goes into the P10 budget.
- Smoke run (local subset, no W&B):
  ```bash
  nanounet_train -d 900 -f 0 --plans nnFoundationCNN_z1p0 --config nanounet/configs/longrun900.json \
    --epochs 2 --iters-per-epoch 50 --val-iters 10 --prompts-per-patch 2 --no-wandb
  ```
  Expect: foundation loaded (448), loss decreasing, checkpoint written. Then `nanounet_predict` on 2 val cases
  (one d013, one KiTS) with that checkpoint; outputs have the original image's size, spacing, origin and direction.

### P9. Docs (same change as the code, D4/D6 of the skill)

Update: `nanounet/docs/steps/preprocess.md` (new flags, foundation default, splits safety, valset naming),
`plan.md` (foundation mode bypasses the planner), `train.md` (foundation defaults table, new flags, conflicts),
`predict.md` (per-case spacing), `valset.md` (mm threshold), `lesion_weights.md` (rerun per plans),
`nanounet/docs/reference/config.md`, `nanounet/docs/index.md` quickstart, `nanounet/README.md` env section
(`NANOUNET_PRETRAINED`), and cite nnFoundation (bibtex from the ckpt) where the README lists external work.
Mention the CC-BY-SA-4.0 license of the weights.

---

### P10. SLURM script for the supervised run

Write `nanounet/scripts/slurm_foundation_900_h200.sh` by copying the structure of
`nanounet/scripts/slurm_final_900_h200.sh` (read it fully first: SBATCH header, container mounts, env exports,
local staging to `/root/NanoUNet_preprocessed`, the resume state machine,
`FRESH`/`SKIP_SUP` guards). Changes:

- `PLANS_NAME=nnFoundationCNN_z1p0`; stage the **new** data folder, plans JSON, sidecars, lesion weights and
  `valset_2000_nnFoundationCNN_z1p0.json` (not the old folder); `--val-manifest` points at the new valset.
- **No MAE stage** (drop `MAE_*`, `--mae-*`); the foundation encoder comes from the plans' `pretrain_info`.
  Stage the checkpoint (`NANOUNET_PRETRAINED`) to the node too and check its sha256 before training.
- Foundation recipe: rely on the P6 defaults (SGD, lr 1e-3, warmup 2, poly, DS off) and print them; do not
  re-pass them as flags unless you must. Keep `PROMPTS_PER_PATCH=2`, `CONSISTENCY_WEIGHT=0.02`, `EMA_DECAY=0.999`,
  `VAL_EVERY_N=2`, `ITERS_PER_EPOCH=1000`, `ROI_CONFIG=nanounet/configs/longrun900.json`; batch size from the plans
  (expect 12; drop the old `BATCH_SIZE=12` override or set it from the plans).
- **Epoch budget: roughly 4 days of supervised training (D10; a target, not a hard cap).** One 192³ patch is 3× the voxels of the old
  64×192² patch; expect about 1.5 s/step at batch 12 (~25 min per 1000-iter epoch; the old run was 578 s), less at
  batch 8–10. From the P8 numbers pick the batch (12, 10 or 8; even, A4) and compute
  `SUP_EPOCHS ≈ (96 h − rclone staging − ~3 h margin) × 3600 / s_per_epoch`; put the arithmetic in the script header
  and confirm with the owner at H2. For scale: ~200 epochs at batch 12 = 2.4 M patches, about 5× nnU-Net's full
  1000-epoch finetune default (500k patches); a pretrained encoder needs fewer updates than our 1200-epoch scratch
  run. The poly schedule is sized to `SUP_EPOCHS`, so it ends on time. The d013 finetune stage comes after and is
  not part of the 4 days.
- **Data source (D11).** Always stage to node-local disk with rclone at job start, reusing the old script's staging
  block (same source/destination layout, gap-filling copy on resubmit) but pointed at the new plans JSON, the new
  `nnFoundationCNN_z1p0_3d_fullres/` folder (with sidecars and lesion weights), `splits_final.json`, `cohorts.json`,
  `dataset.json` and `valset_2000_nnFoundationCNN_z1p0.json`. Do not copy the old `nnUNetPlans_3d_fullres/` folder.
  Point `NANOUNET_PREPROCESSED` at the local copy for training.
- The d013 finetune stage (`FT_*`) stays in the script, pointed at the new plans and
  `nanounet/configs/finetune900_d013.json`, with `--init-weights` from the supervised run (A6: foundation skipped).
- Header comment: what the job does, expected wall time, resume instructions, output dir
  (`${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_foundation`).
- `bash -n` the script; dry-run its non-GPU parts if possible (staging, guards). Do not `sbatch`.

## 5. Order and commits

- **Stage A (unblocks preprocessing):** P1 → P2 → P3 → the preprocess half of P6 → P9 for preprocess/plan docs →
  P7 steps 1–3 (sha256s, 50-case measurement, preprocessing SLURM script). Push, then **STOP at H1**.
- **Owner (H1):** submits the preprocessing job. It runs for hours; Stage B proceeds meanwhile. P7 step 4 checks
  run when it finishes.
- **Stage B:** (P0 merged) P4 → P5 → the train half of P6 → remaining P9 docs → P7 steps 5–6 (lesion weights,
  valset; need P4 and the finished data) → P8 (throughput, smoke) → P10 → **STOP at H2**.

Commit and push after every significant code change, and at least once per phase (container is ephemeral; unpushed work is lost). Every commit passes the nanochat-style
checker (§0.2). Each commit message ends with the attribution line the session tells you to use.

## 6. Out of scope (do not do)

- Any control/ablation run (D5). Submitting the SLURM job (the owner does that at H2).
- Loading the nnFoundation decoder (D8). Reorienting images (D3). The nnFoundationViT/Primus checkpoint.
- Changing anything for old plans beyond the shared safety fixes (splits/cohorts/valset naming) and the new
  sidecar keys.

## 7. Report back with

Rule IDs touched/waived and the checker result; every new error message rendered; P1/P2/P3/P5 acceptance
outputs; P7 timings, sizes, sha256 before/after, gate results; P8 throughput and wall-time extrapolation; the
author decisions A1–A6 as implemented, and anything in this plan that turned out to be wrong.

## 7. Implementation log (2026-10-02)

- Preprocess (P7): 5690 cases, 1.3 TB, run interactively on a 50-CPU/375 GB node, `-np 36`, about 3.7 h wall.
  The first 3 h ran at 12-30 cases/min because every `.b2nd` was saved with zstd clevel 8 on one thread (6-9x
  slower than clevel 3, files 4 % larger). `save_case` now defaults to clevel 3; cases written earlier keep clevel 8
  (identical decoded content).
- Frozen files unchanged: `splits_final.json` 1016c2be..., `cohorts.json` ec46f4d0...; test-patient guard 0 of 60.
- Lesion weights on the new grid: 98.9 % of 5213 centroids matched, median match distance 0.98 voxel (gate 8).
- GPU gate (H200, 24 CPUs, node-local 20 GB cohort-stratified subset of 84 train cases, rclone copy 870 MB/s so ~25-40 min
  for the full 1.3 TB): batch 12 peaks at 133 of 143 GB (~2 s/step, loader-limited here); batch 8 peaks at 94 GB,
  1.15 s/step, GPU util median 93 %. Chosen: batch 8, `SUP_EPOCHS=250` (~90 h incl. val). An xl loader (16 workers)
  is killed by an 80 GB cgroup, so size RAM >= 150 GB for the job (the SLURM script asks for 200 GB).
- Smoke run (3 epochs, foundation loaded 448, guard 0 of 60) and `nanounet_predict` on a d013 and a KiTS case
  (size/spacing/origin/direction equal to the input, 1.6 s per case on GPU) passed.
- CPU `nanounet_predict` with the default `--batch-size 8` at a 192^3 patch exhausts RAM (>280 GB, silently
  OOM-killed); use `--batch-size 1` on CPU. The GPU path clamps the batch to free VRAM.
