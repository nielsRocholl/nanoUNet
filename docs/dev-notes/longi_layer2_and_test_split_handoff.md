# Longitudinal Layer 2 + test-split — session handoff (2026-07-09)

## TL;DR

The longitudinal (`--longi`) two-stream model now trains on **real registered union clicks** in
both streams instead of mask-derived, jittered guesses. Rebuilt the registered dataset as
**`Dataset114_longi`** (229 train cases, test-free), carved the test patients into
**`Dataset115_longi_test`** (57 cases), and re-published the native validation set to HuggingFace.
All code is on `main`. Data lives on `/nnunet_data` (host: `/data/oncology/experiments/universal-lesion-segmentation`).

## The core problem this session fixed

Each longitudinal case = a baseline (BL) + follow-up (FU) scan. Every lesion in the BL∪FU **union**
must have a click in **both** timepoints' JSONs (disappeared lesions get a propagated FU point;
new lesions get a back-propagated BL point) so the two-stream model gets co-located prompts and can
learn appearance/disappearance. The old pipeline dropped per-timepoint lesions and the FU-stream
prompt was built from `cc3d` on the FU GT mask (real FU lesions only, then jittered).

**Two layers:**
- **Layer 1 (data):** per-case union click files `clicksBL/` + `clicksFU/` (+ `lesions/` manifest)
  on the registered root, cross-filled from the meta CSV propagation columns. Already generated:
  `/nnunet_data/unprocessed-universal-lesion-segmentation-registered-unigradicon/{clicksBL,clicksFU,lesions,clickfix_report.csv}`
  — 287 `ok`, 20 `partial` (unsanitized fallback points → excluded from training).
- **Layer 2 (code, this session):** make training + build consume those union clicks.

## Code changes on `main`

| Commit | Change |
|---|---|
| `3433a87` | `build_patch_longi` FU-stream prompt ← `prop["fu_clicks_zyx"]` (BL∪FU union incl. disappeared), not `centroids_zyx`; drop stale `centroid_weights` (uniform sampling) |
| `30cf3cf` | `longi_clicks` drops out-of-bounds mapped clicks (counted, not asserted); fix missing `encode_points_to_heatmap_pair` import in `sampling.py` |
| `2183916` | `longi_build` excludes cases not `status==ok` in `clickfix_report.csv` |
| `6e5d67e` | `longi_build` **binarizes** instance-labeled `targetsTrFU` before writing `labelsTr` (mirrors uclp-pro) |
| `c083c51` | `prompt_channels(jitter=…)`; longi FU-stream calls `jitter=False` — real registered clicks must not be re-jittered |
| `3dff982` | merge of the above (PR #2) |
| `1f6db73` | inference rewrite: joint 2-channel clustered baseline stream (see below) |

New CLI surface: `longi_build` also writes `clicksTrFU/`; `longi_clicks` takes `--clicks-fu-dir` and
writes `<case>_fu_clicks.json`; `build_patch_longi` **asserts** `fu_clicks_zyx` present.

## Pipeline to (re)build the training dataset

```bash
export NANOUNET_RAW=/nnunet_data/nnUNet_raw
export NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results

nanounet_longi_build \
  --register-out /nnunet_data/unprocessed-universal-lesion-segmentation-registered-unigradicon \
  --template-dj /nnunet_data/nnUNet_raw/Dataset013_Longitudinal_CT/dataset.json \
  --out /nnunet_data/nnUNet_raw/Dataset114_longi          # excludes 20 non-ok + binarizes labels

# preprocess: plan reused from d113 for finetune compatibility (data_identifier nnUNetPlans_3d_fullres)
nanounet_preprocess -d 114 -np 16 --skip-fingerprint --skip-plan \
  --plans-name nnUNetResEncUNetLPlans_h200_smallpv

nanounet_longi_clicks -d 114 --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --clicks-dir    /nnunet_data/nnUNet_raw/Dataset114_longi/clicksTr \
  --clicks-fu-dir /nnunet_data/nnUNet_raw/Dataset114_longi/clicksTrFU
```

Preprocessing here was run **per-case in isolated subprocesses** (the 32-worker spawn Pool crashed
with "worker died" — transient memory pressure, not a bad case; every case passed in isolation).
Post-steps (`gt_segmentations/`, `_centroids.json`) were completed with a `--resume` pass.
`splits_final.json` was materialized with `nanounet.plan.splits.make_splits` (5-fold, seed 12345).

## Datasets on disk (container `/nnunet_data` ⇄ host `/data/oncology/experiments/universal-lesion-segmentation`)

| Dataset | Path | Cases |
|---|---|---|
| Train — raw | `nnUNet_raw/Dataset114_longi` | 229 |
| Train — preprocessed | `NanoUNet_preprocessed/Dataset114_longi` | 229 |
| Test — raw | `nnUNet_raw/Dataset115_longi_test` | 57 |
| Test — preprocessed | `NanoUNet_preprocessed/Dataset115_longi_test` | 57 |
| HF eval (native val) | staged `hf_longi_registered_val_staging` → `huggingface.co/datasets/nielsRocholl/longi-registered` | 60 pat / 64 cases |

Test patients (`unprocessed-universal-lesion-segmentation/test_patients.csv`, 60) are **fully
quarantined** from `Dataset114_longi` — verified 0 test cases in the case dir and 0 in
`splits_final.json`. Only 54/60 test patients are represented (57 cases); the other 6 were among the
20 bad-registration cases dropped upstream (need re-registration to include).

## Key findings

- **HF eval must be native, not warped.** The **original** (pre-registration) dataset already ships
  complete paired union clicks in native frames (`inputsTrBL.json` = `cog_bl`∪`cog_backpropagated`;
  `inputsTrFU.json` = `cog_propagated`-preferred ∪ `cog_fu`). The registered set's same-named
  `inputsTr*.json` are per-timepoint incomplete. The HF dataset was re-pushed from the native
  originals; the warped `clicksBL` (misregistration bug on `f157be1f00_00`) is not eval-suitable.
- **Inference is asymmetric to training** (`1f6db73`): training warps BL into the FU frame; inference
  takes a **native BL scan that must already be registered into the FU frame** — `predict_io`
  hard-asserts BL geometry == FU geometry and points you to `nanounet_register_longi`.
- **Dataset113 has the same latent instance-label bug** (raw labels not binarized → `class_locations`
  foreground oversampling only matches id 1, silently under-sampling other lesion instances). It
  "worked" only via the train-time `(target>0)` collapse. Rebuild d113 with the binarize fix if used.

## Gotchas for the next session

- **Ephemeral repo resets**: `/root/nanounet` is reset between sessions and untracked work is lost.
  Push to a branch immediately and often. Remote may need `git remote set-url origin git@github.com:...` (SSH).
- **Stale container image = silent wrong training.** `nanounet_train` runs the nanoUNet baked into
  `nnunet-v2-pro-sol-docker:latest`. If that image predates `3433a87`, `build_patch_longi` uses the
  old `centroids_zyx` path and **won't error**. Rebuild the image from `main`, or add the guard:
  `python3 -c "import inspect,nanounet.data.sampling_longi as m; assert 'fu_clicks_zyx' in inspect.getsource(m.build_patch_longi)"`.
- Sampling change: d114 uses **uniform** patch sampling over union clicks; d113 used lesion-type
  **stratified** sampling. Not apples-to-apples — a Dice delta may partly be the sampling change.

## Training

`Dataset114_longi` is finetune-ready. Warm-start from
`NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/checkpoints/last.ckpt`.
AdamW 1e-5 / poly (right choice for a low-LR warm-started, parameter-light DWB finetune; SGD is the
from-scratch nnU-Net regime). See the finetune slurm script for the full command + the test-leak and
image-freshness guards.
