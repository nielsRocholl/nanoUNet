# Concept subfolders (R20)

Date: 2026-09-29
Status: implemented on branch `nanounet-subfolders` (pure move, gated by the equivalence harness)

## Goal

Every module path says what the file is: `nanounet/<area>/<concept>/<module>.py`. Rule R20 in
`.claude/skills/nanochat-style/SKILL.md`, enforced by `check.py` (flat area > 6 modules, one-module
subfolder, depth > 2, name repeating its folder, subfolder `__init__` without docstring).

## Decisions

- **Grouped:** `data/` (16 flat → 6 concepts), `infer/` (11 → 3), `model/` (7 → `loss/` + 3 flat), `train/`
  (7 → 2 concepts + `fit.py`), `plan/` (6 flat → `dataset/` + `plans.py`, `labels.py`), and
  `dataloader_prefs.py` joins `data/loader/`.
- **Flat, by rule:** `cli/` (1:1 with `[project.scripts]`, so no entry-point or reinstall churn), `prompt/` (4),
  `diag/` (4), `pretrain/` (3), and root `common.py config.py runtime.py lightning_ckpt.py score.py`
  (K16: `slurm_final_900_h200.sh` imports `lightning_ckpt`; K17: `common.py` and `runtime.py` depth is frozen).
- **Renames only where the folder already says it:** `valset_build` → `valset/build`, `predict_case` →
  `predict/case`, `dice_loss` → `loss/dice`, `patch_bbox` → `patch/bbox`, `patch_iterable` → `patches/iterable`,
  `dataset_id` → `dataset/ids`. Where the folder would collide with the old module name, the file gets a noun:
  `augment.py` → `augment/transforms.py`, `valset.py` → `valset/manifest.py`, `export.py` → `export/volume.py`,
  `patch_export.py` → `export/tiles.py`, `segtrack.py` → `segtrack/track.py`.
- **No shims.** On-disk contracts are names, not module paths: `EMACallback` is PL `state_key` = `__qualname__`
  (K4). Hparams are primitives (K6). plans.json stores `resample_data_or_seg_to_shape` (split on `.`, last part)
  and `SimpleITKIO` (the `.SimpleITKIO` suffix match survives a dotted path). Spawn/DataLoader pickles (K10/K11) resolve
  by module path at runtime, and every importer moves in the same commit.
- **Bodies are byte-identical.** Only import lines change. There are two in-function imports: `plan/prep/case_pp.py`
  (`Blosc2Folder`) and `model/loss/losses.py` (`CC_DC_and_CE_loss`). `train/patches/data_module.py` keeps the
  `augment.` prefix via `from nanounet.data.augment import transforms as augment`.

## Move table

| old (`nanounet/`) | new |
|---|---|
| `data/io.py` | `data/store/io.py` |
| `data/blosc2_dataset.py` | `data/store/blosc2_dataset.py` |
| `data/crop.py` | `data/volume/crop.py` |
| `data/normalization.py` | `data/volume/normalization.py` |
| `data/resampling.py` | `data/volume/resampling.py` |
| `data/augment.py` | `data/augment/transforms.py` |
| `data/spatial_points.py` | `data/augment/spatial_points.py` |
| `data/sampling.py` | `data/patch/sampling.py` |
| `data/patch_bbox.py` | `data/patch/bbox.py` |
| `data/instance_target.py` | `data/patch/instance_target.py` |
| `data/error_table.py` | `data/patch/error_table.py` |
| `data/cohorts.py` | `data/patch/cohorts.py` |
| `data/valset.py` | `data/valset/manifest.py` |
| `data/valset_alloc.py` | `data/valset/alloc.py` |
| `data/valset_build.py` | `data/valset/build.py` |
| `data/loader_workers.py` | `data/loader/workers.py` |
| `dataloader_prefs.py` | `data/loader/prefs.py` |
| `infer/predict_case.py` | `infer/predict/case.py` |
| `infer/predict_io.py` | `infer/predict/io.py` |
| `infer/predictor.py` | `infer/predict/predictor.py` |
| `infer/tta.py` | `infer/predict/tta.py` |
| `infer/roi_slices.py` | `infer/predict/roi_slices.py` |
| `infer/points_pad.py` | `infer/predict/points_pad.py` |
| `infer/inference_row.py` | `infer/predict/inference_row.py` |
| `infer/export.py` | `infer/export/volume.py` |
| `infer/patch_export.py` | `infer/export/tiles.py` |
| `infer/segtrack.py` | `infer/segtrack/track.py` |
| `infer/segtrack_case.py` | `infer/segtrack/case.py` |
| `model/losses.py` | `model/loss/losses.py` |
| `model/dice_loss.py` | `model/loss/dice.py` |
| `model/cc_dice_ce.py` | `model/loss/cc_dice_ce.py` |
| `model/dice_metrics.py` | `model/loss/dice_metrics.py` |
| `train/data_module.py` | `train/patches/data_module.py` |
| `train/patch_iterable.py` | `train/patches/iterable.py` |
| `train/patch_render.py` | `train/patches/render.py` |
| `train/lightning_module.py` | `train/module/lightning_module.py` |
| `train/ema.py` | `train/module/ema.py` |
| `train/val_metrics.py` | `train/module/val_metrics.py` |
| `plan/dataset_id.py` | `plan/dataset/ids.py` |
| `plan/splits.py` | `plan/dataset/splits.py` |
| `plan/cohorts.py` | `plan/dataset/cohorts.py` |
| `plan/lesion_types.py` | `plan/dataset/lesion_types.py` |

## New subpackages

| folder | `__init__` docstring |
|---|---|
| `data/store/` | On-disk formats: blosc2 preprocessed cases and SimpleITK raw images. |
| `data/volume/` | Whole-volume array ops shared by preprocessing and inference: crop, resample, normalize. |
| `data/augment/` | Training augmentation chains and the click-carrying spatial transforms. |
| `data/patch/` | Training patch sampling: case draw, patch bbox, click jitter, click-conditional targets. |
| `data/valset/` | Fixed validation manifest: schema and dataset, patch-budget allocation, offline build. |
| `data/loader/` | DataLoader plumbing: fixed worker/prefetch presets, worker startup, collate. |
| `infer/predict/` | Prompt-ROI prediction engine: ckpt load, host IO, click padding, ROI tiles, TTA, batched logits. |
| `infer/export/` | Logits and tile segs back to native scanner space, NIfTI write. |
| `infer/segtrack/` | SegTrack: one-shot BL/FU prediction and lesion linking, case pairing. |
| `model/loss/` | Training losses (DC+CE, CC-DiceCE) and validation Dice on the shared tp/fp/fn core. |
| `train/patches/` | Supervised data side: LightningDataModule, patch iterable, keypoint/heatmap rendering. |
| `train/module/` | Supervised model side: LightningModule, weight EMA callback, validation metric logging. |
| `plan/dataset/` | Dataset-level metadata: id resolution, splits, cohort weights, lesion-type weights. |

## Gate

The `equiv/` harness was restored from `5b3b265^`, run untracked, and not committed. It ran with `--base 0ca3bac` and
`renames.json` = the table above. Result: `ast_guard ok (513 = 513 defs, 0 failures) | golden 486 keys | surface 21 keys | 0 diffs`.
The CLI surface covers --help of all 8 scripts, import side effects, and a spawn pickle probe (K10/K11).
