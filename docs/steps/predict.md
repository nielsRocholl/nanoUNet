# Predict

Prompt-driven GPU-batched inference over a dataset folder or a single case. Points are native scanner voxels `(x,y,z)`; mapping to preprocessed space is automatic.

Recommended: `--inference-mode clustered --merge max --border-expand` (TTA from `nano_config.json` unless overridden).

## Command

Dataset mode (folder of scans + sibling JSON):

```bash
nanounet_predict -i /path/to/scans -o /path/to/out -m /path/to/run \
  --ckpt last.ckpt --border-expand --batch-size 8 --device cuda
```

Single case:

```bash
nanounet_predict -i case.nii.gz -o seg.nii.gz --points case.json \
  -m /path/to/run --ckpt last.ckpt --border-expand
```

Centered mode (one patch per click):

```bash
nanounet_predict -i case.nii.gz -o seg.nii.gz --points case.json \
  -m /path/to/run --ckpt last.ckpt --inference-mode centered --border-expand
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-i`, `--input` | str | (required) | Folder (dataset) or single `.nii.gz` |
| `-o`, `--output` | str | (required) | Output folder (dataset) or single `.nii.gz` |
| `-m`, `--model-dir` | str | (required) | Run dir with `plans.json`, `dataset.json`, `nano_config.json`, checkpoint |
| `--ckpt` | str | auto | Checkpoint name/path; `auto` → `checkpoints/last.ckpt` |
| `--points` | str | none | Points JSON (**single mode only**) |
| `--baseline-image` | str | none | Sibling BL `.nii.gz` for two-stream longi inference |
| `--baseline-points` | str | none | BL click set JSON (**single mode**), same format as `--points`; native voxel `(x,y,z)` in the FU-registered frame |
| `--baseline-dir` | str | none | **Dataset mode** longi: dir with per-case BL `<cid>.nii.gz` + `<cid>.json` |
| `--longi` | flag | off | Force two-stream net build (else auto-detect from ckpt) |
| `--no-prompt-encode` | flag | off | Zero the 2 prompt channels |
| `--border-expand` | flag | off | Large-lesion extra border patches (BFS per cluster) |
| `--max-border-extra` | int | `16` | Max extra patches per cluster |
| `--tta` / `--disable-tta` | flag | from config | Force test-time augmentation on / off |
| `--batch-size` | int | `8` | GPU seed mini-batch (patches per forward) |
| `--num-workers` | int | `4` | CPU preprocess prefetch threads (dataset mode) |
| `--cluster-margin-frac` | float | `0.1` | Cluster bbox margin as fraction of patch size |
| `--inference-mode` | choice | `clustered` | `clustered` \| `centered` |
| `--merge` | choice | `max` | `max` = union (foreground-confident patch wins); `average` = legacy gaussian mean |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` (falls back if unavailable) |
| `--no-amp` | flag | off | Disable autocast (fp32) |
| `--overwrite` | flag | off | Re-run cases whose output exists |
| `--patients-csv` | str | none | CSV with patient column; filter cases by id prefix |

Points JSON format: `{"points": [{"name": "1", "point": [x, y, z]}, ...]}`. Empty `points` → all-background output.

## Checkpoint selection

`--ckpt auto` resolves to `last.ckpt` — correct for `-f all` runs but not always for holdout finetunes. For finetunes with real validation, pick the empirically best checkpoint (validation macro-Dice does not always track per-lesion DSC).

## Inputs / outputs

**Inputs**

- Model run dir (`plans.json`, `dataset.json`, `nano_config.json`, checkpoint)
- Scans (`.nii.gz`) and points JSON (dataset: sibling `<name>.json`; single: `--points`)

**Outputs**

- Dataset: `<out>/<case>.nii.gz` per input scan
- Single: `-o` segmentation file

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `missing points JSON for: …` | Dataset mode without sibling JSON | Add `<basename>.json` next to each scan |
| `single mode requires --points` | Single `.nii.gz` without points | Pass `--points` |
| `--baseline-points requires --baseline-image` | Longi flags mismatched (single mode) | Pass both or neither |
| `--baseline-dir is for dataset mode` | `--baseline-dir` in single mode | Use `--baseline-image`/`--baseline-points` |
| `dataset mode uses --baseline-dir` | `--baseline-image`/`--baseline-points` in dataset mode | Use `--baseline-dir` |
| `baseline given but checkpoint is not longi` | Baseline flags with non-longi ckpt | Drop `--baseline-*` or use a longi ckpt |
| `Baseline geometry does not match follow-up` | BL not registered into FU frame | Run `nanounet_register_longi` first |
| `Missing baseline files for longi dataset inference` | Missing BL siblings in `--baseline-dir` | Build with `nanounet_register_longi` |
| Missing checkpoint | Wrong `--ckpt` or incomplete train | Verify path under `checkpoints/` or `finetune/` |
| CUDA unavailable | No GPU | Use `--device cpu` or `mps` |

Longitudinal two-stream inference: [longi.md](longi.md).

## Preprocessed longi test inference (`nanounet_predict_preprocessed`)

For held-out test sets already in `NanoUNet_preprocessed/` (`.b2nd` + click sidecars). Skips raw NIfTI preprocess and scanner-space export; writes segmentations in **preprocessed resampled space** (same grid as `<case>_seg.b2nd`).

```bash
nanounet_predict_preprocessed \
  -m /nnunet_data/NanoUNet_results/nanounet/Dataset114_longi_nnUNetResEncUNetLPlans_h200_smallpv_f0_finetune_dwb \
  --ckpt finetune/last.ckpt \
  -i /nnunet_data/NanoUNet_preprocessed/Dataset115_longi_test/nnUNetPlans_3d_fullres \
  -o /nnunet_data/NanoUNet_preprocessed/Dataset115_longi_test/preds \
  --border-expand --batch-size 16 --num-workers 8
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-m`, `--model-dir` | str | (required) | Longi training run dir |
| `-i`, `--input` | str | (required) | Preprocessed `data_identifier` folder (`nnUNetPlans_3d_fullres`) |
| `-o`, `--output` | str | (required) | Output preds folder (`<case>.nii.gz` per case) |
| `--ckpt` | str | auto | Checkpoint; finetune runs live under `finetune/` |
| `--border-expand` | flag | off | Large-lesion extra border patches |
| `--max-border-extra` | int | `16` | Max extra patches per cluster |
| `--tta` / `--disable-tta` | flag | from config | Force TTA on / off |
| `--batch-size` | int | `16` | GPU patch mini-batch |
| `--num-workers` | int | `8` | CPU blosc2+pad prefetch threads |
| `--inference-mode` | choice | `clustered` | `clustered` \| `centered` |
| `--merge` | choice | `max` | `max` \| `average` |
| `--device` | choice | `cuda` | `cuda` \| `cpu` (CUDA required for practical throughput) |
| `--no-amp` | flag | off | Disable autocast |
| `--overwrite` | flag | off | Re-run cases whose output exists |

**Inputs:** `<case>.b2nd` (2-channel longi), `<case>_fu_clicks.json`, `<case>_bl_clicks.json` (from `nanounet_longi_clicks`).

**Outputs:** `<out>/<case>.nii.gz` in preprocessed spacing.

| Error | Cause | Fix |
|-------|-------|-----|
| `No .b2nd cases in …` | Wrong `-i` path | Point at `.../NanoUNet_preprocessed/<Dataset>/<data_identifier>` |
| `Missing fu_clicks_zyx sidecars` | Clicks not mapped | `nanounet_longi_clicks -d <id> --plans <plans> --clicks-dir … --clicks-fu-dir …` |
| `CUDA requested but … False` | No GPU | Run on a CUDA node |

## Viewer export (`export_d115_viewer_bundle.py`)

After preprocessed inference, build a viewer-ready bundle with the registered-dataset folder layout. Native scans + union-click JSONs are copied; preprocessed preds are warped back to scanner space.

```bash
python3 scripts/export_d115_viewer_bundle.py \
  --model-dir /nnunet_data/NanoUNet_results/nanounet/Dataset114_longi_nnUNetResEncUNetLPlans_h200_smallpv_f0_finetune_dwb \
  --pred-dir /nnunet_data/NanoUNet_preprocessed/Dataset115_longi_test/preds \
  --preprocessed-dir /nnunet_data/NanoUNet_preprocessed/Dataset115_longi_test/nnUNetPlans_3d_fullres \
  --out /nnunet_data/nnUNet_raw/Dataset115_longi_test/last
```

Output: `<out>/{inputsTsFU,inputsTsBL,targetsTsFU,targetsTsBL,predsTsFU}/`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-raw` | str | `Dataset115_longi_test` | Raw nnUNet dataset (imagesTr, labelsTr, clicks) |
| `--pred-dir` | str | `.../Dataset115_longi_test/preds` | Preprocessed-space preds from `nanounet_predict_preprocessed` |
| `--preprocessed-dir` | str | `.../nnUNetPlans_3d_fullres` | Preprocessed folder with `<case>.pkl` props |
| `--model-dir` | str | finetune run dir | Training run (for `plans.json` warp) |
| `--registered-root` | str | registered unigradicon dir | Source of `targetsTrBL` |
| `--out` | str | `<dataset-raw>/viewer_export` | Output bundle root |
| `--overwrite` | flag | off | Re-export existing files |

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing source file` | Incomplete raw or registered data | Verify `--dataset-raw` and `--registered-root` |
| Exit 1 with missing preds list | Inference incomplete | Finish `nanounet_predict_preprocessed`, then re-run |

## Interactive / embed (library)

Not a CLI flag. Radiom remote interactive session calls these in-process:

| Function | Module | Description |
|----------|--------|-------------|
| `predict_patch_logits` | `nanounet.infer.predict_patch` | One centered patch forward; returns `(logits, slices)`. TTA/border-expand off by default. |
| `patch_logits_to_native_seg` | `nanounet.infer.patch_export` | Argmax patch → native scanner-space seg array |
| `native_seg_to_nifti_bytes` | `nanounet.infer.patch_export` | Gzip NIfTI bytes from native seg + `props["sitk_stuff"]` |

See also [radiom_embed_api.md](../dev-notes/radiom_embed_api.md).
