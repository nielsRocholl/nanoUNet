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
| `--baseline-points` | str | none | BL partner points JSON, parallel to `--points` |
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
| `--baseline-points requires --baseline-image` | Longi flags mismatched | Pass both or neither |
| Missing checkpoint | Wrong `--ckpt` or incomplete train | Verify path under `checkpoints/` or `finetune/` |
| CUDA unavailable | No GPU | Use `--device cpu` or `mps` |

Longitudinal two-stream inference: [longi.md](longi.md).
