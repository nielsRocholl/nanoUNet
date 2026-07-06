# Longitudinal workflow

Register baseline→followup, build a 2-channel raw dataset, preprocess, map BL clicks, finetune with `--longi`, and run two-stream inference with baseline image and partner points.

## Pipeline

```mermaid
flowchart LR
    A[register_longi] --> B[longi_build]
    B --> C[nanounet_preprocess]
    C --> D[longi_clicks]
    D --> E[train --longi]
    E --> F[predict + BL flags]
```

## Two-stream inference

For `--longi` finetune checkpoints, pass the baseline scan and pre-propagation BL centroids so the model receives a co-located BL stream (identity fallback when a partner is absent).

```bash
nanounet_predict \
  -i followup.nii.gz \
  -o out/pred \
  -m "$MODEL_DIR" \
  --ckpt finetune/best-epoch=412-val_dice_macro=0.6649.ckpt \
  --points fu_points.json \
  --baseline-image baseline.nii.gz \
  --baseline-points bl_partners.json \
  --inference-mode clustered --merge max --border-expand
```

- `--baseline-image`: sibling BL `.nii.gz`, preprocessed with the same path as FU.
- `--baseline-points`: same `points` list format as `--points`; one BL partner per FU prompt (`null` for new lesions). Length must match FU points list.
- Without both flags, inference stays single-stream. Net auto-detects longi from `net.dwb.*` keys; pass `--longi` to force.

---

## `nanounet_register_longi`

Warp BL→FU frame (itk-elastix) for one or many longitudinal pairs.

```bash
nanounet_register_longi --data-root /path/to/raw --out /path/to/regout --all
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-root` | str | (required) | Root containing longitudinal source data |
| `--out` | str | (required) | Output directory for registration artifacts |
| `--pid` | str | none | Single case: patient id (with `--idx`) |
| `--idx` | str | none | Single case: timepoint index (with `--pid`) |
| `--sample` | int | none | Process N random BL/FU pairs |
| `--all` | flag | off | Process every BL/FU pair |
| `--seed` | int | `0` | RNG seed for `--sample` |
| `--qc` | flag | off | Write axial QC montage PNG |
| `--no-body-mask` | flag | off | Disable body-masked metric |
| `--no-refine` | flag | off | Disable per-lesion VOI refinement |
| `--threads` | str | `auto` | ITK threads: `auto`, `all`, integer, or `NANOUNET_REG_THREADS` |
| `--verbose` | flag | off | Show elastix/transformix console log |

**Outputs:** `inputsTrFU/`, `inputsTrBL/`, `targetsTrFU/`, warped BL clicks JSON under `inputsTrBL/`.

**Common errors:** must pass `--pid/--idx`, `--sample N`, or `--all`; skipped cases exit code 1.

---

## `nanounet_longi_build`

Build 2-channel raw dataset from registration output.

```bash
nanounet_longi_build \
  --register-out /path/to/regout \
  --template-dj /path/to/Dataset013/dataset.json \
  --out "$NANOUNET_RAW/DatasetNNN_longi"
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--register-out` | str | (required) | `register_longi` output directory |
| `--template-dj` | str | (required) | `dataset.json` to copy labels / `file_ending` from |
| `--out` | str | (required) | Raw dataset dir to create, e.g. `.../DatasetNNN_longi` |

**Outputs:** `_0000` = FU CT, `_0001` = warped BL CT (FU frame), FU seg, `clicksTr/<case>.json`.

**Next step:** `nanounet_preprocess -d <id-for-out>`.

**Common errors:** size mismatch between FU and warped BL (bad registration); empty `inputsTrFU/`.

---

## `nanounet_longi_clicks`

Map warped BL clicks (register xyz, FU frame) into preprocessed voxels.

```bash
nanounet_longi_clicks -d NNN --plans nnUNetResEncUNetLPlans \
  --clicks-dir "$NANOUNET_RAW/DatasetNNN_longi/clicksTr"
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int | (required) | Preprocessed dataset id |
| `--plans` | str | (required) | Plans basename (no `.json`) |
| `--clicks-dir` | str | (required) | Raw `clicksTr` with warped BL click JSON per case |
| `--cog-axis-order` | choice | `xyz` | `xyz` \| `zyx` — must match click file axis order |

**Outputs:** `<case>_bl_clicks.json` next to each preprocessed FU case (`bl_clicks_zyx`, `has_baseline`).

**Next step:** `nanounet_train -d NNN --plans … --longi --init-weights <stage2.ckpt>`.

**Common errors:** mapped click outside preprocessed volume (wrong `--cog-axis-order`).

---

## `nanounet_repair_longi_fu`

Repair FU/meta sidecars for cases whose BL registration already succeeded.

```bash
nanounet_repair_longi_fu --data-root /path/to/raw --out /path/to/regout
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-root` | str | (required) | Original data root |
| `--out` | str | (required) | Registration output directory |

**Outputs:** Repaired FU inputs / metadata for incomplete cases; prints status summary table.

**Common errors:** cases still incomplete after repair (logged in red).

---

## Training finetune

Stage 3 longitudinal finetune requires warm-start from stage-2 supervised weights:

```bash
nanounet_train -d NNN -f 0 --plans nnUNetResEncUNetLPlans --config configs/default.json \
  --longi --init-weights /path/to/stage2/checkpoints/best.ckpt
```

See [train.md](train.md) for full supervised arguments.
