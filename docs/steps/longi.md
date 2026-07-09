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

FU + baseline are preprocessed **jointly as one 2-channel case** at inference (same as training):
one nonzero crop → FU/BL voxel-aligned → clustered mode and `--border-expand` (large-lesion
oversampling) behave exactly like the normal promptable model, and output is the **FU**
segmentation.

```bash
nanounet_predict \
  -i followup.nii.gz -o out/pred.nii.gz \
  -m "$MODEL_DIR" --ckpt finetune/best.ckpt \
  --points fu_points.json \
  --baseline-image baseline.nii.gz --baseline-points bl_clicks.json \
  --inference-mode clustered --merge max --border-expand
```

- `--baseline-image`: sibling BL `.nii.gz`, **already registered into the FU frame** (same
  size/spacing as FU; `nanounet_register_longi` produces this). It is NOT re-preprocessed
  separately.
- `--baseline-points`: a **plain BL click set** (same JSON as `--points`), native voxel `(x,y,z)` in
  the FU-registered frame — one entry per baseline lesion, **not** a partner list parallel to
  `--points`.
- Without both flags a longi checkpoint runs **null-baseline** (duplicated FU stream → identity DWB
  → single-timepoint behaviour).
- Dataset mode: use `--baseline-dir <dir>` with `<cid>.nii.gz` + `<cid>.json` per FU case.
- Net auto-detects longi from `net.dwb.*` keys; pass `--longi` to force.

---

## `nanounet_register_longi`

Warp BL→FU frame (itk-elastix or uniGradICON) for one or many longitudinal pairs.

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
| `--backend` | str | `elastix` | Registration backend: `elastix` (classical) or `unigradicon` (deep learning + native IO) |
| `--io-iterations` | int | `50` | uniGradICON native instance-optimization steps (`0` disables); ignored for `elastix` |

**Outputs:** `inputsTrFU/`, `inputsTrBL/`, `targetsTrFU/`, warped BL clicks JSON under `inputsTrBL/`.
Same layout regardless of backend — output is byte-compatible with the classical pipeline.

**Common errors:** must pass `--pid/--idx`, `--sample N`, or `--all`; skipped cases exit code 1.

### Backends

- `elastix` (default): classical rigid→affine→bspline multi-resolution registration; per-lesion
  `refine` (VOI-local re-registration) runs afterwards unless `--no-refine`.
- `unigradicon`: pretrained foundation model. Its native instance optimization (`--io-iterations`) is
  the refinement step — per-lesion `refine`/`--no-refine` does not apply to this backend. GPU is
  strongly preferred: IO is `--io-iterations` backward passes through a 3D UNet at 175³ per case and is
  slow on CPU. Weights are cached at `$NANOUNET_UNIGRADICON_WEIGHTS` or
  `~/.cache/nanounet/unigradicon/` on first use.

```bash
nanounet_register_longi --data-root /path/to/raw --out /path/to/regout --all --backend unigradicon --io-iterations 50
```

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

**Outputs:** `_0000` = FU CT, `_0001` = warped BL CT (FU frame), FU seg, `clicksTr/<case>.json` (BL
union clicks), `clicksTrFU/<case>.json` (FU union clicks). Both carry the full BL+FU lesion-id
union, including "disappeared" lesions with no FU ground truth.

**Next step:** `nanounet_preprocess -d <id-for-out>`.

**Common errors:** size mismatch between FU and warped BL (bad registration); empty `inputsTrFU/`;
missing `clicksBL/`/`clicksFU/<case>.json` under `--register-out`.

---

## `nanounet_longi_clicks`

Map warped-BL and FU union clicks (register xyz, FU frame) into preprocessed voxels.

```bash
nanounet_longi_clicks -d NNN --plans nnUNetResEncUNetLPlans \
  --clicks-dir "$NANOUNET_RAW/DatasetNNN_longi/clicksTr" \
  --clicks-fu-dir "$NANOUNET_RAW/DatasetNNN_longi/clicksTrFU"
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int | (required) | Preprocessed dataset id |
| `--plans` | str | (required) | Plans basename (no `.json`) |
| `--clicks-dir` | str | (required) | Raw `clicksTr` with warped BL click JSON per case |
| `--clicks-fu-dir` | str | `None` | Raw `clicksTrFU` with FU union click JSON per case |
| `--cog-axis-order` | choice | `xyz` | `xyz` \| `zyx` — must match click file axis order |

**Outputs:** `<case>_bl_clicks.json` next to each preprocessed FU case (`bl_clicks_zyx`, `has_baseline`).
If `--clicks-fu-dir` given, also `<case>_fu_clicks.json` (`fu_clicks_zyx`, `fu_topology`) — the
FU-stream prompt source for `build_patch_longi`, including disappeared-lesion points.

**Next step:** `nanounet_train -d NNN --plans … --longi --init-weights <stage2.ckpt>`.

**Notes:** points mapping outside the preprocessed volume are dropped (counted in the summary line),
not asserted — a DISAPPEARING lesion's BL location can legitimately sit in FU-image territory the
nonzero crop excluded. A wrong `--cog-axis-order` shows up as most/all points dropped.

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
