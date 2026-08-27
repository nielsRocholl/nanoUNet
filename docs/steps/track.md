# Track (seg × track)

One command: two CTs + click JSON → instance masks with shared tracking ids + match CSV. Predicts both timepoints (Dataset999 single-stream), instance-izes click-on-FG, then matches. `--bl-mask` / `--bl-mask-dir` skips BL predict and copies those instance ids. Requires `pip install -e /lesion-tracking`.

Default matcher: `v7_complete/last.ckpt`, matcher EMA on, `--decode hungarian`, `dust_tau=0.125`. `--ema` is the **seg** UNet only. Pair by exact stem (`pid_00` ≠ `pid_01`). Matcher BL positions come from `{dataset}/meta/{pid}.csv` (`cog_propagated`, else `cog_fu`, rows with `img_id_fu` = this stem’s region) when that folder exists. A `drop_dp` matcher checkpoint skips that warp and uses native mask centroids. See [track_ids.md](../reference/track_ids.md).

## Command

Folder (sibling `{stem}.nii.gz` + `{stem}.json`, pair by exact stem):

```bash
nanounet_segtrack \
  --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL \
  --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU
```

Single case:

```bash
nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.nii.gz \
  --bl-clicks /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.json \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.json
```

GT baseline mask (skip BL UNet; BL clicks omitted). Folder:

```bash
nanounet_segtrack \
  --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL \
  --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU \
  --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL
```

Single:

```bash
nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.nii.gz \
  --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/01161aaa0b_00.nii.gz \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.json
```

Writes `$NANOUNET_RESULTS/segtrack/inputsTrFU/{stem}/` (folder) or `$NANOUNET_RESULTS/segtrack/single/{stem}/` (one case). Override with `-o`.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-dir` `--fu-dir` | path | — | Folder mode. Sibling `.nii.gz` + `.json` (BL JSON not required with `--bl-mask-dir`) |
| `--bl-img` `--bl-clicks` `--fu-img` `--fu-clicks` | path | — | Single mode |
| `--bl-mask` | path | unset | Single: native BL instance mask (voxel = lesion_id). Skips BL predict. No `--bl-clicks` |
| `--bl-mask-dir` | path | unset | Folder: `{stem}.nii.gz` or `{stem}.mha`. Skips BL predict |
| `--meta` | path | unset | Single: lesion CSV (`cog_propagated` / `cog_fu` / `img_id_fu` / `lesion_type`) |
| `--meta-dir` | path | `{dataset}/meta` if present | Folder: `{pid}.csv`. Inferred from `--bl-dir` parent. Coords + types |
| `-o, --out` | path | `$NANOUNET_RESULTS/segtrack/...` | Parent (folder) or case dir (single) |
| `-m, --model-dir` | path | Dataset999 `h200_instance_1200ep` | Seg run dir (`plans.json` + ckpt) |
| `--ckpt` | str | `last.ckpt` | Seg checkpoint name |
| `--track-ckpt` | path | `v7_complete/last.ckpt` | Matcher (EMA, hungarian, `dust_tau=0.125`). `drop_dp` ckpts skip `cog_propagated` |
| `--decode` | choice | `hungarian` | `hungarian` / `dense` / `sinkhorn` |
| `--thresh` | float | `0.5` | Dense pair cutoff only |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` |
| `--patients-csv` | path | unset | Optional holdout filter on stem prefix. Unpaired / missing JSON are skipped. |
| `--overwrite` | flag | off | Redo cases that already have `matches.csv` |
| `--keep-pred` | flag | off | Binary FG `pred_bl.mha` / `pred_fu.mha`. Mask mode: `pred_fu.mha` only |
| `--ema` | flag | off | Seg UNet EMA. Matcher EMA is always on |
| `--batch-size` | int | `8` | Predict batch |
| `--inference-mode` | choice | `clustered` | `clustered` \| `centered` |
| `--disable-tta` | flag | config default | Same as predict |
| `--no-amp` | flag | off | |

Env overrides: `NANOUNET_SEGTRACK_MODEL`, `NANOUNET_SEGTRACK_TRACK`.

## Inputs / outputs

**In:** BL/FU CT NIfTIs + click JSON (`points[].name` = lesion_id, `point` = `[x,y,z]` in that scan’s frame). Sibling JSON is seg clicks for that volume. Matcher BL positions: inferred `{dataset}/meta/{pid}.csv` rows with `img_id_fu` matching the stem region (`_00` → 0), unless the matcher ckpt was trained with `drop_dp` (native mask centroids; FU JSON stays the UNet click file). Optional BL instance mask: native BL grid, voxel = lesion_id; BL clicks omitted.

**Out** (each case dir):

1. `{out}/bl.mha` — BL instance mask, ids unchanged (`uint8` or `int16`).
2. `{out}/fu.mha` — FU instance mask, ids remapped so the same integer = same lesion.
3. `{out}/matches.csv` — `bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id`.

`track_id` is the voxel value on `fu.mha` for that pair. Unmatched lesions are not extra CSV rows; they only appear as ids on one mask.

## Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tracking is not installed` | lesion-tracking not on PYTHONPATH | `pip install -e /lesion-tracking` |
| `No seg model at …` | Missing run dir / env | `-m $NANOUNET_RESULTS/nanounet/<run>` or `export NANOUNET_SEGTRACK_MODEL=...` |
| `BL/FU folders do not share any case names` | empty intersection | matching `inputsTrBL` / `inputsTrFU` |
| `skip {stem} (no BL json)` / `no FU json` | incomplete case | skipped; rest of folder runs |
| `skip {stem} (no FU scan)` / `no BL scan` | stem only in one folder | skipped; pair by exact stem `pid_00` ≠ `pid_01` |
| `skip {stem} (no meta csv)` | `{pid}.csv` missing under meta-dir | skipped |
| `drop {stem} BL ids … (not in this FU volume)` | id is other region / no coord | omitted from matcher; stays on `bl.mha` |
| `--bl-clicks was set with --bl-mask` | both given | drop `--bl-clicks` |
| `skip {stem} (no BL mask)` | missing mask for that stem | skipped |
| `skip {stem} (BL mask grid != BL CT grid)` | wrong space / registered mask | native `targetsTrBL`, not FU-warped |
| Empty instance mask | No click hit predicted FG | Check clicks; see [track_ids.md](../reference/track_ids.md) |
