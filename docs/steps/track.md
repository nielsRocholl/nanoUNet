# Track (seg × track)

One command: two CTs + click JSON → instance masks with shared tracking ids + match CSV. Predicts both timepoints (Dataset999 single-stream), instance-izes click-on-FG, then matches. Requires `pip install -e /lesion-tracking`.

Default `--decode hungarian`. FU click JSON is the matcher’s BL coordinates (no meta CSV). See [track_ids.md](../reference/track_ids.md).

## Command

Folder (sibling `{stem}.nii.gz` + `{stem}.json`, pair by exact stem):

```bash
nanounet_segtrack \
  --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL \
  --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU \
  --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv
```

Single case:

```bash
nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.nii.gz \
  --bl-clicks /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.json \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.json
```

Writes `$NANOUNET_RESULTS/segtrack/inputsTrFU/{stem}/` (folder) or `$NANOUNET_RESULTS/segtrack/single/{stem}/` (one case). Override with `-o`.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-dir` `--fu-dir` | path | — | Folder mode. Sibling `.nii.gz` + `.json` |
| `--bl-img` `--bl-clicks` `--fu-img` `--fu-clicks` | path | — | Single mode |
| `--meta` | path | unset | Single: optional types CSV (`lesion_id,lesion_type`). Not coordinates. |
| `--meta-dir` | path | unset | Folder: optional types CSVs `{pid}.csv`. Not coordinates. |
| `-o, --out` | path | `$NANOUNET_RESULTS/segtrack/...` | Parent (folder) or case dir (single) |
| `-m, --model-dir` | path | Dataset999 `h200_instance_1200ep` | Seg run dir (`plans.json` + ckpt) |
| `--ckpt` | str | `last.ckpt` | Seg checkpoint name |
| `--track-ckpt` | path | `h60_r9/best.ckpt` | Matcher Lightning ckpt |
| `--decode` | choice | `hungarian` | `hungarian` / `dense` / `sinkhorn` |
| `--thresh` | float | `0.5` | Dense pair cutoff |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` |
| `--patients-csv` | path | unset | Folder filter |
| `--overwrite` | flag | off | Redo cases that already have `matches.csv` |
| `--keep-pred` | flag | off | Keep binary FG next to masks |
| `--ema` | flag | off | Seg EMA weights |
| `--batch-size` | int | `8` | Predict batch |
| `--inference-mode` | choice | `clustered` | `clustered` \| `centered` |
| `--disable-tta` | flag | config default | Same as predict |
| `--no-amp` | flag | off | |

Env overrides: `NANOUNET_SEGTRACK_MODEL`, `NANOUNET_SEGTRACK_TRACK`.

## Inputs / outputs

**In:** BL/FU CT NIfTIs + click JSON (`points[].name` = lesion_id, `point` = `[x,y,z]` in that scan’s frame; FU JSON must be follow-up space). Optional types CSV.

**Out** (each case dir):

1. `{out}/bl.nii.gz` — BL instance mask, ids unchanged (`int32`).
2. `{out}/fu.nii.gz` — FU instance mask, ids remapped so the same integer = same lesion.
3. `{out}/matches.csv` — `bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id`.

`track_id` is the voxel value on `fu.nii.gz` for that pair. Unmatched lesions are not extra CSV rows; they only appear as ids on one mask.

## Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tracking is not installed` | lesion-tracking not on PYTHONPATH | `pip install -e /lesion-tracking` |
| `No seg model at …` | Missing run dir / env | `-m $NANOUNET_RESULTS/nanounet/<run>` or `export NANOUNET_SEGTRACK_MODEL=...` |
| `BL/FU folders do not share the same case names` | Stem mismatch (`_00` vs `_01`) | Matching `inputsTrBL` / `inputsTrFU`, or `--patients-csv` |
| missing points JSON | No sibling `{stem}.json` | Add the click JSON next to each scan |
| `No FU-frame point for BL mask ids` | FU JSON lacks that BL id | Pass FU JSON in follow-up space, not baseline JSON |
| Empty instance mask | No click hit predicted FG | Check clicks; see [track_ids.md](../reference/track_ids.md) |
