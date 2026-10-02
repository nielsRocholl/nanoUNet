# Lesion weights

Offline build of per-centroid hard-type sampling weights for d013 oversampling: forward-maps
meta-CSV lesion cogs into preprocessed voxel space, nearest-matches each preprocessed centroid to a
lesion type, and writes `<case>_weights.json` next to the case.

## Command

```bash
nanounet_lesion_weights -d 13 --plans nnUNetResEncUNetLPlans --meta-dir /path/to/lesion_csvs
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int | required | Dataset id, e.g. 13 |
| `--plans` | str | required | Plans identifier, no `.json`; rerun per plans (weights live in that plans' data folder and follow its voxel grid), e.g. `nnFoundationCNN_z1p0` |
| `--meta-dir` | str | required | Folder of `<hash>.csv` lesion-type files |
| `--only-prefix` | str | `d013_Longitudinal_CT_` | Case id prefix up to the per-patient hash; used both to filter case ids and to parse hash/timepoint from each id |
| `--cog-axis-order` | choice | `xyz` | Axis order of the `cog_bl`/`cog_fu` columns in the meta CSV: `xyz` or `zyx` |
| `--max-match-dist` | float | `10.0` | Max voxel distance for a centroid-to-lesion match |
| `--max-median-dist` | float | `8.0` | Sanity gate: the overall median match distance must be <= this |

## Inputs / outputs

**Inputs**

- `$NANOUNET_PREPROCESSED/DatasetXXX_*/<plans>.json` — plans, for `transpose_forward` and the
  `3d_fullres` configuration's `data_identifier`
- `$NANOUNET_PREPROCESSED/DatasetXXX_*/<data_identifier>/<case>.pkl` — per-case properties
  (`bbox_used_for_cropping`, `shape_after_cropping_and_before_resampling`, `centroids_zyx`)
- `<meta-dir>/<hash>.csv` — per-patient lesion CSV with `cog_bl`, `cog_fu`, `lesion_type` columns

**Outputs**

- `<data_identifier>/<case>_weights.json` — `{"centroid_weights": [...]}`, one float per centroid,
  in `centroids_zyx` order

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `no cases with prefix ... in ...` | `--only-prefix` matches zero preprocessed case ids | Check `--only-prefix` against the dataset's actual case ids |
| Assertion fails on a bare CSV path | `<hash>.csv` missing under `--meta-dir` for a case matched by `--only-prefix` | Add the missing CSV, or narrow `--only-prefix` to exclude that case |
| `cog->preprocessed mapping looks wrong; check --cog-axis-order` | Overall median match distance exceeds `--max-median-dist` | Re-run with the other `--cog-axis-order` value |
