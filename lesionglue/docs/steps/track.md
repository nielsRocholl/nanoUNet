# Track

Deployment inference: baseline + follow-up CT and lesion instance masks in, a BL to FU match CSV out. Runs one case or a
whole Longitudinal-CT split. Needs `DEPLOYED_CKPT` (`/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt`, EMA,
hungarian, `dust_tau=0.125`); geo checkpoints also need propagated BL centroids in the FU voxel grid.

## Command

Single case (`--propagated` is required because the deployed ckpt has `drop_dp=false`):

```bash
lesionglue_track \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/16a5cdae36_00.nii.gz \
  --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/16a5cdae36_00.nii.gz \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/16a5cdae36_00.nii.gz \
  --fu-mask /nnunet_data/Longitudinal-CT/targetsTrFU/16a5cdae36_00.nii.gz \
  --propagated /nnunet_data/Longitudinal-CT/meta/16a5cdae36.csv \
  --out runs/track_case/16a5cdae36.csv --pairs-out runs/track_case/16a5cdae36_pairs.csv
```

Dataset mode (`--root` plus exactly one of `--split` / `--patients-csv`):

```bash
lesionglue_track --root /nnunet_data/Longitudinal-CT --split test --out runs/track_test
```

```bash
lesionglue_track --root /nnunet_data/Longitudinal-CT --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv --decode sinkhorn --out runs/track_test_sinkhorn
```

Decode modes (`--decode`):

| Mode | Rule | Merges / splits |
|---|---|---|
| `hungarian` (default) | Sinkhorn plan, then strict 1-to-1 assignment above `--sinkhorn-tau` | dropped |
| `sinkhorn` | Keep every pair whose row-normalised Sinkhorn mass clears `--sinkhorn-tau` | kept, up to about `1/--sinkhorn-tau` lesions per merge |
| `dense` | Keep every pair with probability above `--thresh` | kept |

`hungarian` is the deployed decode and the holdout gate (test match **0.9701**, 57 graphs). Details: `lesionglue/model/decode.py`.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-img` | path | `""` | Single case: baseline CT NIfTI |
| `--bl-mask` | path | `""` | Single case: baseline lesion instance mask NIfTI (binary FG with `--bl-clicks`) |
| `--fu-img` | path | `""` | Single case: follow-up CT NIfTI |
| `--fu-mask` | path | `""` | Single case: follow-up lesion instance mask NIfTI (binary FG with `--fu-clicks`) |
| `--propagated` | path | `""` | BL lesion_id to FU-frame centroid: meta CSV, slim CSV (`lesion_id,z,y,x`) or nanoUNet JSON in the FU frame. Required unless the ckpt was trained with `drop_dp` |
| `--ckpt` | path | `DEPLOYED_CKPT` | Matcher Lightning checkpoint |
| `--out` | path | required | Single case: match CSV to write; dataset mode: dir for one `<pid>.csv` per patient |
| `--decode` | choice | `hungarian` | `dense`, `sinkhorn` or `hungarian` (table above) |
| `--thresh` | float | 0.5 | Pair probability cutoff for `--decode dense`; unused by sinkhorn and hungarian |
| `--device` | choice | `cuda` | `cuda`, `cpu` or `mps` |
| `--k-intra` | int | 8 | kNN neighbours per lesion in the intra-scan graph; must equal the checkpoint value |
| `--sinkhorn-iters` | int | 20 | Sinkhorn iterations for `--decode sinkhorn` and `hungarian` |
| `--sinkhorn-tau` | float | `0.125` (`DEPLOYED_DUST_TAU`) | Min row-normalised Sinkhorn mass to keep a pair (sinkhorn, hungarian) |
| `--default-lesion-type` | str | `unclear` | Lesion type for lesions absent from `--types-csv` (single case); `unclear` = no real type |
| `--types-csv` | path | `""` | `lesion_id,lesion_type` CSV; required for `type_mask` ckpts unless `--default-lesion-type` is not `unclear` |
| `--no-ema` | flag | off | Run the raw weights instead of EMA |
| `--pairs-out` | path | `""` | Single case: also write every BL x FU pair probability to this CSV; empty skips |
| `--bl-clicks` | path | `""` | Instance JSON; treat `--bl-mask` as binary FG |
| `--fu-clicks` | path | `""` | Instance JSON; treat `--fu-mask` as binary FG |
| `--root` | path | `""` | Longitudinal-CT root; setting it selects dataset mode |
| `--split` | choice | none | Dataset mode: `train`, `val` or `test` patients from `lesionglue/configs/split.json` (or use `--patients-csv`) |
| `--patients-csv` | path | `""` | Dataset mode: CSV with a `patient` column (or use `--split`) |
| `--bl-mask-dir` | path | `""` | Dataset mode: dir of `<pid>_<idx>.nii.gz` baseline masks; empty = `<root>/targetsTrBL` |
| `--fu-mask-dir` | path | `""` | Dataset mode: dir of `<pid>_<idx>.nii.gz` follow-up masks; empty = `<root>/targetsTrFU` |
| `--prop-dir` | path | `""` | Dataset mode: dir of `<pid>.csv` or `<pid>_<idx>.json` propagated coords; empty = `<root>/meta` |

## Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/Longitudinal-CT/{inputsTrBL,inputsTrFU}/<pid>_<idx>.nii.gz` | NIfTI CT | you |
| `/nnunet_data/Longitudinal-CT/{targetsTrBL,targetsTrFU}/<pid>_<idx>.nii.gz` | NIfTI instance mask | you |
| `/nnunet_data/Longitudinal-CT/meta/<pid>.csv` | meta CSV (propagated centroids, types) | you |
| `runs/track_test/<pid>.csv` (dataset) or `--out` (single) | CSV: `bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id` | this step |
| `--pairs-out` | CSV: `bl_lesion_id,fu_lesion_id,prob` (every BL x FU pair) | this step |

Dataset mode skips a patient (prints `skip <pid>`) when a file is missing or either mask has no lesions, then prints an
ok / skip / pairs table. The last line prints a `next:` command.

## Common errors

| Message starts with | Fix |
|---|---|
| `No checkpoint at` | `--ckpt /nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` |
| `Need either a single case` | Give all of `--bl-img --bl-mask --fu-img --fu-mask --propagated`, or only `--root /nnunet_data/Longitudinal-CT --split test` |
| `--split / --patients-csv require --root.` | Add `--root /nnunet_data/Longitudinal-CT` |
| `Dataset mode needs exactly one of` | Pass `--split test` or `--patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv`, not both |
| `--k-intra` | Omit `--k-intra`, or pass the checkpoint value named in the message |
| `type_mask checkpoint needs lesion types` | Pass `--types-csv /nnunet_data/Longitudinal-CT/meta/16a5cdae36.csv` (single case) or a real `--default-lesion-type` |
| `drop_dp checkpoint does not use propagated` | Omit `--propagated` |
| `No bl-img at` | Also `No bl-mask at`, `No fu-img at`, `No fu-mask at`, `No propagated at`: pass an existing path |
| `No clicks JSON at` | `--bl-clicks` / `--fu-clicks` must point at the sibling nanoUNet click JSON |
| `--device cuda but CUDA not available.` | `--device cpu` |
| `No tracking split at` | `lesionglue_split --root /nnunet_data/Longitudinal-CT --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out lesionglue/configs/split.json` |
| `No holdout CSV at` | Fix the `--patients-csv` path |
