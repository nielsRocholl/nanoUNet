# Preprocess

Fingerprint the raw dataset, write the plan, and resample cases to blosc2 (`3d_fullres`).
**Default (foundation mode):** nnFoundationCNN topology and 192³ patch, per-image Z-score, and z-only
resampling (each case's thickest axis to `--target-z` mm, the other axes keep native spacing and grid).
`--no-foundation` restores the ResEnc planner at the dataset median spacing. Supports single ids or merging.

## Command

```bash
export NANOUNET_PRETRAINED=/nnunet_data/NanoUNet_pretrained
nanounet_preprocess -d 900 --target-z 1.0 --gpu-memory-gb 141 -np 16
```

Old behaviour (planner, median spacing, CTNormalization):

```bash
nanounet_preprocess -d 001 --no-foundation --planner nnUNetPlannerResEncL -np 8
```

Merge example:

```bash
nanounet_preprocess -d 1 2 3 --merged-id 999 --merged-name Merged -np 8
```

Merge with a custom split fraction and a fixed validation manifest built in the same step:

```bash
nanounet_preprocess -d 11 12 13 --merged-id 900 --merged-name Merged -np 16 \
  --val-frac 0.15 --split-seed 12345 --valset-config nanounet/configs/default.json --valset-n 1500
```

Tiny local model:

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncTiny --patch-vol small -np 4
```

Regenerate centroid sidecars only (already-preprocessed dataset, no `.b2nd`/plans rewrite):

```bash
nanounet_preprocess -d 999 --plans-name nnUNetResEncUNetLPlans --sidecars-only -np 8
```

Regenerating sidecars overwrites files that existing checkpoints depend on. Back up the current
`*_centroids.json` before regenerating.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int+ | (required) | One or more dataset ids, e.g. `-d 001` or `-d 1 2 3` for merge |
| `--merged-id` | int | `999` | Output dataset id when merging multiple `-d` |
| `--merged-name` | str | `Merged` | Name segment for merged folder `DatasetNNN_<name>` |
| `--planner` | str | `nnUNetPlannerResEncL` | Planner class (e.g. `nnUNetPlannerResEncTiny`); only with `--no-foundation` |
| `--no-foundation` | flag | off | Use the ResEnc planner at the dataset median spacing instead of nnFoundationCNN + z-only resampling |
| `--target-z` | float | `1.0` | Foundation mode: target spacing (mm) of each case's thickest axis (plans name `nnFoundationCNN_z1p0`) |
| `-np`, `--num_processes` | int | `8` | Parallel workers for fingerprint / preprocess |
| `--resume` | flag | off | Keep existing preprocess output; do not wipe `3d_fullres` folder |
| `--gpu-memory-gb` | float | none | VRAM budget (GB) the batch size is sized for (foundation: even batch, 141 → 12); planner patch shrink with `--no-foundation` |
| `--patch-vol` | choice | `large` | `small` (128) \| `medium` (192) \| `large` (256) \| `xlarge` (320) isotropic edge before aniso handling; only with `--no-foundation` |
| `--plans-name` | str | none | Basename of plans JSON (no `.json`); required with `--skip-plan` or `--sidecars-only` |
| `--skip-fingerprint` | flag | off | Skip fingerprint; use existing `dataset_fingerprint.json` |
| `--skip-plan` | flag | off | Skip planning; requires `--plans-name` |
| `--sidecars-only` | flag | off | Regenerate `*_centroids.json` sidecars only; requires `--plans-name`; never touches `.b2nd`, plans, or `gt_segmentations` |
| `--val-frac` | float | `0.15` | Held-out fraction for `splits_final.json`, balanced within each source dataset (see [`nanounet_build_splits`](../../cli/build_splits.py)) |
| `--split-seed` | int | `12345` | RNG seed for the balanced train/val split |
| `--no-splits` | flag | off | Skip the `splits_final.json` / `cohorts.json` step (including its test-patient check) |
| `--valset-config` | str | `None` | ROI config path; when set, also builds the fixed validation manifest via `nanounet_build_valset` |
| `--valset-n` | int | `1500` | Patch count for the fixed validation manifest, only used with `--valset-config` |

See [plan.md](plan.md) for the foundation plan and the `--no-foundation` trade-offs.

## Inputs / outputs

**Inputs**

- `$NANOUNET_RAW/DatasetXXX_*/` — nnUNet raw layout (`imagesTr`, `labelsTr`, `dataset.json`)
- Environment: `NANOUNET_RAW`, `NANOUNET_PREPROCESSED`, and for foundation mode `NANOUNET_PRETRAINED`
- `/nnunet_data/Longitudinal-CT/test_patients.csv` — the 60 hold-out patients, which must never appear in train or val (checked at split time)

**Outputs** (under `$NANOUNET_PREPROCESSED/DatasetXXX_*/`)

- `dataset_fingerprint.json`
- `<plans>.json` — patch size, batch size, network topology; foundation plans add `spacing_mode: z_only` and `pretrain_info` (checkpoint path + sha256)
- `$NANOUNET_PRETRAINED/` — nnFoundationCNN checkpoint (CC-BY-SA-4.0), downloaded at plan time
- `<plans>/3d_fullres/*.b2nd` — blosc2 training tensors
- `<plans>/3d_fullres/*_centroids.json` — per-lesion centroid, bbox, EDT seed, and voxel count
- `gt_segmentations/` — resampled labels (also under raw dataset folder)
- `splits_final.json` — single balanced train/val split (skip with `--no-splits`). **Never rewritten:** if it exists it is verified against the data folder and kept (`✓ splits kept (sha256 …)`); only created when absent. Same for `cohorts.json`
- `cohorts.json` — per-cohort sampling weights derived from `lesion_site` (see below); consumed by
  `nanounet_train`'s default cohort-weighted sampler
- `valset_<n>_<plans>.json` + `.targets.npz` — fixed validation manifest, only with `--valset-config`

### lesion_site and cohorts.json

Each source `dataset.json` under `$NANOUNET_RAW` carries a `"lesion_site"` key (e.g. `liver`,
`lung`, `pancreas`) — a lowercase controlled-vocabulary tag for the anatomical site the dataset's
lesions come from. Datasets covering the same organ share the exact same string. Preprocess reads
this key from every merged source and writes `cohorts.json` with one weight per site instead of
per source dataset, so a site spread across many small datasets (or one dataset dominating a site
by raw case count) doesn't silently dominate the training mixture. Every source `dataset.json`
passed to a merge must carry `lesion_site` — add the key (see the 21 datasets under
`$NANOUNET_RAW/Dataset0{11..31}_*` for examples) before preprocessing a new merge.

### seed_zyx and volume_vox

The plain lesion centroid is a bad click prompt: for irregular/concave lesions it falls outside the
lesion's own mask about 12% of the time, which breaks "which component does this click belong to."
Each `*_centroids.json` therefore also carries, per lesion, `seed_zyx` — the argmax-EDT voxel, i.e.
the point deepest inside the component, guaranteed to carry that component's label — and
`volume_vox`, the component's voxel count. Training converts `volume_vox` to an equivalent-sphere
diameter to pick the lesion's size bin in the registration-error offset table, then perturbs the
stored centroid by an offset drawn from that table to simulate how a real point click drifts between
baseline and follow-up scans.

The offset table itself (`propagated.error_table` in the ROI config) is a JSON file matching the
schema in [reference/config.md](../reference/config.md), produced once against Longitudinal-CT
derivatives and not part of the per-dataset preprocess loop above — regenerate only if those
derivatives change. `propagated.mode: "gaussian"` needs no table.

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `--skip-plan needs --plans-name` | `--skip-plan` without plans basename | Pass `--plans-name nnUNetResEncUNetLPlans` (or your existing plans file) |
| Missing raw dataset folder | Wrong id or `NANOUNET_RAW` | Check `DatasetXXX_*` exists under raw root |
| Planner OOM / tiny patch | `--gpu-memory-gb` too low for `--patch-vol` | Lower `--patch-vol` or raise `--gpu-memory-gb` to match training GPU |
| Wiped preprocess mid-run | Re-run without `--resume` | Use `--resume` to keep existing `3d_fullres` output |
| `--sidecars-only needs --plans-name` | `--sidecars-only` without plans basename | Pass `--plans-name` for the dataset's existing plans json |
| `--sidecars-only needs an existing preprocessed folder` | Target `<data_identifier>` folder missing | Run a full preprocess first (command shown in the error) |
| `NANOUNET_PRETRAINED is not set` | Foundation mode needs a weights cache dir | `export NANOUNET_PRETRAINED=/nnunet_data/NanoUNet_pretrained` |
| `No dataset fingerprint at` | Foundation plan reads spacings/shapes from the fingerprint | Run without `--skip-fingerprint` once |
| `data_identifier '...' is already used by` | Plans name collides with another plans' data folder | Pass a new `--plans-name` |
| `... does not match the dataset` | Existing `splits_final.json` disagrees with the data folder | Preprocess the missing cases (`--resume`); the file is never rewritten for you |
| `... belong to Longitudinal-CT test patients` | A hold-out patient is in the dataset | Remove those cases from the raw `dataset.json` and rebuild the splits |
| `A preprocess worker was killed with no Python exception (SIGKILL)` | OOM kill by the cgroup, not a code bug — large volumes can peak ~50 GB/worker during resampling | Lower `-np` (the error suggests half the current count) and rerun with `--resume` to skip already-finished cases |
