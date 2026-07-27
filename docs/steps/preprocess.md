# Preprocess

Fingerprint the raw dataset, run the ResEnc planner, and resample cases to blosc2 (`3d_fullres`).
Supports single dataset ids or merging multiple ids into one preprocessed folder.

## Command

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncL -np 8
```

Merge example:

```bash
nanounet_preprocess -d 1 2 3 --merged-id 999 --merged-name Merged -np 8
```

Tiny local model:

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncTiny --patch-vol small -np 4
```

Regenerate centroid sidecars only (already-preprocessed dataset, no `.b2nd`/plans rewrite):

```bash
nanounet_preprocess -d 999 --plans-name nnUNetResEncUNetLPlans --sidecars-only -np 8
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d`, `--dataset_id` | int+ | (required) | One or more dataset ids, e.g. `-d 001` or `-d 1 2 3` for merge |
| `--merged-id` | int | `999` | Output dataset id when merging multiple `-d` |
| `--merged-name` | str | `Merged` | Name segment for merged folder `DatasetNNN_<name>` |
| `--planner` | str | `nnUNetPlannerResEncL` | Planner class (e.g. `nnUNetPlannerResEncTiny`, `nnUNetPlannerResEncL`) |
| `-np`, `--num_processes` | int | `8` | Parallel workers for fingerprint / preprocess |
| `--resume` | flag | off | Keep existing preprocess output; do not wipe `3d_fullres` folder |
| `--gpu-memory-gb` | float | none | VRAM budget (GB) for planner patch shrink loop |
| `--patch-vol` | choice | `large` | `small` (128) \| `medium` (192) \| `large` (256) \| `xlarge` (320) isotropic edge before aniso handling |
| `--plans-name` | str | none | Basename of plans JSON (no `.json`) when using `--skip-plan` |
| `--config-path` | str | none | Optional path forwarded into preprocessing |
| `--skip-fingerprint` | flag | off | Skip fingerprint; use existing `dataset_fingerprint.json` |
| `--skip-plan` | flag | off | Skip planning; requires `--plans-name` |
| `--sidecars-only` | flag | off | Regenerate `*_centroids.json` sidecars only; requires `--plans-name`; never touches `.b2nd`, plans, or `gt_segmentations` |

See [plan.md](plan.md) for `--patch-vol`, `--planner`, and `--gpu-memory-gb` trade-offs.

## Inputs / outputs

**Inputs**

- `$NANOUNET_RAW/DatasetXXX_*/` — nnUNet raw layout (`imagesTr`, `labelsTr`, `dataset.json`)
- Environment: `NANOUNET_RAW`, `NANOUNET_PREPROCESSED`

**Outputs** (under `$NANOUNET_PREPROCESSED/DatasetXXX_*/`)

- `dataset_fingerprint.json`
- `<plans>.json` — patch size, batch size, network topology
- `<plans>/3d_fullres/*.b2nd` — blosc2 training tensors
- `<plans>/3d_fullres/*_centroids.json` — per-lesion centroid, bbox, EDT seed, and voxel count
- `gt_segmentations/` — resampled labels (also under raw dataset folder)

### seed_zyx and volume_vox

The plain lesion centroid is a bad click prompt: for irregular/concave lesions it falls outside the
lesion's own mask about 12% of the time, which breaks "which component does this click belong to."
Each `*_centroids.json` therefore also carries, per lesion, `seed_zyx` — the argmax-EDT voxel, i.e.
the point deepest inside the component, guaranteed to carry that component's label — and
`volume_vox`, the component's voxel count. Training converts `volume_vox` to an equivalent-sphere
diameter to pick the lesion's size bin in the registration-error offset table, then perturbs the
stored centroid by an offset drawn from that table to simulate how a real point click drifts between
baseline and follow-up scans.

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `--skip-plan needs --plans-name` | `--skip-plan` without plans basename | Pass `--plans-name nnUNetResEncUNetLPlans` (or your existing plans file) |
| Missing raw dataset folder | Wrong id or `NANOUNET_RAW` | Check `DatasetXXX_*` exists under raw root |
| Planner OOM / tiny patch | `--gpu-memory-gb` too low for `--patch-vol` | Lower `--patch-vol` or raise `--gpu-memory-gb` to match training GPU |
| Wiped preprocess mid-run | Re-run without `--resume` | Use `--resume` to keep existing `3d_fullres` output |
| `--sidecars-only needs --plans-name` | `--sidecars-only` without plans basename | Pass `--plans-name` for the dataset's existing plans json |
| `--sidecars-only needs an existing preprocessed folder` | Target `<data_identifier>` folder missing | Run a full preprocess first (command shown in the error) |
