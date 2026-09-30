# Data, cache and code layout

## Dataset

`/nnunet_data/Longitudinal-CT/` (override with `--root`):

| Path | Content |
|---|---|
| `meta/{patient}.csv` | Lesion rows: topology, `cog_propagated`, `cog_fu`, `lesion_type` (see [technical.md](../technical.md)) |
| `inputsTrBL/`, `inputsTrFU/` | CT NIfTIs (`{pid}_{idx}.nii.gz`) |
| `targetsTrBL/`, `targetsTrFU/` | Integer lesion instance masks |
| `data_split.json` | Official train / val / test patients |

The tracking split is `lesionglue/configs/split.json` (written by `lesionglue_split`): 240 train/val patients, 60 holdout.

## Cache

`{CACHE_ROOT}/processed/{split}_v7_native.pt` (default `CACHE_ROOT=/nnunet_data/lesion_tracking/cache`), built by `lesionglue_preprocess`.
Multi-region graphs (one per follow-up body-region volume) live in `/nnunet_data/lesion_tracking/cache_v8_regions`; that cache does not overwrite v7.

## Training vs deployment

- **Preprocess and train need the CSV:** it gives the supervision labels and `cog_propagated`.
- **Deployment** (`lesionglue_track`) needs only CT + instance masks + propagated centroids in the FU voxel grid; no CSV supervision.
- `lesionglue_predict` is a cached-graph benchmark, not a deployment path.

## Code layout

```
lesionglue/
  config.py             # Config dataclass + JSON load/save
  common.py             # paths, DEPLOYED_CKPT / DEPLOYED_DUST_TAU, print0, seed
  infer.py              # CSV-free inference used by lesionglue_track
  configs/              # base.json (kNN), complete.json (deployed graph recipe), nodp*.json, split.json
  model/                # matcher (encoder + heterogeneous GNN + edge head), decode, sinkhorn, matchability
  data/                 # graph build, dataset/cache, features (L0), instances, source
  train/                # Lightning DataModule + Module, objective, val_score
  eval/                 # report, bootstrap, qc_view
  baselines/            # nearest_mask
  cli/                  # one file per lesionglue_* command
  scripts/
    round9.sh           # CV -> optional final + tau sweep + test gate
    lesion-round9-cv.sh # SLURM wrapper for round9.sh
```

## Common failures

- Missing NIfTI or CSV under `--root`.
- Empty BL or FU side.
- Missing `lesionglue/configs/split.json`: run `lesionglue_split`.
- Stale `{split}_v6_h60.pt` files in `processed/` are ignored; the cache tag is `v7_native`.

Per-command error messages and fixes are in the `Common errors` tables of [steps/](../steps/).
