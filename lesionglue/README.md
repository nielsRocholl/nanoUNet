# LesionGlue

LesionGlue matches lesions between a baseline (BL) CT and a follow-up (FU) CT. Each scan becomes a graph of lesion nodes
(multi-scale intensity descriptor, volume, mean HU, sphericity, anatomy, position). A heterogeneous graph neural network
(PyTorch Geometric + Lightning, [nanochat](https://github.com/karpathy/nanochat) style code) passes messages inside each
scan and across all dense bipartite BL to FU pairs, and predicts a pair logit per edge plus a dust (no-match) logit per
node. A decode step (Hungarian by default) turns the scores into a BL to FU match table with track ids, so
disappearing and newly appearing lesions fall out as unmatched nodes. Design, losses and metrics:
[docs/technical.md](docs/technical.md).

## Deployed matcher

Local weights, nothing to download. `lesionglue_track`, `lesionglue_eval` and `segtrack_run` use this unless overridden:

| Knob | Value |
|------|-------|
| checkpoint | `/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` |
| weights | EMA (`--no-ema` to disable) |
| decode | hungarian, `dust_tau=0.125` (`--sinkhorn-tau`) |
| graph | `drop_dp=false`, `intra=complete`, `type_mask=false` (ckpt hparams) |
| holdout | cache_v7 test match **0.9701** (57 graphs) |

Node features are L0 descriptors (`DESC_DIM=1372`). The R13 v8-region retrain lost the common-set gate (0.9688) and was
not deployed; `h60_r9/best.ckpt` was 0.9453. History and encoding cost:
[docs/reference/experiments.md](docs/reference/experiments.md).

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[lesionglue]"
```

The `lesionglue` extra adds torch-geometric and friends. Weights & Biases logging is optional: `wandb login` once.

## Quickstart

Dataset: `/nnunet_data/Longitudinal-CT/` (`meta/`, `inputsTrBL/FU`, `targetsTrBL/FU`, `data_split.json`; layout in
[docs/reference/layout.md](docs/reference/layout.md)).

```bash
lesionglue_split --root /nnunet_data/Longitudinal-CT
lesionglue_preprocess --split all --jobs 16
lesionglue_eval --split test
lesionglue_track --root /nnunet_data/Longitudinal-CT --split test --out /tmp/track_test
```

Single case (`--propagated` is the BL lesion_id to centroid table in the FU voxel grid: meta CSV, slim CSV
`lesion_id,z,y,x`, or nanoUNet JSON in the FU frame):

```bash
lesionglue_track \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/0a09c8844b_00.nii.gz \
  --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/0a09c8844b_00.nii.gz \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/0a09c8844b_00.nii.gz \
  --fu-mask /nnunet_data/Longitudinal-CT/targetsTrFU/0a09c8844b_00.nii.gz \
  --propagated /nnunet_data/Longitudinal-CT/meta/0a09c8844b.csv \
  --out matches.csv
```

Output columns: `bl_lesion_id, fu_lesion_id, pair_prob, decode, track_id`. Training needs the CSV (supervision and
`cog_propagated`); deployment does not. Optional: `lesionglue_cv --config lesionglue/configs/base.json --out runs/cv`
for patient-level cross-validation.

## Commands

After install, every command is a `lesionglue_*` console script. Each step doc has the arguments, inputs and outputs,
and common errors.

| Command | Purpose | Step doc |
|---|---|---|
| `lesionglue_split` | Carve train/val/test patient split (`lesionglue/configs/split.json`) | [data.md](docs/steps/data.md) |
| `lesionglue_preprocess` | Build cached L0 graphs (`v8_native`) | [data.md](docs/steps/data.md) |
| `lesionglue_audit` | Read-only label audit | [data.md](docs/steps/data.md) |
| `lesionglue_train` | Train the matcher from a config JSON | [train.md](docs/steps/train.md) |
| `lesionglue_cv` | Patient-level k-fold training | [train.md](docs/steps/train.md) |
| `lesionglue_oof` | Score a fold checkpoint on its held-out patients | [train.md](docs/steps/train.md) |
| `lesionglue_pool` | Pool out-of-fold scores with a bootstrap 95% CI | [train.md](docs/steps/train.md) |
| `lesionglue_eval` | Validation metrics on cached val/test graphs, `--dust-tau` sweep | [eval.md](docs/steps/eval.md) |
| `lesionglue_report` | Train or load, then benchmark against the nearest-mask baseline | [eval.md](docs/steps/eval.md) |
| `lesionglue_predict` | Cached-graph edge CSVs (benchmark only) | [eval.md](docs/steps/eval.md) |
| `lesionglue_baseline_distance` | Distance-only floor (AP, AUROC) | [eval.md](docs/steps/eval.md) |
| `lesionglue_track` | Deployment: CT + instance masks in, match CSV out | [track.md](docs/steps/track.md) |
| `lesionglue_qc` | Dash viewer for one cached patient graph | [qc.md](docs/steps/qc.md) |

Flow diagram and the full quickstart sequence: [docs/index.md](docs/index.md). Config keys:
[docs/reference/config.md](docs/reference/config.md).

## In the monorepo

LesionGlue depends only on `core`. The contract with [`nanounet/`](../nanounet/) is on-disk files: nanounet writes
instance masks and click JSON, `lesionglue_track` reads them (`--bl-clicks` / `--fu-clicks` treat a mask as binary
foreground and instance-label it). [`segtrack/`](../segtrack/) composes both: `segtrack_run` predicts both timepoints
with nanounet, then matches with this checkpoint (see [segtrack/README.md](../segtrack/README.md)).

```bash
segtrack_run --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU
```
