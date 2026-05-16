# Lesion tracking (longitudinal CT)

This repo trains a small graph neural network to **match lesions between a baseline CT and a follow-up CT** for the same patient: baseline nodes on one side of a dense bipartite graph, follow-up nodes on the other, and the model outputs **pair scores plus no-match scores** for disappeared/new lesions.

---

## Training vs inference (read this once)

**Training** needs the full research dataset layout: paired CTs, integer lesion masks, **and** the official `meta/{patient}.csv`. The CSV is doing two jobs: (1) **supervision** — which baseline lesion links to which follow-up lesion (unchanged / merge / split / disappear / new); (2) **geometry the model was built around** — especially **`cog_propagated`**, i.e. the baseline lesion centre warped into **follow-up voxel space** with your conventional registration. Dense BL↔FU pair features and the baseline node positions in that shared frame come from that pipeline.

**What you will have at real inference** is different: **no CSV**, only **predicted instance masks** (connected components with IDs), the **two CT volumes**, and maybe a **coarse organ/site label** per lesion (liver vs lung, etc.). That is enough *in principle* to **run the same kind of model**, but you still need to reproduce what the CSV gave you for free:

| Piece | Training (today) | Deployment (what you described) |
|--------|-------------------|----------------------------------|
| Who are the nodes? | Rows + masks | One node per predicted component on each scan |
| Follow-up position | `cog_fu` from CSV (same as COG of mask) | Centroid (or COG) of that component in **follow-up** mask space |
| Baseline appearance descriptor | Sample CT at baseline COG | Same: centroid of baseline component in **baseline** volume |
| Follow-up appearance descriptor | Sample CT at follow-up COG | Centroid of follow-up component |
| Volume / shape stats | From masks | From your predicted masks |
| Anatomy embedding | `lesion_type` from CSV | Map your coarse location → same vocabulary as `LESION_TYPES`, or use `'unclear'` if you don’t trust it |
| **Where to pair baseline ↔ follow-up in space** | **`cog_propagated`** (baseline COG in **follow-up** voxel space) | You need the **same thing**: baseline centroid **warped into follow-up space** using your registration field (or an equivalent). Without that, you’re not feeding the network the spatial signal it was trained on unless you change the architecture / retrain. |

So: **the CSV is not required forever** — it’s required **today** because our preprocess builds graphs from it (labels + propagated centres). For production you’d build the same `HeteroData` from masks + CT + **registration output** + optional anatomy; then load the checkpoint and run `Matcher` forward. That path is **not** exposed as a finished CLI yet; `predict.py` in this repo still assumes you ran `preprocess.py` on the benchmark dataset (graphs built with CSV).

It does **not** run registration or your segmentation model here — those stay upstream — but **at inference you still need a propagated baseline centre per lesion** if you want behaviour consistent with how this code was trained.

---

## What you need for the commands below (dataset benchmark mode)

- Python 3.10+ (what we’ve used: 3.13 in a venv is fine)
- PyTorch, PyTorch Geometric, Lightning, the rest in `requirements.txt`
- On disk: `meta/{patient}.csv`, `inputsTrBL/FU`, `targetsTrBL/FU`, and `data_split.json` (**Longitudinal CT v2** layout — see that dataset’s `README.md`)

The default dataset path is **hardcoded** in `tracking/common.py` as `DATASET_ROOT`. If your data lives somewhere else, either change that constant or pass `--root` on every CLI command below.

---

## Install

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`rich` and `wandb` are listed in `requirements.txt`. For Weights & Biases, run `wandb login` once on the machine.

Imports assume the package is on `PYTHONPATH`, so run scripts like this **from the repo root**:

```bash
export PYTHONPATH=.
```

---

## How you actually use it

**1. Build cached graphs** (slow-ish first time; reads NIfTIs and builds one PyG graph per patient):

```bash
PYTHONPATH=. python tracking/cli/preprocess.py --split train
PYTHONPATH=. python tracking/cli/preprocess.py --split val
# optional:
PYTHONPATH=. python tracking/cli/preprocess.py --split test
# faster on big RAM machines (see note below):
# PYTHONPATH=. python tracking/cli/preprocess.py --split train --jobs 6
```

This writes under `.cache/graphs/processed/`: `train_v2.pt`, `val_v2.pt`, plus `{split}_v2_meta.pt` (e.g. `pos_weight` for the dense pair loss).

On a machine with plenty of RAM (e.g. your 48 GB M4 setup), you can overlap patients with **`--jobs N`** (`ProcessPoolExecutor`). Rule of thumb: each concurrent worker peaks at roughly **one patient’s worth** of CT/mask load + descriptor work — start with **4–8**, watch Activity Monitor, and back off if memory pressure spikes. Default stays **`--jobs 1`** so laptops don’t get surprised.

**2. Train**

```bash
PYTHONPATH=. python tracking/cli/train.py --epochs 200 --out lightning_logs
```

- Progress: Rich-based bar via PyTorch Lightning’s `RichProgressBar` (requires `rich` from `requirements.txt`).
- Optional Weights & Biases: `pip install` picks up `wandb`; run `wandb login` once. Logging turns on with `--wandb` or whenever `--wandb-run-name` is non-empty.

```bash
PYTHONPATH=. python tracking/cli/train.py --wandb-project lesion-tracking --wandb-run-name my-run
# same with explicit flag:
PYTHONPATH=. python tracking/cli/train.py --wandb --wandb-project lesion-tracking --wandb-run-name my-run
```

Checkpoints land under `--out` (Lightning + `ModelCheckpoint` on `val_loss`).

On Apple Silicon, training uses the **MPS** backend when PyTorch reports it available; otherwise Lightning’s **`accelerator=auto`** picks CUDA or CPU. There is no extra CLI flag — device selection is automatic from the environment.

**3. Predict / export edges** (benchmark mode; still uses graphs from step 1 — those graphs were built **with** the CSV for node lists, propagated centres, dense edge features, and labels)

```bash
PYTHONPATH=. python tracking/cli/predict.py --ckpt path/to.ckpt --split val --out preds
```

By default it only writes rows above `--thresh` (0.5). Use `--dump-all` for every cross-edge. `--strict` turns on a Hungarian-style 1:1 decode for the `decoded` column (see `technical.md` if you care why).

For **CSV-free deployment**, you’d call the same `Matcher` on a graph built from your masks + CT + registration-warped baseline centroids; see **`technical.md`** for the checklist.

```bash
PYTHONPATH=. python tracking/cli/predict_masks.py \
  --bl-img bl.nii.gz --bl-mask bl_instances.nii.gz \
  --fu-img fu.nii.gz --fu-mask fu_instances.nii.gz \
  --propagated propagated_centroids.csv \
  --default-lesion-type unclear \
  --ckpt path/to.ckpt --out best_matches.csv --pairs-out dense_pairs.csv
```

---

## Where to read more

- **`technical.md`** — how the graph is built, what the model is doing, loss and metrics, limitations (multi-scan patients, missing propagated COGs, etc.).
- **`blueprint.md`** — older design notes; some of it still talks about Sinkhorn/SuperGlue. The **code** is edge classification + PyG + Lightning; treat `blueprint.md` as background, not the spec.

---

## Repo layout (short)

| Path | Role |
|------|------|
| `tracking/common.py` | Default paths, constants, tiny helpers |
| `tracking/data/` | CSV parsing (train/eval graphs), descriptors, `build_hetero_data`, `LesionDataset` |
| `tracking/matcher.py` | Encoder + heterogeneous GNN + edge head |
| `tracking/train/` | Lightning `DataModule` + `LightningModule` |
| `tracking/cli/` | `preprocess`, `train`, `predict` |

That’s it. If something breaks, it’s usually a missing file under the dataset tree, a patient skipped because there’s no valid baseline/follow-up side, or you forgot `PYTHONPATH=`.
