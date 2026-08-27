# Technical reference — lesion tracking pipeline

This document describes what the **implemented** code does: graph construction, features, model, training objectives, and inference. It’s written for someone who’s comfortable with PyTorch / GNNs and longitudinal medical imaging, without re-deriving basic ML notation.

---

## Problem framing

**Input (per patient):**

- Baseline CT + integer lesion mask (`targetsTrBL/{pid}_{idx}.nii.gz`).
- Follow-up CT + integer lesion mask (`targetsTrFU/{pid}_{idx}.nii.gz`).
- Lesion-level metadata + topology labels (`meta/{pid}.csv`).

**Output (what we supervise / predict):**

- A **binary label on every dense cross-timepoint edge** \((i,j)\): baseline lesion \(i\) corresponds to follow-up lesion \(j\) or not.
- A baseline no-match label for disappeared lesions and a follow-up no-match label for newly appearing lesions.

This is **not** a strict one-to-one assignment matrix. Many-to-one (merge) and one-to-many (split) are handled naturally as **multiple positive edges** incident on the same node. Absence of any positive edge on a baseline node corresponds to disappearance; on a follow-up node, to a new lesion.

---

## Dataset assumptions (Longitudinal CT v2)

- CSV rows encode `topology_class` (normalized: `DISAPPEARING`→`DISAPPEARED`, `MERGING`→`MERGED`).
- Baseline spatial cue for registration-aligned geometry: **`cog_propagated`** (baseline COG pushed into **follow-up voxel space**).
- Follow-up cue: **`cog_fu`** (native follow-up voxel indices).
- Rows with `linking_unclear=True` are skipped.
- **Lesion ID** in the CSV doubles as the **voxel label** in the masks.

**Important implementation choice:** we **do not** fall back to `cog_bl` when `cog_propagated` is missing. Those baseline-side rows are dropped (`print0`), because mixing coordinate frames without an explicit flag violates the “no silent fallback” rule we wanted for training signal integrity.

---

## Graph construction (`tracking/data/graph.py`)

### Node sets

- **Baseline (`bl`) nodes:** rows with `topology ∈ {UNCHANGED, DISAPPEARED, MERGED, SPLIT}` **and** non-null `cog_propagated`.
- **Follow-up (`fu`) nodes:** rows with `topology ∈ {UNCHANGED, NEWLYAPPEARING, SPLIT}` **and** non-null `cog_fu`.

`MERGED` rows contribute a baseline node (the disappearing baseline identity) but **not** a separate follow-up node for that row — the positive edge targets `merged_into`, which must appear as a follow-up node from another row.

### Multi–body-region handling (`img_id_fu`)

Cross-edge distances are only meaningful when baseline propagated COG and follow-up COG live in the **same follow-up volume**. `build_hetero_data` returns one graph per follow-up body-region volume (`img_id_fu`). Rebuild into a new cache root (`cache_v8_regions`); do not overwrite `cache_v7`.

### Intra-timepoint edges

- Default: **k nearest neighbors in mm-space** (`torch.cdist` + `topk`). `intra=complete` replaces kNN with a directed complete graph (`i≠j`) into the same intra `TransformerConv` (`edge_dim=1`).
- **`edge_attr`:** Euclidean distance / 100 in mm. `drop_dp` zeros the first 5 `cross_attr` channels (`dp`/`dist`); `CROSS_DIM` stays 27.
- Isolated node → **self-loop** with distance 0 so `TransformerConv` always sees a relation. `type_mask` keeps only same-`lesion_type` intra edges (not cross).

### Cross edges and labels

1. **Dense graph:** all baseline-follow-up pairs \((i,j)\), no radius prefilter.
2. **Positive labels:**
   - `UNCHANGED` / `SPLIT`: \((\text{lesion\_id}, \text{lesion\_id})\) when both COGs exist on that row.
   - `MERGED`: \((\text{lesion\_id}, \text{merged\_into})\).

Labels are **0/1** stored as `data['bl','cross','fu'].edge_label`. Baseline/follow-up node stores also carry `no_match_label`. Reverse edges `('fu','cross','bl')` duplicate indices flipped for bidirectional message passing and carry sign-flipped relational features.

---

## Node features (`appearance.py` + packing in `graph.py`)

Per-node vector dimension depends on `--feat` (`l0`: **1387**, `mae`: **335**, `yerebakan`: **6875**), with descriptor/HU clipped to `[-1000, 1000] / 1000` for hand-crafted modes, `log1p(volume_mm³) / 10`, and sphericity clipped to `[0, 2]`:

| Block | Size (l0) | Meaning |
|-------|------|---------|
| Descriptor | 1372 | L0 **Yerebakan-style** multi-scale intensity samples (fixed offsets in **world mm**, trilinear sampling back into index space). `yerebakan` mode concatenates 5 hierarchy levels (6860-D). `mae` mode uses 320-D masked encoder pool. |
| Mask stats | 3 | `log1p(volume_mm³)`, mean HU inside lesion mask, sphericity from voxel-face surface area. |
| Anatomy index | 1 | Integer index into fixed `LESION_TYPES` vocabulary (embedded in the encoder). |
| Position | 3 | Normalized follow-up-space coordinates: COG / `(shape − 1)` per axis (baseline uses **propagated** COG here). |

The encoder (`matcher.NodeEncoder`) splits this tensor, applies `nn.Embedding` on the anatomy index, concatenates, then runs a shallow MLP → model width `d` (default 128).

---

## Model (`tracking/matcher.py`)

**Architecture:**

1. Shared **NodeEncoder** for both node types.
2. Stack of **`HeteroConv`** layers, each bundling four **`TransformerConv`** modules:
   - `bl → bl` (intra), `fu → fu` (intra) — `edge_dim=1`.
   - `bl ⇄ fu` (cross, both directions) — `edge_dim=11`: delta mm, distance, log-distance, volume/HU/sphericity differences, same-type flag.
3. Residual + ReLU update per layer (`x := ReLU(x + conv_out)`).
4. **Edge head:** MLP on `concat(h_bl[src], h_fu[dst], cross_edge_attr)`.
5. **No-match heads:** one linear head per baseline node and per follow-up node.

**Output:** `MatcherOutput(pair, bl_no_match, fu_no_match)`.

**`decode_hungarian`:** optional post-processing that builds an \((N_{\text{bl}}+1)×(N_{\text{fu}}+1)\) cost matrix from `-log p`, adds a dustbin row/col with threshold costs, and runs `linear_sum_assignment`. Used only when CLI `--strict` is set; training itself does **not** enforce one-to-one structure.

---

## Training (`tracking/train/module.py`)

**Loss:** dense pair BCE with train-split `pos_weight`, row-wise soft-target cross entropy over each baseline row plus a dustbin, and BCE for BL/FU no-match heads. Default weights: pair `1.0`, row `0.5`, no-match `0.2`. Stored in `{split}_v2_meta.pt`; `MatcherDataModule` reads **`train_v2_meta.pt`**.

**Metrics:**

- **AUROC / Average Precision** on edge scores (torchmetrics), aggregated at epoch end on validation.
- **`val_row_acc`:** per-graph row accuracy — argmax over dense follow-up row plus baseline dustbin; no-positive rows are correct only if the dustbin wins.

**Optimizer:** AdamW + `ReduceLROnPlateau` on `val_loss`.

**Batching:** PyG `DataLoader` batches heterogeneous graphs; `edge_index_dict` and batched `edge_label` are handled by the library.

---

## Caching (`tracking/data/dataset.py`)

`LesionDataset` subclasses `InMemoryDataset`. `process()` walks `data_split.json` for the requested split, builds a Python list of `HeteroData`, collates via PyG’s built-in `save`, and writes `processed/{split}.pt`. Preprocessing is intentionally **offline** so training I/O stays light.

---

## Deployment inference (no CSV)

The **trained weights** only need a `HeteroData` that matches what `build_hetero_data` produces: same node feature layout (1379-D), same dense cross edge contract, same intra/cross `edge_attr`, and node no-match tensors. **Supervision (`edge_label`) is unnecessary** at inference.

**Inputs you described** (predicted components + CTs + optional site):

1. **Volumes:** baseline CT + follow-up CT (aligned whatever way your registration pipeline assumes when you compute propagated points).
2. **Masks:** integer instance labels per lesion on each scan (your segmenter’s connected components).
3. **Per baseline lesion:** centroid (or COG) in **baseline voxel indices** → drives `descriptor_l0` on the baseline volume and `mask_stats` on the baseline mask.
4. **Per follow-up lesion:** centroid in **follow-up voxel indices** → same on the follow-up side.
5. **Propagated baseline centroid:** same baseline centroid expressed in **follow-up voxel coordinates**, using the **same deformation / registration** family you used (or will use) clinically. This replaces `cog_propagated` from the CSV. It feeds:
   - normalized position channels (last 3 floats of the baseline node feature vector),
   - mm-space position used for dense pair features and intra-BL kNN context (baseline branch stores mm coords derived from propagated × follow-up spacing in the current code path).
6. **Anatomy:** map your coarse label (liver, lung, …) to an index in `LESION_TYPES`; use `'unclear'` if unknown.

**Cross-edges:** build every baseline-follow-up pair and the 11-D relational feature vector. No labels are known in deployment.

**Gap in repo:** there is **no** standalone `predict_from_masks.py` yet; `cli/predict.py` reads **cached splits whose graphs were built with CSV during preprocess**. Adding a thin builder that duplicates `build_hetero_data` logic without CSV rows is straightforward engineering once propagation is available in your inference stack.

---

## Inference CLI (`tracking/cli/predict.py`) — benchmark / evaluation only

Loads `MatcherModule` from checkpoint and runs forward on **`LesionDataset`** graphs (built by `preprocess.py`, which **parses CSV** for nodes, propagated centres, dense edge features, and labels). Output CSV: `bl_lesion_id`, `fu_lesion_id`, `prob`, `decoded`.

---

## Known limitations / sharp edges

1. **`predict.py` is not the production path** without CSV-backed preprocess; see **Deployment inference** above.
2. **Dominant `img_id_fu`:** rare multi-region patients may discard minority regions entirely.
3. **Dropped baseline rows** without `cog_propagated`: label noise if annotations omit propagation but the lesion still exists — acceptable only if rare.
4. **Class imbalance metrics:** AUROC can be unstable when a mini-batch has only negatives or only positives.
5. **`DATASET_ROOT`** default is machine-specific in `common.py`; override with `--root` for portability.

---

## Dependency rationale

- **PyG:** native heterogeneous batches and conv APIs.
- **Lightning:** checkpointing, logging hooks, LR scheduling without custom trainer classes.
- **nibabel + scipy:** NIfTI I/O and Hungarian / sparse sampling helpers.
- **pandas:** CSV parsing only in `meta.py`.

---

## File → responsibility map

| Module | Responsibility |
|--------|----------------|
| `common.py` | Paths, constants, JSON helpers, `print0`, seed |
| `data/meta.py` | `V2Paths`, strict CSV → `LesionRow` |
| `data/appearance.py` | Descriptor offsets + `mask_stats` |
| `data/graph.py` | `GraphConfig`, `build_hetero_data` |
| `data/dataset.py` | `LesionDataset`, `pos_weight` sidecar |
| `matcher.py` | Full model + Hungarian helper |
| `train/datamodule.py` | Loads caches, exposes PyG loaders |
| `train/module.py` | Loss, metrics, optimizer config |
| `cli/*.py` | Procedural entrypoints |

This matches the running code; if `blueprint.md` disagrees on Sinkhorn or feature recipes, trust the implementation and this file.
