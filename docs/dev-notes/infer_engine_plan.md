# Plan: replace `nanounet_predict` with a grid engine

Status: implementation spec. One way. nanochat-style is a hard rule (R1 <200 LOC/file, R3 one path, G4 measure).
Serves the Dataset999 instance-conditional run: 3-ch `[CT, hm+, hm−]`, patch `[96,160,160]`, EDT r=2, scale=0.5. Longi/DWB is the same engine with a 6-ch row — do not fork.

---

## 0. What we are building

Prompt-ROI inference: pack clicks into the fewest covering tiles, forward those as one GPU batch, then grow a **per-cluster integer grid** only where predicted FG hits a tile face. Output = binary FU seg.

Not a sliding window. Not the current greedy-BFS-cc3d expander.

---

## 1. Evidence (measured, not guessed)

**Training distribution** — 5866 Dataset999 centroid sidecars, exact preprocessed voxels, patch `[96,160,160]`:

| Fact | Number |
|------|--------|
| Lesions | 20033; /case p50=1 p95=12 max=101 mean=3.42 |
| Lesion bbox vs patch | p50 `(8,14,13)` p95 `(51,105,106)` max `(258,421,375)` |
| **Overflow** (needs expand) | **2.44%** of lesions (z/y/x 0.98/1.65/1.88%) |
| Half-step extras to cover overflow | p95=0, **max=10** → cap 16 is enough |
| Cases whose **all-centroid AABB** fits one patch | 73.5%. **FAR = 26.5%** (58.6% of multi-lesion cases) |
| Greedy click clusters | p50=1 p95=**4** max=17; 65.8% of cases are 1 cluster |
| nnUNet sliding window on a 512²×256 CT @ stride 0.75 | ~65 tiles, **every case**, including 1-lesion |

**Deployment** — all 336 `targetsTrFU` (SimpleITK label stats, ~80s):

| Fact | Number |
|------|--------|
| Instances | 3103; /case min=0 p50=**4** p95=**31** max=**95** |
| Resampled bbox p50 / p95 / max | Z `7/36/199`  Y `11/50/256`  X `12/48/346` |
| **Overflow** | **0.58%** (18/3103) — even rarer than d999; expand still load-bearing on those 18 |
| All-centroid AABB fits one patch | **38%**. Multi-lesion FAR = **79%** (207/262) |
| Native volume | p50 `(263,512,512)` p95 `(441,512,512)` max Z=1393 |
| Clicks vs instances | exact match **37%**; clicks p50=7 vs inst p50=4 (often extra clicks). JSON: `points[].point=[x,y,z]` |

One global grid is *worse* here than on d999. Extra clicks just add seeds; clustering still packs them. Eval GT = `(label > 0)`. Overflow examples for G1/G2: `0a09c8844b_00`, `1afa34a7f9_00`, `2a79ea27c2_00`.

**Training already assumed expand** (`nanounet/data/instance_target.py`): membership is voxel-overlap so a large lesion stays FG in neighbouring tiles; a kept lesion always gets an in-crop click (on its own tissue, not patch centre). Current infer plants a **phantom centre click** when the user click left the tile — that is not the training fallback.

---

## 2. Why the current engine is wrong

1. **Not a grid.** `cluster_points_for_patch_size` is the right *seed* step (and we keep it). Expand is irregular half-step BFS + cc3d hull components + host `.cpu().numpy()` per extra tile.
2. **Serial GPU.** Seeds batch; every expand tile is `B=1`. TTA is **8 sequential** `net()` calls (`tta.py`).
3. **Phantom click.** `local_prompt_points_for_patch` → patch centre if the click is outside. Instance-conditional: empty heatmap = “leave it”. Centre heatmap = “there is a lesion here”. Both are wrong on expand tiles. Training puts the click **on the lesion in this crop**.
4. **`--border-expand` off by default** even though 2.44% (d999) / 0.58% (FU) of lesions need it. `--merge average` + `gaussian.py` + `tile_step_size` are dead/legacy.
5. **`predict_patch_logits`** is a second path (no expand). Radiom large-lesion clicks stop at the seed FOV.

---

## 3. Algorithm (the one way)

Per case, after `resolve_pts_pad` (unchanged):

### 3.1 Seeds — keep greedy clustering

`cluster_points_for_patch_size(pts, patch, margin_frac=0.1)` then `spatial_slices_covering_points`. One covering tile per cluster. This is the “smallest set of patches that contains all prompts” under the constraint **do not tile the gap** (forced by FAR=26.5%).

`--inference-mode centered` = do not merge clicks (one cluster per point). Same engine, one `if`.

### 3.2 Grid — replace `border_expand.py`

Each cluster owns a lattice:

```
stride[d] = max(1, int(round(patch[d] * tile_step_size)))   # default 0.5 until gate G3
origin    = covering-tile start (z0,y0,x0)
cell (i,j,k) → start = origin + (i*sz, j*sy, k*sx), clamped to padded volume
```

Seed cell is `(0,0,0)`. Neighbours are 6-connected in `(i,j,k)`. Face contact (GPU, no cc3d):

```
fg = logits.argmax(0) > 0          # stay on GPU
touch[+z] = fg[-1].any(); ...      # 6 bools
```

If `touch[+z]` and `(i+1,j,k)` not visited and cluster extra < cap: enqueue. **Cap = 16 / cluster** (measured max 10).

### 3.3 Prompt on a tile (load-bearing)

```
loc = cluster_prompts_patch_local(user_clicks, sz, sy, sx)
if loc: encode loc
elif expand_tile:
    loc = [fg centroid on the contacting face of the PARENT logits]
    # this is training's fallback: click on in-crop tissue of the kept lesion
else:
    assert False  # seed tiles cover their clicks by construction
```

Delete `local_prompt_points_for_patch`'s centre fallback. Empty heatmap only when `--no-prompt-encode`.

### 3.4 Loop — one queue, always batched

```
pending = seed cells of all clusters          # typically p95=4
visited = set(keys)
while pending:
    batch = pending[:batch_size]
    rows  = encode all
    logits = predict_batch_with_tta(net, stack(rows), use_tta)   # fused TTA
    accumulate max-merge into padded logit volume
    if expand: enqueue unseen face-neighbours with face-FG click
```

Seeds and later expand tiles share this loop. No `B=1` path.

### 3.5 Merge — `max` only

Per-voxel winner = highest `max(fg)−bg` margin. Delete `average` and `gaussian.py`. Uncovered voxels stay the bg logit vector.

### 3.6 TTA — one fused forward

`predict_batch_with_tta`: `torch.cat` the 8 mirror copies along batch, **one** `net()`, unflip, mean. If `8B` does not fit, chunk by 2–4 mirrors — never 8 python `net()` calls. Delete unused `predict_with_optional_tta`.

Default TTA stays **on** (`disable_tta_default: false`) until gate G4 says otherwise. Interactive `predict_patch_logits` stays TTA-off (latency).

---

## 4. File map

| File | Action | Notes |
|------|--------|--------|
| `nanounet/infer/border_expand.py` | **Delete** | Replaced by grid in `prompt/cluster.py` + loop in `predict_case.py` |
| `nanounet/infer/gaussian.py` | **Delete** | `accum_dtype` (8 lines) moves into `predict_case.py` |
| `nanounet/prompt/cluster.py` | Edit | Add `cell_slices(origin, ijk, stride, patch, shape)` and `face_neighbours(ijk, touch6)` — keep clustering |
| `nanounet/infer/predict_case.py` | Rewrite | Queue loop above. Must stay <200 LOC. Drop unused `dj` |
| `nanounet/infer/tta.py` | Rewrite | Fused batch TTA. ~40 LOC |
| `nanounet/infer/longi_row.py` | Edit | `encode_inference_row(..., extra_clicks=())`; no centre fallback |
| `nanounet/infer/roi_slices.py` | Edit | Delete `spatial_slices_from_lbs`, `local_prompt_points_for_patch` |
| `nanounet/infer/predict_patch.py` | Keep signature | Still one centered forward. Docstring: large lesions → `predict_case_logits` + expand. Radiom changes later |
| `nanounet/cli/predict.py` | Edit | `--border-expand` **default on** (`store_false` via `--no-border-expand`). Drop `--merge`. Wire `tile_step_size` from config |
| `nanounet/cli/predict_preprocessed.py` | Same flag defaults | Still `longi=True`; same engine |
| `docs/steps/predict.md` | Same change | Args table, recommended command without `--border-expand` (it is default) |
| `docs/dev-notes/radiom_embed_api.md` | Note | `predict_patch_logits` unchanged; full-case expand is `predict_case_logits` |

Do **not** add `infer/grid.py`. `infer/` is already over the ~6-file guideline; we delete two files.

Keep: `predictor.py`, `predict_io.py`, `export.py`, `patch_export.py`, `points_pad.py`. Preprocess stays full-volume `run_case` — one resample per case, not the bottleneck.

---

## 5. CLI / Radiom contract

```bash
nanounet_predict -i /nnunet_data/Longitudinal-CT/inputsTrFU \
  -o /tmp/nanounet_infer_probe \
  -m /nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_instance_1200ep \
  --ckpt last.ckpt --batch-size 16 --device cuda
```

Expand on, clustered, max-merge, TTA from `nano_config.json`. `--no-border-expand` / `--disable-tta` / `--inference-mode centered` are the ablations.

`predict_case_logits` keeps `points_zyx_unpadded` and `on_forward(done, total)` (Radiom). `on_forward` fires once per finished tile after TTA mean, including expand tiles. Import `Callable`.

This 999 ckpt is **not** longi. `nanounet_predict_preprocessed` still forces `longi=True` — do not point it at this run.

---

## 6. Verification gates (real data, in this order)

Eval set: `inputsTrFU` + `targetsTrFU`, binary GT = `(label > 0)`, clicks = sibling JSON. Probe script is temporary (`scripts/_probe_infer.py`), delete after (R16). Checkpoint: instance-1200ep `last.ckpt`; if the job is mid-run that is still the right weights. Report **n_forwards, wall s, binary Dice**.

| Gate | What | Pass |
|------|------|------|
| **G0** | n_forwards on 20 mixed FU cases (incl. a 45-click FAR case) | clustered ≪ ~65 sliding tiles; FAR cases do not emit a thorax-sized grid |
| **G1** | Prompt A/B/C on the overflowing subset (bbox > patch): (A) empty (B) patch-centre (C) face-FG click | **C ≥ A and C ≥ B** on Dice of those lesions. Locks §3.3 |
| **G2** | Expand on vs off on the same overflowing subset | on beats off (otherwise expand is theatre) |
| **G3** | `tile_step_size` ∈ `{0.5, 0.75, 1.0}` on overflowing subset | pick the smallest n_forwards whose Dice is within 0.5 of the best; write that into `InferenceConfig` default |
| **G4** | TTA on vs off, fused vs old serial (time only for fused) | fused ≤ serial+5%; Dice decides whether default TTA stays |
| **G5** | Batch 8 vs 16 vs 32, GPU util / step ms | lock `--batch-size` default; H200 should sit near full util on the 45-click cases |
| **G6** | Regression: 20 random FU cases, new engine vs current `nanounet_predict --border-expand --merge max` | Dice ≥ old − 0.5; n_forwards ≤ old |

G4 is a data-path change → G4 numbers go in the commit message (G4 rule).

---

## 7. Non-goals

No sliding window (`tile_step_size` = grid stride). No second predictor class. No ROI-only preprocess. No change to 999 training / heatmap / patch size. No Radiom UI (`predict_patch_logits` signature stays). No `merge=average`.

## 8. Order

1. Fused TTA (`tta.py`). 2. Grid + `predict_case.py` + prompt rule; delete `border_expand.py` / `gaussian.py` / centre fallback. 3. CLI defaults + `docs/steps/predict.md`. 4. Gates G0–G6; lock stride + TTA from numbers. 5. `wc -l` < 200; `graphify update .`

If `predict_case.py` blows 200: extract accumulate/merge into `roi_slices.py`, not a new file.
