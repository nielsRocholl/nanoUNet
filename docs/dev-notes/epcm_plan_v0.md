# EPCM: error-predictive confidence for nanoUNet

Date: 2026-09-23
Status: archived — superseded by `epcm_plan.md`; kept as history.

Auxiliary head predicting **expected signed boundary displacement** per voxel, from frozen
nanoUNet decoder features + its own prediction. One forward pass. AlphaFold pLDDT/PAE structure,
segmentation target.

Constraint that shapes everything: **no segmenter retraining.** One week per run. We use the
existing promptable checkpoint, frozen, and its held-out fold-0 val split (~1200 scans).

## Locked decisions

| # | Decision | Why |
|---|---|---|
| D1 | Head predicts binned **signed** δ (mm), not a 5-level score | δ is continuous → bins are meaningful; sign separates over/under-seg |
| D2 | pLDDT-style score `S` is a **readout**, not a target | Derived from the predicted δ distribution. One head, all metrics |
| D3 | δ is **surface-supported**: thin extension (±2 voxels), soft weights, imbalance fixed by sampling not masking | δ is a property of the boundary, not of distant tissue. See "Why not a wide band" |
| D3b | Loss **normalised per connected component** via a global per-component shell count stored in the cache | Surface voxels scale with area; micro-averaging hands the objective to huge lesions and starves nodules |
| D3c | Every metric — and P0's error budget — **stratified by lesion diameter** (<5, 5–10, 10–30, >30 mm) | Achievable accuracy is size-dependent; a pooled number hides it in both directions |
| D4 | Backbone frozen, target detached | See "AlphaFold precedent" below. Also forced: 1 week/run |
| D5 | Train on held-out fold-0 val, never on segmenter's train set | Train-set predictions are memorised → head learns to be overconfident |
| D6 | Persist mean + spread everywhere, full 13 bins in the shell | Spread is the ambiguity signal; unrecoverable later without re-running inference |
| D7 | No deep ensemble / MC-dropout baseline | Needs training runs we cannot afford. Stated as a limitation |
| D8 | Prompted regime only | Every lesion is prompted → missed lesions are rare → the task is boundary quality |

## Target

`SDT_gt` = signed distance transform of the GT mask, positive outside, anisotropic
(`distance_transform_edt(sampling=spacing)`).

```
delta(v) = SDT_gt(nearest predicted-surface point to v)      # mm, signed
```

Read: *how far, and in which direction, must the predicted boundary move to be right near v.*
`delta > 0` prediction bulges past truth. `delta < 0` prediction falls short. Whole band gets the
local displacement — every voxel in a 2 mm strip is labelled 2 mm.

Kept in **absolute mm**, not normalised by lesion size. δ in mm is the physically real quantity — how
far the radiologist moves the contour — and it is scanner-comparable. Scale sensitivity belongs in the
readouts: predicted Dice and volume error integrate δ over the surface and pick it up automatically.

Bins (13), log-spaced and signed, breaks in mm:

```
[-16, -8, -4, -2, -1, -r, +r, 1, 2, 4, 8, 16]      r = max(1.0, 2 * min(cm.spacing))
```

Centre bin `(-r, +r)` = "correct to within r". `r` is resolved from `plans.json` at startup, not
hardcoded — a voxel EDT quantises distance, so a break finer than ~2 voxels measures quantisation
rather than model error. Tails saturate.

### Why not a wide band

The first draft masked the loss to ±8 mm around the predicted surface. Four problems, and the fourth
is structural:

1. **Semantics.** A voxel 8 mm out in air gets the δ of *its nearest surface point* — a fact about the
   boundary, not about that voxel. At large radius this is exactly the "confidence bleeding into air"
   failure from the original notes, reintroduced.
2. **Hard edge.** Supervision stops abruptly at 8 mm; output just outside is unconstrained
   extrapolation, and masking the display produces a ring that means nothing anatomical.
3. **Wrong tool.** The band existed to fix class imbalance. Stage 2 runs on a cache with full control
   of sampling, and the repo already does mode-based patch sampling. Fix imbalance at the sampler with
   soft distance weights; don't encode a statistical fix as geometry.
4. **It cannot express the worst errors.** The band hangs off the *predicted* surface. Under-segment a
   confluent mass by 20 mm and the missed region lies outside the band, unlabelled and unreportable.

Point 4 does not get fixed by a bigger radius, because large errors are a different **kind** of error:

| Type | Description | Parametrisation |
|---|---|---|
| A — displacement | predicted surface exists, roughly right place, off by δ | signed δ on the surface |
| B — presence | GT region with no predicted surface near it, or predicted region with no GT | binary presence error, not a displacement |

Forcing B into a displacement field is what broke the original `D(v)`. So δ stays surface-supported,
and B — *if* P0 shows it matters in the prompted regime — becomes a second output on its own domain
(coarse resolution, body mask). Consequence for the build: the head is written as surface-supported
prediction, not as a dense field with a mask, so adding output B later is additive.

### Why not the D(v) from the original notes

Old rule: for a false positive, `D` = distance to the GT boundary. In the worked over-segmentation
example the FP voxel adjacent to the GT boundary gets `D = 0.5` → `S = 1.0`, "very accurate". It is a
false positive. It is 100% wrong.

`D` measured *depth into the error strip*, not *size of the error*. For a thin strip that is small
everywhere regardless of how bad the segmentation is. lDDT has no such behaviour: a badly placed
residue scores low, it does not inherit a high score from a correct neighbour.

Second reason: `S = ¼Σ 1[D < t_k]` at a single voxel takes only 5 values, so "expected value over
bins" has nothing to average. lDDT is continuous because it averages over *neighbours* (all atoms
within 15 Å). Binned δ recovers continuity directly.

### Readouts (no extra parameters)

| Readout | From |
|---|---|
| `S(v) ∈ [0,1]`, pLDDT-style | `Σ_k p_k · ¼Σ_j 1[abs(center_k) < t_j]`, `t = 0.5,1,2,4` mm; optional 15 mm ball smoothing for display |
| predicted ASSD / NSD@2mm / HD95 | surface integrals of the predicted δ distribution |
| predicted volume error | `ΔV ≈ ∮ δ dA` |
| predicted Dice per lesion | from predicted ASSD + component geometry |

Naive per-voxel independence makes volume CIs far too tight — surface errors are spatially
correlated. Fix by calibrating aggregate scalars on a held-out calibration split (split conformal),
not by modelling the correlation. Phase 3.

## AlphaFold precedent for D4/D5

AlphaFold does it two different ways, and we follow the second.

| Head | How trained | Config |
|---|---|---|
| pLDDT | **jointly, during training**, moving target recomputed each step | `weight: 0.01`, target `stop_gradient`ed, input representation **not** detached → gradient reaches the trunk |
| PAE | **afterwards**, fine-tuned onto a finished model, separate `*_ptm` checkpoints | `weight: 0.0` base, `0.1` in `*_ptm` |

The moving target is not the risk — target and prediction are always a matched pair, so it co-adapts
rather than going stale. The risk is **memorised** targets: a jointly trained head learns the error
statistics of training-set predictions, which is only valid if those resemble test-set predictions.
True for AlphaFold at ~100k structures. Not true for nanoUNet at 4800 scans × 1000 epochs.

Second reason to freeze even if compute allowed it: at nonzero weight the model can cut the
confidence loss by making its errors *more predictable* rather than smaller (smoothing boundaries
until error is a clean function of geometry). Negligible for AlphaFold at 0.01 against a 1.0 structure
loss; not negligible for a small head on a small model.

P0 measures this instead of assuming it: sweep ~200 train-split scans alongside fold-0 val and
compare δ distributions. Matching → memorisation absent, joint training would have been free (useful
for the next nanoUNet). Diverging → D5 is vindicated with a number, and the number goes in the paper.

## Head

Tap: `register_forward_pre_hook` on `net.decoder.seg_layers[-1]` — captures the full-res feature map
that produced the logits. No edit to `dynamic_network_architectures`, no edit to `dwb.py`.

```
inputs  = [decoder_feat (C), softmax p, |grad p|, SDT of binarized pred (mm)]   # all detached
head    = Conv3d(C+3, 32, 3) -> GELU -> Conv3d(32, 32, 3) -> GELU -> Conv3d(32, 13, 1)
loss    = sum_c mean_{v in shell(c)} w(v) * CE(logits_v, bin(delta_v))  /  n_components
          shell(c) = predicted surface of component c, dilated +-2 voxels
          w(v)     = soft taper to 0 at the shell edge (no hard cutoff)
```

### Per-component normalisation across patches

A 150 mm mass does not fit in a patch, so it is sliced across many. Naively each slice looks like its
own component and the mass is counted many times — reintroducing the domination D3b exists to prevent.

Fix: the P0 cache stores `shell_voxels[component_id]`, counted once over the whole scan. Each voxel is
weighted `1 / shell_voxels[its component]`. The arithmetic then yields exactly one unit of loss per
lesion however the lesion was cut up, and no patch needs to contain a whole component. Sampling stays
plain surface-centric.

Remaining, unfixable in code: touching lesions merge into one component, so a radiologist's three
lesions can be the blob-finder's one. Use instance label IDs when the dataset has them — the branch
already exists in `centroids_from_seg` — otherwise state it as a limitation.

Ablation A1 drops `decoder_feat` — geometry only. **If A1 matches the full head, the project is a
ruler and stops.** This is the single most important experiment in the plan.

Do **not** add a distance-to-prompt channel. Prompt position is fine (a human click gives it too),
but it invites a leakage argument for zero gain.

## Phases and gates

| Phase | Work | Gate to pass |
|---|---|---|
| P0 | Run frozen ckpt over fold-0 val **and ~200 train-split scans**. Cache logits + δ + band mask. Emit **error-budget table**: share of total error from ≤1-voxel jitter / >4 mm boundary failure / missed / spurious components. Plus per-lesion Dice histogram, and train-vs-val δ distributions. | If ≤1-voxel jitter dominates, δ-regression predicts annotation noise → redesign before P2 |
| P1 | Baselines on the same cache: entropy, `1-p_max`, `abs(grad p)`, TTA variance (8 flips, already in `infer/tta.py`), geometry-only A1 | Establishes the bar. No head yet |
| P2 | Head + banded CE. Split held-out cases into head-train / cal / test | Beat every P1 baseline on predicted-vs-actual ASSD (Spearman + MAE mm) |
| P3 | Conformal calibration of aggregate scalars; coverage curves for volume and long-axis diameter | Nominal 90% coverage within ±3% |
| P4 | Report: reliability diagrams, AUSE, review-budget curve, lesion-type transfer (hold out lesion types) | — |

P0 and P1 share one cache and one inference sweep. Run them together.

## Sub-voxel precision

Voxel EDT, inner break `r = max(1.0, 2 * min(cm.spacing))` mm, resolved at startup. Marching-cubes
mesh distances are the sub-voxel alternative; deferred, revisit only if the centre bin turns out to
carry signal.

Nodule caveat: at <5 mm diameter the annotation itself is ±1 voxel from partial-volume effects, so the
irreducible noise floor spans the whole error range. Expect the head to look strong on large lesions
and near-useless on nodules. That is why D3c stratifies — a pooled number would flatter it on one end
and libel it on the other.

## Files

| Path | LOC budget | Contents |
|---|---|---|
| `nanounet/confid/target.py` | ~120 | `delta_field()`, `band_mask()`, `bin_delta()` |
| `nanounet/confid/head.py` | ~110 | `ConfHead`, decoder tap hook, banded CE |
| `nanounet/confid/module.py` | ~150 | `ConfidLM` — frozen backbone, stage-2 fit |
| `nanounet/confid/readout.py` | ~130 | bins → `S`, ASSD/NSD/volume, conformal calibration |
| `nanounet/cli/confid_cache.py` | ~150 | P0/P1: inference sweep, δ cache, error-budget + baseline tables |
| `nanounet/cli/confid_train.py` | ~120 | P2 |
| `nanounet/cli/confid_eval.py` | ~150 | P3/P4 metrics + figures |

`confid/` stays at 4 files. All <200 LOC (R1). No new folder outside these.

## Commands

```bash
nanounet_confid_cache -d 501 --fold 0 --ckpt runs/<run>/last.ckpt --out cache/epcm_501
```

```bash
nanounet_confid_train -d 501 --cache cache/epcm_501 --epochs 100
```

```bash
nanounet_confid_eval -d 501 --cache cache/epcm_501 --head runs/epcm/last.ckpt
```

## Risks

| Risk | Detection | Response |
|---|---|---|
| Head is a ruler | A1 geometry-only matches it | Stop. Publish the negative result or pivot to the PAE analogue |
| Annotation noise dominates | P0 error budget: jitter ≤1 voxel is most of the error | Move to lesion-level detection risk; δ-regression is the wrong primitive |
| Prompted errors too small to predict | P0 per-lesion Dice histogram is a spike near 1.0 | Weaken prompts (offset centroids) to widen the error distribution |
| Type-B presence errors dominate | P0 budget: missed/spurious components carry most of the error | Add the coarse presence output. Head is written to make this additive (see "Why not a wide band") |
| Objective captured by large lesions | Per-size-class δ MAE diverges; nodule class flat | D3b per-component normalisation. Verify by comparing micro vs macro loss curves |
| ConfidNet prior art | — | Differentiator is signed metric displacement + derived clinical scalars, not "auxiliary confidence head". Say so in the intro |
| Overconfidence from contaminated training | Head-train cases leak into segmenter train | D5. Assert split disjointness at startup (R15) |

## Deferred

- **PAE analogue.** "Given one boundary point is correct, how wrong is the rest?" Splits error into
  localization vs shape — a lesion shifted 2 mm keeps its volume, a ragged one does not. Most
  AlphaFold-faithful idea here, and unclaimed. Not v1.
- **Multi-reader validation.** Predicted spread should coincide with inter-observer disagreement.
  Strongest validation available; needs data we do not have. D6 keeps it possible later.
- **Annotation-quality filtering.** AlphaFold zeroes the confidence loss outside a crystal-resolution
  window (`filter_by_resolution`) so pLDDT means model error, not experimental noise. Same move
  applies once per-case annotation quality is known.
- Unprompted head, OOD head, epistemic head. Out of scope by decision.

## Sources

pLDDT / PAE head structure, `stop_gradient` on target, fine-tuning-only confidence heads,
`bin_index = floor(lddt * num_bins)`:
[Jumper et al., Nature 2021](https://www.nature.com/articles/s41586-021-03819-2) ·
[alphafold/model/modules.py](https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/modules.py) ·
Apache-2.0
