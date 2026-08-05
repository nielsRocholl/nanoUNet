# Handoff — Prompt-Sensitivity Sweep (Tier-0 diagnostic)

**Audience:** a coding agent running on the compute cluster. You do **not** have access to the
chat that produced this doc. Everything you need is here. Read it fully before writing code.

**Job in one sentence:** measure, and *decompose*, how much nanoUNet's follow-up lesion
segmentation changes when we move the prompt click around a single lesion — splitting that
variance into a "field-of-view" component and a "prompt-channel" component, and measuring how
much accuracy is recoverable by ensembling clicks.

**This is a pure evaluation/analysis task. Do NOT train anything. Do NOT modify model or
training code.** You only run inference and compute statistics.

---

## 0. Inputs the human will give you

The human will hand you two paths. Treat them as variables throughout:

- `CKPT` — path to the best nanoUNet checkpoint (`.ckpt`).
- `DATA_ROOT` — root of the longitudinal-registered follow-up dataset.

From the reference figure pipeline, `DATA_ROOT` is expected to contain (confirm the exact
layout on the cluster — names may differ slightly):

| Path under `DATA_ROOT` | Role |
|---|---|
| `preds_single_timepoint/<case>.nii.gz` | existing binary FU predictions (reference only; you generate your own) |
| `targetsTrFU/<case>.nii.gz` | **instance-labelled** GT — voxel value = `lesion_id` |
| `meta/<patient>.csv` | `lesion_id → lesion_type`, plus `img_id_fu`, `cog_fu` (lesion selection) |
| `lesions/<case>.json` | per-case propagated prompts; field `bl_click` = baseline centroid mapped into the FU frame by uniGradICON (the "automatic" click actually used in deployment) |

`<case>` = `<patient>_<img_id>`. **Excluded cases (drop them):** `92c8c96e4c_01`,
`2b24d49505_00`. Reference result set is **53 scans / 673 FU lesions**.

If any of these files/fields are missing or named differently, **stop and report** what you
found instead of guessing a substitute.

---

## 1. Background — why this task exists

nanoUNet is a promptable metastatic-lesion segmenter (nnU-Net-style core). Deployment on the
**Longitudinal-CT** melanoma dataset: the center-of-gravity of each baseline lesion is
propagated into the follow-up scan, and that point prompts the model on the FU timepoint.

Observed problem: **detection is excellent, delineation is mediocre.** On the FU set,
overall per-lesion mean Dice ≈ 65, detection rate ≈ 93%. Using the identity
`mean per-lesion Dice = detection_rate × Dice-on-found`, the **Dice-on-found ≈ 70** — i.e.
once a lesion is found, the boundary is only ~70–74 Dice.

Critical clue: **the segmentation is unstable to prompt placement.** For a single lesion,
moving the click (center vs. border vs. just-outside) produces visibly different masks and a
Dice range of ~73–78. A good delineator should be *prompt-invariant* — the mask is a property
of the image and of *which* lesion, not of *where inside it* you clicked. That instability is
the target of this diagnostic.

### Two problems hide behind one symptom
1. **Variance** — same lesion, different click → different mask.
2. **Mean/ceiling** — even the best contour is only ~75, boundaries systematically fuzzy.

These need different fixes. This sweep exists to quantify (1), separate it from (2), and tell
the next session which fix to build.

### 1.1 Two inference geometries — READ THIS CAREFULLY

There are **two different ways the model is run**, and they have different patch geometry. The
sweep must treat them as distinct, because the fix depends on which one carries the variance.

1. **Interactive single-centered-patch mode.** One click → one patch **centered on that click**
   (`s = p - ps//2`, `roi_slices.py`). Moving the click moves the field of view. **The
   prompt-sensitivity screenshots that motivated this task came from this mode.**
2. **Batch / grid deployment mode (how scans are actually processed).** Inference builds a
   **grid of patches that covers all prompt points** and tiles the region. The patch a lesion
   lands in is set by the **grid tiling**, largely **not** by the exact click position — so
   moving a click mostly changes the *prompt channel*, not the field of view (though it can
   change which grid cell "owns" the point, and a lesion can sit at a patch edge with truncated
   context). **The ~70 Dice ceiling in the figure was produced in THIS mode.**

Consequence: the "patch follows the click" confound is an **interactive-mode** effect. It may be
strong in the screenshots yet weak or absent in deployment. **Do not assume the interactive
instability explains the deployment ceiling.** Measure prompt sensitivity in *both* geometries
and compare — that comparison is now a primary output of this sweep (Section 6).

---

## 2. Confirmed code facts (already verified — use as anchors)

These were verified by reading the repo. Use them; re-confirm line numbers with `graphify`
before relying on exact lines (code may have moved).

| Mechanism | Anchor | Actual logic |
|---|---|---|
| Prompt encoding | `nanounet/prompt/encoding.py` (~L39) | click → **hard ball** (`ball(radius_vox)`) or EDT; radius = `cfg.prompt.point_radius_vox`; **no Gaussian sigma** — sharp positional marker |
| Patch extraction | `nanounet/infer/roi_slices.py` (~L23) | `s = p - ps // 2` → **patch is CENTERED on the click**. Moving the click **moves the field of view**. This is the confound we are decomposing. |
| Longi click sampler (train) | `nanounet/data/sampling.py` (~L44–45) | uniform random pick from a precomputed `fu_clicks_zyx` list (not mask-derived at train time) |
| Input channels (longi) | `nanounet/data/sampling_longi.py` (~L66) | 6 channels: `[FU_CT, FU_hm+, FU_hm-, BL_CT, BL_hm+, BL_hm-]`. **No baseline mask channel.** |
| Loss | `nanounet/model/losses.py` (~L47–68) | `DC_and_CE_loss` (Soft Dice + CE). **No boundary / Hausdorff / clDice term.** |
| Authoritative evaluator | `eval/eval_longi_fu_dice.py` | The lab's reference FU-Dice evaluator. **Reuse its data loading, click reading, coordinate handling, and Dice definition** so your numbers are comparable to the published figure. NOTE: it may be uncommitted/deleted in some working trees — if absent, restore it from git history (`git log --all -- eval/eval_longi_fu_dice.py`, then `git checkout <sha> -- eval/eval_longi_fu_dice.py`) before proceeding. |

**A `graphify` knowledge graph exists at `graphify-out/graph.json`.** For every "where/how does
X work" question, run `graphify query "<question>"` (or `graphify explain "<concept>"`,
`graphify path "<A>" "<B>"`) **before** grepping or reading source. It returns a scoped
subgraph and is much cheaper than raw search. After you write your script, run
`graphify update .` to keep the graph current.

**Follow the `nanochat-style` skill** for any code you write (files < 200 LOC, no
`utils/`/ABCs/factories, rich-CLI output, actionable error messages, zero GPU starvation).
Invoke the skill and obey it.

---

## 3. Before you write anything — locate the inference path

Use `graphify` to answer these, then confirm by reading the specific files:

1. **How does one run single-patch inference from a checkpoint + a click?** Find the public
   inference entry point. There is an embedded infer API (commit "feat(infer): embed API for
   Radiom — points_zyx_unpadded, on_forward"). Search: `points_zyx`, `points_zyx_unpadded`,
   `on_forward`, `infer`, `predict`.
2. **What is the model's input-channel count for `CKPT`?** (single-timepoint ≈ 3 ch:
   `CT, hm+, hm-`; longi = 6 ch as above.) Read it from the checkpoint/config — **do not
   assume**. Your input-building code must match whatever `CKPT` expects.
3. **What coordinate frame do clicks live in?** (`points_zyx_unpadded` implies pre-pad,
   resampled voxel coords.) The evaluator already handles click → model-space correctly.
   **Mirror it exactly.** Do not re-derive resampling/padding yourself.
4. **What is the lowest-level forward** that accepts an already-built input tensor and returns
   logits/probabilities, bypassing `roi_slices` patch extraction? You need this for Condition B
   (below). Search: `forward`, `sliding_window`, `predict_logits`.
5. **How is the BATCH / GRID inference path built?** This is the deployment path that produced
   the figure's numbers. Find how patches are laid out over the set of prompt points: is it
   one centered patch per point, a regular sliding-window tiling over the points' bounding
   region, or something else? How are overlapping-patch predictions merged (average / Gaussian-
   weighted / max)? Does moving one point change the tiling for its neighborhood? Search:
   `grid`, `tile`, `sliding_window`, `batch`, `aggregate`, `gaussian`, the batch/full-scan
   inference entry point. You need this for **Condition A-grid** (Section 4.3).

If the public infer path *couples* patch extraction and prompt placement so tightly that you
cannot set them independently, drop to that lowest-level forward and build the input tensor
manually (Section 4, Condition B). Report which path you used.

---

## 4. The experiment

Work on the FU validation lesions (the 673-lesion set, minus excluded cases). **Reuse the
evaluator's lesion selection** (`load_lesions_for_stem`-style: rows in `meta/<patient>.csv`
whose `img_id_fu` matches the case and whose `cog_fu` is non-empty). Attach `lesion_type` from
the same row.

### 4.1 Subset & determinism
- **Fixed seed = 1234** everywhere (numpy + torch).
- Start with a **stratified subset of ~150 lesions** spanning types, deliberately
  over-sampling the hard buckets (**Skeleton, Soft tissue / Skin, Others**) and including a
  clean contrast set (**Lung, Liver**). Record exactly which lesion_ids you used.
- Skip lesions smaller than **27 voxels** OR whose max interior EDT < **2 voxels** (border /
  outside clicks are undefined). Log skipped lesions separately with reason — do not silently
  drop.
- Make the subset size a CLI flag so a full 673-lesion run is a one-liner later.

### 4.2 Per-lesion click grid (deterministic)
For each selected lesion, build its binary mask `M` in the **model input voxel frame** (get
this frame from the evaluator's pipeline — `targetsTrFU == lesion_id`, resampled the same way
the evaluator/infer path resamples). Then define clicks (all in zyx voxel coords):

1. **`auto`** — the deployed click: `bl_click` from `lesions/<case>.json` for this lesion, if
   present. This is the reference/anchor click. If absent, fall back to `deep`.
2. **`deep`** — voxel with **max EDT** of `M` (distance transform to background); the most
   interior point.
3. **`border`** — voxels ~1 voxel inside the surface (EDT ∈ [1, 2]). Take **6** of them, spread
   by principal directions: for each of ±z, ±y, ±x, pick the eligible voxel farthest from the
   centroid along that axis.
4. **`outside`** — dilate `M` by 2 voxels, take the shell `(dilate(M,2) \ M)`, sample points
   ~1–2 voxels outside. Take **6**, spread by the same principal-direction rule.
   **Exclude any `outside` click that lands inside another lesion's GT** (would confound); log
   how many were dropped.

Result: ~14 clicks/lesion (`auto`, `deep`, 6×`border`, 6×`outside`). Store each click's
position, category, and EDT value.

### 4.3 The four conditions

**Condition A-int — interactive single-centered-patch (matches the screenshots).**
For every click, run the **interactive single-patch path** (patch centered on click,
`s = p - ps//2`; prompt ball at patch center). Both FOV and prompt-channel vary together. Save
the predicted binary mask (cropped consistently for scoring) and the probability map.

**Condition A-grid — batch/grid deployment path (matches the figure numbers).**
For every click, run the **real batch/grid inference path** (Section 3.5): the grid is built
over the point(s) and tiled; the lesion's patch is set by the tiling, not by the click. Both
FOV and prompt-channel vary, but FOV varies only as much as the *grid* lets it. This is the
geometry the deployment ceiling was measured in. Save mask + probability map.
- Run this **per lesion with a single click at a time** (one point → its grid) so a click's
  effect is isolated, mirroring how a propagated per-lesion click is used. If the deployment
  path only accepts all points at once, report that and instead pass one point per run.

**Condition B — "patch frozen" (isolates the prompt channel).**
Fix the patch to a single anchor = the `auto` click (or `deep` if no `auto`) for **all** clicks
of this lesion. FOV is identical across clicks; only the prompt heatmap moves.
**Cleanest implementation, no channel guesswork:**
1. Build the reference input tensor once, via the normal infer path, for the anchor click.
   This gives you the exact multi-channel patch (CT + baseline channels + prompt channels)
   the model expects.
2. For each other click, **clone that reference tensor and overwrite ONLY the positive-prompt
   heatmap channel** with a ball (radius = `cfg.prompt.point_radius_vox`, same encoding as
   `nanounet/prompt/encoding.py`) centered at the click's position *expressed in patch-local
   coords*. Leave the CT patch, the negative-prompt channel, and any baseline channels byte-for-
   byte identical.
3. Forward the cloned tensor through the lowest-level forward (Section 3.4).
This guarantees the *only* thing that changes between clicks is the prompt-channel position.
Clicks whose patch-local coordinate falls outside the fixed patch are invalid for B — log and
skip them for this condition only.

**Condition C — prompt-ensemble (recoverable headroom).**
Reuse **Condition A-grid** outputs (the deployment geometry). For each lesion, take all
**interior** clicks (`auto`, `deep`, `border`) — exclude `outside` — average their
**probability maps**, threshold at 0.5, and score that ensembled mask. (Also compute a
majority-vote variant for robustness.) `outside` clicks are excluded from the ensemble because a
real deployment never intentionally clicks outside. If cheap, also produce the A-int ensemble for
comparison.

### 4.4 Metrics (per click, per lesion, per condition)
Score inside the evaluator's per-lesion window (GT bbox + 10-voxel margin, excluding other GT
lesions' voxels from the prediction — **copy this exactly from the evaluator**).

Per click:
- **Dice** vs GT.
- **Surface distance**: HD95 and ASSD vs GT (via `scipy.ndimage.distance_transform_edt` on
  both masks; reuse an evaluator helper if one exists).

Per lesion, per condition (**A-int, A-grid, and B — all three separately**):
- **Dice std** and **Dice range** (max − min) across that lesion's clicks → the instability.
- **Mean pairwise prediction-Dice**: mean over all click-pairs of `Dice(pred_i, pred_j)`
  (prediction-to-prediction, **not** to GT). This measures raw instability *independent of GT
  quality* — high value = stable, low = unstable.
- **Best-click Dice** and **worst-click Dice** vs GT (the oracle spread = the opportunity).

Per lesion, Condition C:
- **Ensemble Dice** vs GT, and **ensemble gain** = ensembleDice − mean(interior single-click
  Dice).

### 4.5 Aggregation
Report every metric **overall** and **by `lesion_type`** (pooled over lesions of a type, matching
the figure's aggregation). Produce:
- A summary markdown/CSV table.
- The three headline numbers below (Section 6).

### 4.6 Qualitative output
For ~10 lesions (mix of high-variance and low-variance, include a couple of hard-bucket ones),
save an overlay PNG per lesion: the GT contour (red) plus every prediction contour on the central
slice — same style as the screenshots that motivated this. Make **two panels per lesion**:
Condition **A-int** (reproduces the motivating screenshots) and Condition **A-grid** (deployment).
Side-by-side, they show whether deployment is visibly more stable than interactive. For eyeballing.

---

## 5. Deliverables

Write to `RESULTS_DIR` (make it a CLI flag; default `eval/results/prompt_sensitivity/`):

1. `records.json` — one record per (lesion, click, condition) with: patient, case, lesion_id,
   lesion_type, click category, click zyx, EDT, Dice, HD95, ASSD, condition, and paths to saved
   masks (or run-length/compact form — do not bloat).
2. `summary.md` and `summary.csv` — the aggregated tables (overall + by type), for A, B, C.
3. `overlays/*.png` — the qualitative overlays.
4. `RUN_NOTES.md` — what `CKPT`/`DATA_ROOT` you used, channel count detected, which infer path
   you used, seed, subset lesion_ids, counts of skipped/excluded lesions and dropped `outside`
   clicks, wall-clock, and **the interpretation** (Section 6 filled in).

Keep the driver script under `eval/` (confirm the right location via `graphify`), one file,
< 200 LOC, rich-CLI progress, actionable errors (nanochat-style). Factor heavy helpers only if a
single file would exceed the limit.

### Compute budget
~14 clicks × 3 inference conditions (A-int, A-grid, B) × 150 lesions ≈ 6,300 runs. A-int and B
are single patch forwards (cheap); A-grid is a small multi-patch grid per lesion (a few patches),
so budget it higher. Condition C is free (reuses A-grid). Still an hour or two on one GPU. If it's
slow, you are almost certainly re-loading the model or re-resampling per click — cache the model
and the per-lesion resampled frame.

---

## 6. How to read the results (the decision this sweep exists to make)

Fill these into `RUN_NOTES.md`. Let `σ_Aint`, `σ_Agrid`, `σ_B` = mean per-lesion Dice-std for
conditions A-int, A-grid, B; `P_*` = mean pairwise prediction-Dice per condition; `ΔE` = mean
ensemble gain.

**First, the geometry question (the whole reason for the update):**

| Observation | Interpretation | Points next session at |
|---|---|---|
| `σ_Agrid ≪ σ_Aint` (grid/deployment is far more stable than interactive) | The prompt instability is largely an **interactive-mode artifact**; deployment's ~70 ceiling is **not** a variance problem. | Interactive UX fix (#1) is cosmetic for deployment. Focus deployment gains on **#7 loss + label ceiling**. |
| `σ_Agrid ≈ σ_Aint` (grid is also unstable) | Prompt variance is **real in deployment**. | Proceed to the FOV-vs-channel split below and attack it. |

**Then, split the variance (use A-int and B, both patch-centered-ish vs frozen):**

| Observation | Interpretation | Points next session at |
|---|---|---|
| `σ_B ≪ σ_Aint` (freezing the patch removes most variance) | **Field-of-view dominates.** The click moving the camera is the problem, not the prompt channel. | **#1 Patch–prompt decouple** — a no-retraining inference fix. |
| `σ_B ≈ σ_Aint` (variance survives with the patch frozen) | **Prompt channel dominates.** The hard ball's position itself steers the boundary. | **#5 prompt-augmentation + #8 prompt-consistency training** (retrain). |
| `ΔE` large & positive (ensembling beats single click, in A-grid) | Variance is **recoverable** in deployment. | Ship **prompt-ensembling at inference now** (free); worth attacking variance. |
| `P_*` **high** but Dice-vs-GT **low** (predictions agree with each other, disagree with GT) | Model is **stable but systematically wrong** — not a variance problem. | **#7 boundary/clDice loss**, resolution, **label-ceiling check** — not invariance work. |
| `best-click − worst-click` large (in A-grid) | Big oracle headroom from prompt choice alone, in deployment | Reinforces attacking variance / ensembling. |

Most likely outcome (prior, not fact): interactive mode (A-int) shows a **large FOV component**
that **shrinks in grid mode** (A-grid), leaving deployment variance dominated by the
**prompt-channel** component (needs #5/#8) plus a **systematic boundary error** floor (needs #7 +
label-ceiling check). If A-grid turns out nearly as stable as A-int is unstable, the headline is
"deployment ceiling is a *mean* problem, not a *variance* problem" — which redirects the whole
next session. The data decides.

---

## 7. Next session (do NOT do these now — for planning only)

Depending on Section 6:

1. **If FOV dominates → #1 Patch–prompt decouple.** Two-stage inference: coarse-localize from
   the click → recenter the patch on the predicted lesion centroid (a stable anchor) → segment.
   The click then only feeds the prompt channel; the FOV no longer follows it. No retraining.
   **Deployment-relevant form:** the batch/grid path already fixes the tiling, so a lesion can
   sit at a patch edge with truncated context; add a **per-lesion recenter-and-refine pass**
   after grid detection (crop a patch centered on each detected lesion → re-infer), giving every
   lesion a full-context centered view regardless of grid phase. Only worth it if A-grid showed
   real variance / edge effects.
2. **If prompt channel dominates → #5 + #8.** (#5) During finetune, sample training clicks from
   across the GT mask (center, border, occasional just-outside) instead of the current
   uniform-from-precomputed-list. (#8) Add a **prompt-consistency** term: show the same lesion
   under two click positions and penalize the two masks for disagreeing (mean-teacher /
   segmentation-FixMatch style). This directly optimizes the pairwise prediction-Dice measured
   here.
3. **Regardless → #7 boundary/connectivity loss.** Add **clDice** (punishes dropping thin
   protrusions — the missed-prong failure) and a boundary/Hausdorff-DT term to the next
   finetune. `dc_ce` alone gives weak gradient exactly at the boundary. Independent of the
   variance work; bundle it into the next training run either way.
4. **Label ceiling (gate everything).** Double-annotate ~20 hard-bucket lesions (or obtain
   inter-rater agreement from the data owners) and compare to model Dice-on-found. If GT
   inter-rater Dice on Skeleton / Soft-tissue is ~0.75–0.80, then part of the ~70–74 ceiling is
   irreducible label noise, and effort should shift from the mean toward the variance.

**Explicitly rejected ideas (do not resurrect without new evidence):**
- **Baseline-mask channel as a shape prior** — killed by domain knowledge: melanoma lesions
  change too much between timepoints (grow / shrink / split / merge), so the registered baseline
  mask is an *unreliable, biased* prior exactly at the boundary. Detection is already solved, so
  it buys nothing.
- **RL refinement agent** — over-engineered for this symptom; Dice is already differentiable and
  3D credit assignment is sample-inefficient. Parked.
- **Bigger model** — won't fix an inference confound or label noise; dataloader is already the
  bottleneck.

---

## 8. Guardrails

- **No training, no model/loss/training-code edits.** Inference + analysis only.
- **Match the evaluator exactly** for lesion selection, coordinate frame, per-lesion window, and
  Dice definition — comparability with the published figure is the whole point.
- **Detect channel count from `CKPT`; never assume it.**
- Deterministic seed (1234). Exclude the two excluded cases and any `outside` click landing in
  another lesion.
- If any assumption in this doc contradicts what you find in the code or data (missing files,
  different channel layout, coupled infer API), **stop and report** rather than improvising a
  workaround. This is a measurement — a wrong workaround produces a confidently wrong decision.
- `graphify query` first; `graphify update .` after; nanochat-style throughout.
