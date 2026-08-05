# Handoff — Single-Stage Stratified Training Overhaul

**Audience:** a coding agent working on the GPU cluster. You do **not** have access to the chat
that produced this doc. Everything you need is here. Read it fully before writing code.

**Job in one sentence:** reduce the model's **variance under prompt placement** — the same lesion
segmented differently depending on where you click — while replacing the two-stage
(supervised → d013 finetune) recipe with a single stratified supervised stage.

**The primary problem, stated plainly:** for a fixed lesion, moving the click produces materially
different masks. Everything else in this doc is downstream of that. `val_prompt_agreement` ended
the 600-epoch run at **0.68** while `val_dice` was **0.78** — the model disagrees with itself more
than it disagrees with ground truth. See §1 for why that number needs re-measuring before it can be
trusted, and §Step 0 for the fix.

**You work under human supervision, step by step.** Seven steps (0–6). Do **one step at a time**,
show the diff, and wait for the human to approve before starting the next. Step 0 is a blocker:
two validation metrics are broken or confounded, so nothing measured before it is trustworthy.
Steps 1 and 2 are prerequisites for the rest. Step 6 is gated on evidence from Step 1.

---

## 0. Verified facts about the current codebase

Every claim in this section was read out of the source. Line numbers were accurate at the time of
writing; re-confirm before editing, but treat the *semantics* as established.

| Fact | Location |
|---|---|
| Segmentation targets are **binary** (foreground = 1), not instance-labelled | `nanounet/plan/labels.py:27` (`Labels`, single-task, scalar background) |
| Instance structure is derived **offline** by `cc3d.connected_components((s > 0))` and only the centroid/seed/volume survive into `*_centroids.json` | `nanounet/prompt/centroids.py:33` |
| Every annotated centroid that falls in the patch gets a positive click; the drop branch is dead because both configs set `click_modes.drop = 0.0` | `nanounet/data/sampling.py:47` |
| Clicks are displaced by a registration-error model (per-axis sigma ≈ `[5.95, 6.39, 5.93]` vox, `max_vox` 34) **before** being filtered into the patch, so a displaced click can leave the patch entirely | `nanounet/data/sampling.py:41-48`, `configs/default.json` |
| The target is the whole `seg_crop`, unmodified, and the **same** target tensor is reused for every prompt variant of a patch | `nanounet/data/sampling.py:145`, `nanounet/train/patch_render.py:98` |
| Training cases are drawn **uniformly** — this is the line stratified sampling must replace | `nanounet/train/patch_iterable.py:120` |
| `--only-prefix` is a **data filter**, not a layer freeze; it filters both `tr_keys` and `val_keys` | `nanounet/train/data_module.py:119` |
| Validation overrides sampling to `fg_patch_prob = 1 - no_lesion_frac` and `false_pos_probability = 1.0` (every lesion-free val patch gets a deliberately misleading decoy click) | `nanounet/train/data_module.py:75-80` |
| `no_lesion_frac` defaults to **0.3** even when absent from the JSON, so all configs validate on the same composition | `nanounet/config.py:155` |
| The validation set is **re-drawn randomly every epoch** (`val_iters` × batch random patches) — there is no fixed val set | `nanounet/train/data_module.py`, `PatchIterable` |
| `val_dice` is pooled pseudo-Dice over **all** rows; `val_dice_macro` is the mean per-row Dice over **foreground-bearing rows only** | `nanounet/model/dice_helpers.py` (`pooled_fg_dice`, and `da.mean()` at `lightning_module.py:167`) |
| Prompt ablation zeroes the prompt channels and re-runs a forward pass | `nanounet/train/lightning_module.py:139-141` |
| Checkpoint selection monitors `val_dice_macro` whenever `--init-weights` is set, otherwise `val_dice` | `nanounet/train/fit.py:189` |
| There is **no** `EarlyStopping`, **no** LR warmup, and **no** weight EMA anywhere in the repo | verified by grep |
| Optimizer defaults: SGD, `lr=0.01`, `momentum=0.99`, nesterov; `--lr` default `0.01` | `nanounet/train/lightning_module.py:187`, `nanounet/cli/train_parser.py:19` |

### The `stretched_tail_poly` schedule

`nanounet/model/lr_schedule.py:52`. With the values used in production (`--stretched-k 188
--stretched-ref 250 --stretched-exp 0.9`) over 600 epochs it is:

- epochs 0–187: poly decay against a **250**-epoch reference, ending at `0.285 × lr0`
- epochs 188–599: **linear** decay from `0.285 × lr0` down to `0.0069 × lr0`

This is important: it is *not* nnU-Net's poly-to-zero. There is no compressed final anneal, so a
late upturn in a metric under this schedule is real signal, not an annealing artifact.

---

## 1. Background — what was observed and what it means

### The runs

Two runs exist (see `scripts/` for the exact sbatch files):

- **Supervised**, `-d 999` merged pool (~7000 cases), 600 epochs × 1000 iters, `--batch-size 12`
  with `--prompts-per-patch 2` (so **6 distinct patches per step**), MAE-initialised, SGD @ 0.01,
  `stretched_tail_poly`, `--consistency-weight 0.02`.
- **Finetune**, same dataset id but `--only-prefix d013_` (~537 longitudinal-CT cases), 600 epochs
  × 1000 iters, AdamW @ `1e-5`, `wd 3e-5`, `--grad-clip 1.0`, plain `poly`, initialised from the
  supervised `last.ckpt`.

### Evidence that the model is underfit

**The primary evidence is the curve shape, and it needs no assumptions.** The `val_dice` curve is a
smooth concave rise with a **clearly positive slope at epoch 600 and no knee**. Fitting
`Dice ≈ a + b·log(epoch)` over epochs 200→600 gives `b ≈ 0.105` per nat. Naive extrapolation to
1200 epochs predicts ~0.85; realistically expect **+4 to +7 Dice points** from doubling the budget,
because log extrapolation over-promises near the annotation-noise ceiling. Nothing in the curve
indicates where that ceiling is.

Supporting, and independent of any comparison: with ~30M parameters against 7000 scans the model is
**underparameterised**, so the classic "stop when validation turns up" signal will never fire.
Absence of overfitting here is a diagnosis (compute-limited), not a reassurance.

### Compute budget — direction is certain, magnitude is NOT

Current run: 600 × 1000 × 6 = **3.6M patches** over ~7000 cases ≈ **514 patches per case**.

It is tempting to compare this against nnU-Net's fixed 1000 × 250 × 2 = 500k patches and conclude
the run is "N× under-exposed." **Do not quote such a ratio as fact.** The comparison requires
knowing the typical dataset size nnU-Net's 1000-epoch budget was tuned against, and we do not know
it. An earlier draft of this doc asserted "~6× under" by assuming ~150 cases; that assumption is
unverified and the figure has been removed.

What *is* solid:

- nnU-Net's budget is a **fixed patch count**, independent of dataset size (an "epoch" there is 250
  iterations of random patch sampling, not a pass over the data).
- Therefore per-case exposure falls as the dataset grows, and at ~7000 cases it is **substantially
  lower** than in the small-dataset regime the recipe was tuned for.
- Direction: certain. Magnitude: unknown.

Do not use this section to justify a specific epoch count. Use the curve-shape evidence above.

With ~30M parameters against 7000 scans you are in the underparameterised regime, so the classic
"stop when validation turns up" signal will never fire. Absence of overfitting here is a diagnosis
(compute-limited), not a reassurance.

### Finetune run — what went wrong

Read within-run trends only; the two runs validate on different case pools (full vs d013), so
absolute levels are not comparable.

| Metric | Trend over the finetune |
|---|---|
| `val_dice` | 0.70 → 0.61 — **declines monotonically** |
| `val_dice_macro` | 0.57 → 0.61, plateaus ~epoch 150 — **rises** |
| `val_dice_prompt_ablated` | ~0.55 → ~0.43 — declines |
| `val_dice_click_outside` | 0.55 → 0.61, plateaus ~epoch 200 |
| `val_prompt_agreement` | 0.52 → 0.72 — rises above the supervised run |
| `val_loss` | flat after ~epoch 200 while `val_dice` keeps degrading |

**The macro-up / pooled-down split is exactly decodable.** `val_dice_macro` averages per-row Dice
over foreground-bearing rows only. `val_dice` is `2ΣTP/(2ΣTP+ΣFP+ΣFN)` pooled over *all* rows,
including the 30% lesion-free rows that carry a forced decoy click and can therefore contribute
only FP. Macro up 4 points with pooled down 9 points means **per-lesion accuracy improved slightly
while false positives on lesion-free patches grew sharply.**

→ **First thing to do when you get access to the W&B runs: plot `val_fp`** (already logged,
`lightning_module.py:168`). It settles this directly.

This is not a sampling artifact: `configs/finetune_d013.json` *lowered* `fg_patch_prob` from 0.67
to 0.55 (more background in training), which should have suppressed FP. It went the other way.

Mechanism: 600k steps over 537 cases ≈ 1100 steps per case, on a cohort with a high lesion prior.
The model learned "click ⇒ lesion," which is nearly always true on d013 train and exactly wrong on
the forced-decoy val rows. Full network unfrozen, no replay, no anchor to base weights — textbook
catastrophic specialisation.

### A real bug found along the way

`nanounet/train/fit.py:189` — the finetune monitors `val_dice_macro`, the one metric that improves.
So `best-*.ckpt` from that run points at a late epoch where pooled Dice and prompt-ablated Dice are
at their worst. The justifying comment claims macro is chosen for "FP suppression," but macro
averages over `has_fg` rows only and **cannot see false positives at all** — every FP that matters
lives on the `~has_fg` rows. Fixed in Step 4.

### The task definition — read this before touching anything prompt-related

**The rule, stated by the human and authoritative for this project:**

> Segment a lesion **if and only if** it has a corresponding click. The click is the hint.
> Unannotated lesions can never be identified, and must never be segmented.

This collapses four cases into one uniform rule:

| Lesion in patch | Click on it | Correct target |
|---|---|---|
| annotated | yes | foreground |
| annotated | no | **background** |
| unannotated | no (by construction) | background |
| nothing there (decoy) | yes | background |

Two consequences that matter for every step below:

**You never need to identify unannotated lesions.** Under this rule they are handled by the same
"no click ⇒ background" clause as everything else. Any plan that requires detecting them is
misconceived — the human has confirmed it is impossible, and it is also unnecessary.

**The current target is not this rule.** The target is *all annotated lesions in the patch*
(`sampling.py:145`), regardless of which ones got clicks. Row 2 of the table is therefore almost
never trained: an annotated lesion goes unclicked only when registration displacement happens to
push its click out of the patch. **Selectivity — segment this lesion, not that one — is essentially
never trained.**

Note also that under the *current* target, rows 2 and 3 disagree while looking identical to the
click channel: an unannotated lesion is background and an annotated one is foreground, and no click
distinguishes them. The model must infer annotation status from something else. Under the human's
rule that ambiguity disappears entirely.

### The primary problem: variance under prompt placement

For a fixed lesion, moving the click changes the mask. The repo's other handoff
(`HANDOFF_prompt_sensitivity_sweep.md`) measured a Dice range of ~73–78 across click positions on a
single lesion, with detection ≈93% but Dice-on-found ≈70 — i.e. **delineation, not detection, is
the weak part, and it is prompt-sensitive.**

`--prompts-per-patch 2` and `--consistency-weight` exist to attack exactly this.

**Why the variance exists, mechanically.** Clicks are displaced ~6 vox/axis by the registration
error model and frequently land outside their lesion. Nothing forces the model to resolve a click
to a lesion *identity*, so it uses the click as a soft spatial prior — continuous evidence about
where mass belongs. Perturb the click, perturb the prior, get a different mask.

**Invariance and selectivity are the same fix, not competing goals.**

- *Invariance*: the output must not depend on **where in** the lesion you click.
- *Selectivity*: the output must depend on **which** lesion you click.

Both follow from one specification: **the click is an instance selector, not a spatial cue.**
Resolve click → lesion identity, and the mask becomes a function of the identity alone. This is why
Step 6 (instance-conditional targets) is the principal fix for variance and not merely a
selectivity feature: two clicks resolving to the same lesion get the *same target*, so invariance
is trained by supervision instead of coaxed by a penalty.

**Why the consistency term underdelivers today.** Both prompt variants already share a target
(`patch_render.py:98`), so the segmentation loss is already pulling them together; the consistency
term is a weaker second-order nudge on top, and it can be satisfied by ignoring the click entirely.
It is not wrong — it is underpowered, because the target never encodes what "same lesion" means.

### The leading hypothesis for `val_dice_prompt_ablated = 0.72`

Two strategies both score well on the current validation distribution:

- **(a)** read the click and segment what it points at — the intended behaviour;
- **(b)** infer the annotation protocol from image context (modality, organ, field of view,
  cohort-specific appearance) and segment everything matching it — no click needed.

Strategy (b) plausibly accounts for much of the 0.72 recovered with the prompt channels zeroed.
This predicts a concrete deployment failure: in longitudinal follow-up, baseline lesions get
propagated clicks but a **new** lesion at follow-up has none and must not be segmented. A model
running (b) segments it anyway. Same when a radiologist clicks three of eight lesions.

**Do not read the 0.06 gap as "the prompt does almost nothing."** Six Dice points is a large effect
in segmentation, and the gap does not decompose — it could be boundary refinement, extra detection,
or selection, and only the third is what the rule in §1 needs. Two effects also push it in opposite
directions: zeroing the prompt channels is an out-of-distribution perturbation (inflates the gap),
while the val set barely contains cases where selection matters (compresses it).

**This is a hypothesis, not a finding. Step 1 tests it. Do not act on it before then.**

#### The decisive test (buildable, needs nothing we lack)

Take validation patches containing **two or more annotated lesions**. Click a strict subset. Score
the prediction twice — against the *clicked-subset* target and against the *all-lesion* target.
Scoring higher against all-lesions means the model is running (b) and ignoring the selection.

This needs instance separation of **annotated** lesions only, which `cc3d` already computes at
preprocessing (`centroids.py:33`). Because the val set becomes a fixed offline manifest in Step 1,
those instance maps are precomputed there — **zero cost on the training hot path.**

#### Two prior hypotheses, both rejected — do not revive them

1. **"The prompt is redundant with the target."** Rejected. Clicks are displaced by the
   registration-error model (sigma ≈ 6 vox/axis) *before* patch filtering, so they are noisy
   pointers rather than a function of the target; and redundant-in-the-Bayes-sense would not imply
   useless as conditioning for a finite-capacity network anyway.
2. **"The gap is small because the model is damaged."** Rejected. A large `val_prompt_gap` is the
   design *goal* under the rule above. Note this contradicts the comment at
   `lightning_module.py:136-138`, which frames a *closing* gap as the failure mode. That comment is
   backwards for this project; rewrite it once Step 1 has evidence.

One measurement artifact to keep in mind when reading the gap: validation forces
`false_pos_probability = 1.0`, so every lesion-free val patch carries a misleading decoy click.
Ablation zeroes that decoy, handing the ablated model a **less adversarial** input than the prompted
one. Part of the 0.06 gap is this artifact, biasing it downward.

---

## 2. Constraints you must respect

- **`docs/../nanochat-style` conventions apply.** Files under `nanounet/` stay **under 200 LOC**.
  No `utils/`, no ABCs, no factory indirection. Read the `nanochat-style` skill if available.
- **Several target files are already at the ceiling:** `fit.py` (211), `lightning_module.py` (199),
  `data_module.py` (192), `patch_iterable.py` (188). **New logic goes into new small modules** that
  these files import. Do not grow them.
- **Never starve the GPU. This is a hard requirement, not a preference.**
  **Average GPU utilisation must stay above 95%.** The dataloader is the known bottleneck on this
  project — a previous full-volume EDT in the sampler dropped utilisation to ~60%, and the sbatch
  scripts pick `--dl-bucket xl` and `cpus-per-task=64` specifically to keep the H200 fed.
  - Measure utilisation before and after **every** step, including the metric and instrumentation
    steps. "It's only bookkeeping" is not an argument — measure it.
  - Anything that touches the **per-patch** path (sampling, augmentation, rendering, collate) is
    hot. Per-patch `cc3d`, EDT, connected components, or Python loops over voxels belong in
    **preprocessing or the offline val manifest**, never in the dataloader.
  - Validation metrics run per val batch and are on the same critical path. Keep them to tensor ops
    on data already in memory. Extra forward passes (the ablation and agreement draws already cost
    two) must be justified and counted.
  - If a step cannot hit >95%, stop and report the measurement rather than shipping it.
  - Use `epoch_wall_time_sec` (already logged, `lightning_module.py:160`) as the second signal
    alongside `nvidia-smi` utilisation. Same node, same bucket, same batch size, before and after.
  - Do **not** compensate for starvation by raising the batch size. That masks the problem instead
    of removing it, and the sbatch headers say so explicitly.

  **Live risk: on-demand `cc3d`.** Steps 1 and 6 both want connected components, and they differ
  sharply in cost. Step 1 is safe — its instance maps are built **offline into the val manifest**.
  **Step 6 is the dangerous one**: it needs instance identity for every *training* patch, on the hot
  path, every step. Options, best first — **all costs unmeasured, benchmark before choosing**:

  1. Persist the `cc3d` label map once at preprocessing next to `*_seg.b2nd` and crop it like the
     seg. Zero hot-path CPU, but it costs storage and staging bandwidth — the merged pool is already
     ~543 GB over a slow link, so this is not free either. Measure it.
  2. Precompute per-lesion **bounding boxes** into the existing `*_centroids.json` sidecars and
     resolve click → instance by bbox lookup. No label map, no hot-path CC.
  3. Run `cc3d` on the crop at sample time. **Last resort.** A patch-sized call is far cheaper than
     the full-volume EDT that caused the 60% incident, but "cheaper than the thing that broke it"
     is not evidence that it is cheap enough.

  Report measured numbers to the human before committing to an option. Do not assume option 3 is
  fine because the crop is small.
- Errors must be actionable: say what failed, what was expected, and the command that fixes it.
- Keep the rich-CLI output style used elsewhere in `nanounet/cli/`.
- **If a file or field this doc describes does not exist or is named differently on the cluster,
  stop and report what you actually found. Do not guess a substitute.**

---

## Step 0 — Fix two broken/confounded validation metrics

**Status: BLOCKER. Already implemented on the laptop — see `nanounet/train/patch_render.py`,
`nanounet/model/dice_helpers.py`, `nanounet/train/lightning_module.py` and
`tests/test_prompt_metrics.py`. Your job on the cluster is to re-measure, not to re-implement.**

Nothing measured before this step is trustworthy, including the 0.68 agreement number quoted above.

### 0a. `val_dice_click_outside` was wrong (real bug)

Validation forces `false_pos_probability = 1.0` (`data_module.py:78`), so **every** val patch gets a
decoy click appended to the positive list (`sampling.py:48`). `click_inside_flags` then
majority-voted over *all* positive clicks, decoy included:

```python
flags.append(1 if 2 * n_in > len(idx) else 0)
```

For a patch with L correctly-placed lesion clicks plus one decoy: `n_in = L`, `len = L+1`, so the
condition reduces to **`L > 1`**. Single-lesion patches were therefore *always* flagged "click
outside" even with a perfectly centred click. `val_dice_click_outside` was dominated by
single-lesion patches and `val_dice_click_inside` by multi-lesion ones — the metric was tracking
lesion count more than click placement.

**Fix applied:** the decoy count is carried through from sampling and excluded from the vote.

### 0b. `val_prompt_agreement` was confounded (not a coding bug)

Both prompt draws receive a decoy at **independently random locations** (the diagnostic variant
draws from `extra_rng`, `sampling.py:141`). So the two inputs differed by *both* per-lesion
displacement — the quantity of interest — and an entirely different spurious click. A low score
could not be attributed to either. Because `prompt_pair_dice` compares whole-patch masks, two
predictions agreeing perfectly on the lesion but differing on decoy false-positives scored low, and
the loss was blamed on prompt sensitivity.

**Fix applied:** `draw_false_pos` draws the decoy **once per patch** (`sampling.py`) and every
variant shares it, so the two draws differ only in per-lesion click placement. No new metric and no
extra forward pass — decoy robustness is already covered by `val_fp` on the lesion-free val rows,
which all carry a decoy.

**This changes training too, deliberately.** The consistency pair (`--prompts-per-patch 2`) now
also shares its decoy, so `loss_consistency` trains placement-invariance rather than being partly
spent on decoy-position differences. That matches the intent documented in the sbatch scripts
("same crop, same augmentation, same target; only the click differs"). Note the RNG stream moved,
so runs are **not** bit-reproducible against pre-fix seeds.

### 0c. Both metrics are patch-level; the question is lesion-level

`prompt_pair_dice` and the per-row Dice are computed over the whole patch, so a multi-lesion patch
collapses every lesion into one number. Neither can answer "how much does *this* lesion's mask move
when *its* click moves" — the quantity the consistency work exists to reduce.

Per-instance scoring needs `cc3d` resolution of each click to a lesion instance. **Deferred to
Step 1 on purpose**, because the fixed val manifest computes instance maps offline anyway, so it
costs nothing on the training hot path. Do not add per-patch `cc3d` to the dataloader.

### What to do on the cluster

1. Run `.venv/bin/python -m pytest tests/test_prompt_metrics.py -q` and confirm 12 pass. The two
   real-data tests self-skip when the laptop sample set is absent, which it will be on the cluster
   — the 10 synthetic tests still cover both fixes.
2. Re-run **inference only** with the existing 600-epoch checkpoint against the corrected metrics.
   No retraining.
3. Report corrected `val_prompt_agreement`, `val_dice_click_inside`, `val_dice_click_outside`
   beside the old values. **The corrected agreement number sets the priority for everything
   downstream** — if prompt variance is smaller than 0.68 suggested, say so before Step 6 is built.
4. Confirm GPU utilisation is unchanged (>95%). These fixes remove work rather than add it — the
   decoy is drawn once per patch instead of once per variant — so utilisation should not move.

### Known non-bug you will otherwise re-discover

The *plain centroid* of a lesion falls **outside its own lesion ~12% of the time** on concave
shapes; this is why `seed_zyx` (argmax-EDT, guaranteed interior) exists alongside `centroids_zyx`
(`centroids.py:1-7`). Such patches are legitimately scored "click outside" even with a perfectly
placed click. That is lesion shape, not a metric defect. `tests/test_prompt_metrics.py` clicks the
seed for exactly this reason — an earlier draft of the test clicked the centroid and failed on 1 of
5 real single-lesion cases.

---

## Step 1 — Fixed, stratified validation set

**Status: prerequisite. Do this after Step 0.**

### Goal

Replace the randomly re-drawn per-epoch validation set with a fixed, seeded, stratified one, and
report the prompt metrics *per stratum*.

### Why

Two independent problems:

1. **Noise.** At `--val-iters 50` with 6 patches per batch you get ~300 val patches per epoch,
   freshly sampled each time. Split that across cohorts and per-group curves become unreadable.
   A fixed set removes resampling variance from every curve at zero cost.
2. **Blindness.** The current val composition cannot see whether the prompt is doing its job,
   because the discriminating cases are a small, uncontrolled fraction of it (see §1).

### Design

Generate the val patch list **once**, offline, into a JSON manifest. Each entry fully determines a
patch: case id, bbox, and the click coordinates (post-displacement, pre-augmentation). At
validation the loader reads the manifest instead of sampling. No augmentation on val.

Strata to include, each with an explicit target count so the composition is known and stable:

| Stratum | Purpose |
|---|---|
| `all_clicked` | every annotated lesion in the patch is clicked — today's training condition, baseline |
| `subset_clicked` | **≥2 annotated lesions, a strict subset clicked** — the selectivity test (§1) |
| `none_clicked` | annotated lesions present, **no** clicks — the pure suppression test; correct output is empty |
| `lesion_free_decoy` | no lesion, forced decoy click — measures FP directly |
| `small_lesion` / `large_lesion` | by `volume_vox` from the centroid sidecars |
| `click_outside` | click displaced off its lesion — the deployment-realistic case |
| per cohort | one slice per dataset prefix (`d013_`, etc.) |

**Unannotated lesions do not appear in this table and must not.** They cannot be identified (human,
confirmed), and under the project's rule they need no special stratum — "no click ⇒ background"
covers them. An earlier draft of this doc proposed a `mixed_annotation` stratum; it was removed as
both impossible and unnecessary. Do not reintroduce it.

`subset_clicked` and `none_clicked` require per-lesion instance maps at val time. Compute them with
`cc3d` **in the offline manifest builder** and store the instance id per click, so the training hot
path is untouched. For these two strata the manifest must record **two** targets per patch — the
clicked-subset target and the all-lesion target — so the diagnostic in §1 can score against both.

### Files

- new `nanounet/cli/build_valset.py` — offline manifest builder, plus a console-script entry
- new `nanounet/data/valset.py` — manifest schema, load, and a deterministic patch iterator
- `nanounet/train/data_module.py` — use the manifest for `val_dataloader` when given one
- `nanounet/cli/train_parser.py` — `--val-manifest PATH`

### Acceptance criteria

- Two runs with the same manifest and the same checkpoint produce **identical** val metrics.
- Per-stratum `val_dice`, `val_dice_macro`, `val_fp`, `val_dice_prompt_ablated`, `val_prompt_gap`
  appear in W&B as `val/<stratum>/<metric>`.
- On `subset_clicked`, both `val_dice_vs_clicked_subset` and `val_dice_vs_all_lesions` are logged.
  **This pair is the headline result of Step 1** and decides whether Step 6 goes ahead:
  `vs_all_lesions` > `vs_clicked_subset` means the model ignores the selection (strategy (b)).
- On `none_clicked`, the predicted foreground fraction is logged. Under the project's rule the
  correct value is ~0.
- **Per-instance prompt agreement** is logged (deferred here from Step 0c): resolve each click to a
  `cc3d` lesion instance in the offline manifest, then score agreement restricted to that
  instance's neighbourhood. This is the direct measurement of the primary problem (§1) and the
  number Step 6 must move.
- The aggregate metrics keep their existing flat names so old runs stay comparable.
- Manifest generation is reproducible from a seed and records that seed inside the file.

### Gotcha

Keep the aggregate `val_dice` definition **byte-identical** to today's `pooled_fg_dice`. If you
change it, every historical curve becomes incomparable and the 600-epoch baseline above is lost.

---

## Step 2 — Stratified (per-cohort) metrics

**Status: prerequisite.**

### Goal

Bucket validation metrics by cohort so you can see which part of the data manifold is still
improving.

### Why

The aggregate `val_dice` still rising at epoch 600 may be broad progress, or it may be one slow
cohort climbing while the others plateaued hundreds of epochs earlier. These have opposite
implications for what to do next, and right now they are indistinguishable.

### Design

Thread a group label from `PatchIterable._producer` — which already has `cid` at
`patch_iterable.py:120` — through `build_patch` → the item dict → `collate_patches` → the val
batch, then bucket in `on_validation_epoch_end`.

Derive the group from the case-id prefix (`d013_` → `d013`). Keep it a plain string; do not build a
registry class.

`collate_patches` (`patch_render.py:86`) is the natural place to stack the labels, alongside the
existing `pair_id` and `click_inside` handling. It is a pure function — extend it the same way.

### Files

- `nanounet/train/patch_iterable.py` — emit the group in the item dict
- `nanounet/train/patch_render.py` — carry it through the collate
- new `nanounet/train/val_metrics.py` — bucketing and logging (keeps `lightning_module.py` under 200)
- `nanounet/train/lightning_module.py` — call into it

### Acceptance criteria

- W&B shows `val/<cohort>/val_dice`, `.../val_dice_macro`, `.../val_fp`, `.../val_prompt_gap`.
- Aggregate metrics are unchanged to within float noise.
- Training throughput is unchanged — this is bookkeeping, not compute.

---

## Step 3 — Cohort-weighted sampling

### Goal

Control the training mixture explicitly, so d013 (or any cohort) can be oversampled **within the
single supervised stage** instead of chased with a finetune.

### Why

The finetune failed by catastrophic specialisation (§1). Oversampling inside one stage gets the
same emphasis with none of the forgetting, because the other cohorts keep regularising the shared
trunk throughout. d013 is ~537 of ~7000 cases (~8%); a reasonable first target is 20–30%.

### Design

Replace the uniform draw at `patch_iterable.py:120`:

```python
cid = self.keys[int(rng.integers(0, len(self.keys)))]
```

with a two-level draw: sample a cohort from configured weights, then a case uniformly within it.

Config lives in the ROI JSON under a new `sampling.cohorts` block, e.g.:

```json
"sampling": {
  "cohorts": { "d013_": 0.25 }
}
```

Semantics: named prefixes take the stated probability; the remaining mass is spread over all other
cases in proportion to their counts. Absent or empty block ⇒ current uniform behaviour, exactly.

**This composes with, and does not replace, the existing lesion-type stratification.** The
`*_weights.json` sidecars feed patch-*location* weights at `sampling.py:120`. Cohort weights pick
*which case*; lesion weights pick *where inside it*. Two independent knobs.

`--only-prefix` becomes the degenerate case (one cohort at weight 1.0). Keep the flag working for
backward compatibility.

### Files

- `nanounet/config.py` — `CohortConfig` on `SamplingConfig`, defaulting to empty
- new `nanounet/data/cohorts.py` — group the key list, build the alias/CDF table once at init
- `nanounet/train/patch_iterable.py` — use it in `_producer`

### Acceptance criteria

- With no `cohorts` block, sampled case frequencies match uniform within Monte-Carlo error.
- With `{"d013_": 0.25}`, the observed d013 share over ≥50k draws is 0.25 ± 0.01.
- Sampler setup is O(#cases) **once**, not per draw. No per-draw prefix scan — precompute the
  index groups at construction.
- No measurable change in dataloader throughput (benchmark it).

---

## Step 4 — Warmup, EMA, checkpoint-monitor fix

Three small independent changes. They can share one review.

### 4a. LR warmup

No warmup exists. Today step 1 runs at full LR with momentum 0.99 on an MAE-initialised network.
Add linear warmup over the first N epochs (default 5–10) applied to both schedules in
`nanounet/model/lr_schedule.py`. Expose `--warmup-epochs`, default `0` so existing runs are
bit-for-bit reproducible.

### 4b. Weight EMA

No EMA exists. In a noise-dominated regime it typically buys the equivalent of a few hundred epochs
for almost nothing. Add an EMA callback (decay ~0.999, configurable), keep EMA weights in the
checkpoint alongside the raw weights, and log `val_dice_ema` next to `val_dice` so the human can
compare before trusting it. Put it in a new `nanounet/train/ema.py`.

### 4c. Checkpoint monitor

`fit.py:189` currently selects on `val_dice_macro` whenever `--init-weights` is set — a metric
structurally blind to false positives (§1). Replace with an explicit `--monitor` flag defaulting to
`val_dice` in **all** cases, and delete the `init_weights` special case and its incorrect comment.

### Acceptance criteria

- `--warmup-epochs 0` reproduces current LR curves exactly.
- EMA adds < 2% step time.
- Existing checkpoints still load.

---

## Step 5 — LR probe, then the long run

### 5a. Probe

The current `lr=0.01` / `momentum=0.99` come from nnU-Net's batch-2 regime. At 6 distinct patches
per step the gradient variance is lower than that momentum was chosen to smooth (0.99 gives an
effective averaging window of ~100 steps ≈ 600 patches — over-damped).

Do not settle this by argument. Run **three 60-epoch probes** at `lr` ∈ {0.005, 0.01, 0.03},
everything else fixed, and compare `val_dice` at epoch 60 on the fixed manifest from Step 1. Then
optionally one probe at `momentum=0.97` on the winning LR.

Cost is a small fraction of one full run. Report a table, not a recommendation.

### 5b. Long run

Once the LR is chosen, launch the single stratified supervised stage:

- **1200–1800 epochs** (from ~600 today; the log fit predicts +4 to +7 Dice)
- cohort weights from Step 3
- fixed val manifest from Step 1, per-cohort metrics from Step 2
- warmup + EMA from Step 4
- `--monitor val_dice`
- **no finetune stage**

Keep `stretched_tail_poly`, but re-tune `--stretched-k` / `--stretched-ref` for the new horizon —
they are absolute epoch counts, not fractions, so reusing `188 / 250` at 1800 epochs would put
almost the entire run in the linear tail.

**Optional, discuss with the human first:** a warmup-stable-decay schedule would let you branch a
checkpoint, anneal it over ~5% of budget, evaluate, and *then* decide whether to continue — instead
of committing to the horizon up front. That is the right tool for "the curve is still rising and I
do not know where to stop." It is a small addition to `lr_schedule.py`.

### On early stopping

Do **not** add `EarlyStopping` to the supervised run. Its curve never turns over, so it would never
fire; the real constraint is budget, not overfitting. (It *would* have been the right tool for the
finetune, which turned over around epoch 20 — but that stage is being removed.)

---

## Step 6 — Instance-conditional targets (CONDITIONAL — do not start unsolicited)

**Gate — either condition triggers this step:**

1. **Variance (primary).** Per-instance prompt agreement from Step 1 is low after the corrected
   measurement in Step 0. This is the main reason to build it.
2. **Selectivity (confirmatory).** `val_dice_vs_all_lesions` exceeds `val_dice_vs_clicked_subset`
   on the `subset_clicked` stratum — the model is ignoring the selection.

Both are symptoms of the same cause (the click is read as a spatial cue, not an instance selector,
§1), and instance-conditional targets fix both. If agreement is already high *and* the model scores
higher against the clicked subset, report that and stop — the mechanism is already working.

### What it is

Make the target click-conditional: foreground only for lesion instances that received a click,
background for instances that did not. This makes "no click ⇒ no segmentation" an explicitly
trained behaviour rather than a hoped-for one.

Framing note: under the project's rule (§1) this is not an enhancement — it **is** the correct
target, and the current all-annotated-lesions target is an approximation that happens to be nearly
exact on the training distribution while omitting the one behaviour that matters at deployment. The
gate exists because the source discussion twice reasoned ahead of the data and was wrong, not
because the direction is in doubt.

**This is also the principal fix for the variance problem (§1).** Two clicks resolving to the same
lesion produce the *same* target, so placement-invariance becomes supervised rather than penalised.
That is a stronger signal than the consistency term can provide on its own.

### Expect the loss curves to get worse before they get better

The same lesion will be foreground in one patch and background in another. This is **not** label
noise — the input differs (the click channel), so the task stays well-posed — but it removes the
shortcut the model is currently using, and training loss will rise relative to the old objective.
Do not read that as a regression. Judge only on the Step 1 strata, especially `subset_clicked` and
`none_clicked`.

### Why it is hard here

The seg is **binary**, so there is no way at training time to say which voxels belong to the
clicked lesion. `cc3d` already runs at preprocessing (`centroids.py:33`) but only the centroid,
seed voxel, and volume survive.

**This is the step most likely to break GPU utilisation, and the cost has not been measured.**
Unlike Step 1 — whose instance maps are built offline into the val manifest — this needs instance
identity for every *training* patch, on the hot path, every step. Read the "Live risk: on-demand
`cc3d`" block in §2 and follow it: three options, ranked, all unmeasured, benchmark before choosing,
report numbers to the human before committing. Do not default to per-patch `cc3d` on the assumption
that a crop is small enough.

**Sequencing consequence:** benchmark the chosen instance-resolution mechanism *in isolation*,
against the >95% gate, **before** writing any of the four changes below. If none of the three
options clears the gate, this step does not ship in this form and the human needs to know that
before the work is done, not after.

### Four coupled changes, if it happens

1. **Instance-conditional targets** — unclicked instances → **background**. This is settled, not an
   open choice: ignore-label would exclude those voxels from the loss and therefore never teach
   suppression, which is the entire point of the rule in §1. Do not use ignore-label here.
2. **Per-variant targets.** `collate_patches` currently reuses one target across all variants
   (`patch_render.py:98`). Once the kept click set varies per draw, each variant needs its own
   target. Structural change to that function.
3. **Click dropout, now that masking makes it correct.** Enable the dead branch at
   `sampling.py:47` (`click_modes.drop != 0`, `pos < 1.0`) so a fraction of annotated lesions go
   unclicked in training. Paired with change 1 those lesions become background targets, which is
   what actually creates the "visible lesion, no click, suppress it" examples the model has never
   seen. Rate is a human decision; start low (`pos ≈ 0.8`) and raise it while watching
   `none_clicked` and false negatives on `all_clicked`.
4. **The consistency term must be re-scoped or removed.** `loss_consistency`
   (`lightning_module.py:116`) rewards two prompt draws producing the same mask. If the two draws
   keep *different* lesion sets, their correct outputs differ, so the term would penalise correct
   behaviour. Fix: draw the kept set **once per patch** and vary only the jitter across variants —
   preserving the term's real intent (robustness to click placement noise) while keeping targets
   identical within a pair. With `drop = 0.0` today this constraint holds by accident, which is why
   nothing has broken yet.

### Explicitly rejected

**Click dropout (`click_modes.pos < 1.0`) without target masking.** It would leave unclicked
lesions as foreground in the target, teaching the model to segment things it was not pointed at —
the exact opposite of the goal. This was proposed and withdrawn during the source discussion. Do
not reintroduce it as a standalone change.

---

## Decisions reserved for the human

Do not resolve these yourself:

1. Target **cohort weights** (Step 3) — the 0.25 for d013 above is an illustration, not a decision.
2. Final **epoch budget** and whether to adopt a WSD schedule (Step 5).
3. If Step 6 is triggered: whether the consistency term is **re-scoped or dropped**, and the click
   **dropout rate** (what fraction of annotated lesions go unclicked in training).

Already settled — do **not** reopen: unannotated lesions are unidentifiable and need no special
handling (§1); unclicked instances map to **background**, not ignore-label (Step 6).

---

## Do not do

- Do not resurrect the d013 finetune stage. It is being removed deliberately (§1).
- Do not change the objective, the loss, or the prompt semantics before Step 1 produces evidence.
- Do not modify the definition of the aggregate `val_dice` — it is the link to the 600-epoch
  baseline.
- Do not grow `fit.py`, `lightning_module.py`, `data_module.py`, or `patch_iterable.py` past 200
  LOC. New logic goes in new modules.
- Do not add work to the per-patch dataloader path without benchmarking GPU utilisation.
- Do not batch multiple steps into one review. One step, one diff, one approval.
