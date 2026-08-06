# Decisions taken without the human — review these

The human authorised autonomous work to completion, on the condition that every choice normally
reserved for them is recorded here with the reasoning, for review afterwards.

Each entry: what was decided, why, how to reverse it, and how much it matters.

---

## D-A1 — Cohort names are bare prefixes (`d013`), not underscored (`d013_`)

**Decision.** `sampling.cohorts` keys are `"d013"`.

**Why.** `cohort_of()` in `nanounet/plan/splits.py` splits a case id on the first underscore and
already returns `d013`. Accepting the underscored form would need a second normalisation path for
no benefit, and the same function is what the split builder and the val manifest already use.

**Reversible.** Trivially — a config key. An unknown name raises at startup naming the available
cohorts, so a stale `"d013_"` fails loudly.

**Stakes.** Low. `--only-prefix` still takes the underscored form; the train doc says so.

---

## D-A2 — Click dropout: `pos = 0.90` for the long run, `pos = 0.80` for the probe

### What was measured (C7, 389 patches, foreground voxels removed vs the old target)

| `pos` | lesions dropped | foreground voxels removed |
|---|---|---|
| 1.00 | 0% | **13.69%** ← boundary clipping alone |
| 0.92 | 8% | 16.15% |
| 0.80 | 20% | 28.45% |

**Boundary clipping alone removes 13.7% of foreground** — far more than anyone assumed. Lesions
whose centroid falls outside the patch are never clicked, so they are correctly background under
the project's rule, but they were never *designed* as a suppression signal and they dominate it.

### Two errors this exposed, both mine

1. `PLAN_step6_instance_targets.md` §6a proposed `pos = 1 - (0.20 - X)`, which assumes dropping 20%
   of *lesions* removes 20% of *voxels*. It removes 14.8 points. The formula is wrong; ignore it.
2. Linear interpolation from two points then predicted `pos = 0.915`. Measuring `pos = 0.92`
   returned 16.15%, not ~19%. The relationship is **strongly non-linear and noisy**: a handful of
   very large lesions dominate the voxel count (one case seen earlier is 573k voxels ≈ 2.8% of the
   whole 20.8M sample), so which lesions the dropout happens to remove swings the number.

### The decision

**Long run: `pos = 0.90`** (`configs/longrun.json`). Interpolating the two bracketing measurements
gives 0.895 for a 20% total; 0.90 is that, without pretending to precision the measurement does not
support.

**Probe: `pos = 0.80`** (`configs/instance_conditional.json`, unchanged).

**Why they differ.** The probe asks *"does instance-conditional targeting teach selectivity at
all?"* — that is cleanest at maximum signal. If it works at 0.80 we tune down; if it barely moves
at 0.80 it certainly will not at 0.90, and the answer is to raise the rate, not lower it. Probing
at the production value would return an ambiguous half-result.

### The tradeoff the human should weigh

Meeting the ~20% voxel target **weakens the signal that actually teaches selectivity**:

| | total suppression | whole lesions dropped |
|---|---|---|
| `pos = 0.80` | 28.5% | **20%** |
| `pos = 0.90` | ~20% | **10%** |

These are not equivalent teachers. Whole-lesion drops are the clean signal — "this entire object is
visible, has no click, leave it alone" — which is exactly what the −0.2709 selectivity margin says
is missing. Boundary fragments are lesion *pieces* at the patch edge, and the model may learn
"suppress things touching the edge", a cue with nothing to do with clicks.

The human asked for ~20% total, so that is what was set. **If the probe shows selectivity failing
to move, raising `pos` back toward 0.80 is the first lever, not the last.**

**Reversible.** One number in `configs/longrun.json`.

**Stakes.** Medium — it changes how strongly the new behaviour is trained.

---

## D-A3 — Cohort weights: d013 0.25, d025 0.08

**Decision.** `configs/longrun.json` sets `{"d013": 0.25, "d025": 0.08}`. The remaining 0.67 spreads
over the other 15 cohorts in proportion to their case counts.

**Why d013 → 0.25.** It is the deployment target (longitudinal CT) and currently 9.1% of training.
The parent handoff suggests 20–30% and calls its own 0.25 "an illustration, not a decision"; 0.25 is
mid-range. Oversampling inside the single stage is the whole point — the d013 *finetune* failed by
catastrophic specialisation, and weighting keeps the other cohorts regularising the trunk throughout.

**Why d025 → 0.08.** RUMC_Bone is the worst cohort on both axes measured on the fixed manifest —
`val_dice` **0.516**, agreement **0.616**, against a 0.79 aggregate — and it is only 2.6% of
training. This is corroborated independently: `/nnunet_data/prompt_sensitivity/FINDINGS.md` found
Skeleton the worst lesion type (0.39). 0.08 is roughly 3× its natural share.

**Why not more.** d025 has only 128 training cases. Beyond ~3× the same volumes recur often enough
to risk memorisation rather than generalisation, and the aim is to lift a weak cohort, not overfit it.

**Not applied to the probe** — see D-A4.

**Reversible.** A config block; empty reproduces the uniform draw exactly.

**Stakes.** Medium. This is the training mixture for a ~7-day run.

---

## D-A4 — The probe changes exactly one thing

**Decision.** `scripts/slurm_step6_probe_h200.sh` leaves `sampling.cohorts` empty and warm-starts
from `best-epoch=570-val_dice=0.8030.ckpt` at `lr 0.003` with 5 warmup epochs, 80 epochs.

**Why warm-start.** 80 epochs from MAE init would still be climbing basic Dice and would say nothing
about selectivity. Warm-starting isolates the objective change. The cost: this is a diagnostic, not
a clean training curve.

**Why no cohort weights.** Changing the objective and the data mixture together makes a null result
uninterpretable.

**Why lr 0.003.** 0.01 with momentum 0.99 would kick a warm-started net hard enough to confound the
read.

**Stakes.** Low — it is a diagnostic run, ~1 day.

---

## D-A5 — Long run: 1200 epochs, stretched-tail retuned to 376/500

**Decision.** `SUP_EPOCHS=1200`, `--stretched-k 376 --stretched-ref 500`.

**Why 1200 and not 1800.** The handoff's range is 1200–1800, resting on one log fit over epochs
200→600 predicting +4 to +7 Dice from doubling. 1800 is a 3× budget on the same single fit. If 1200
proves insufficient, the WSD schedule the handoff floats (branch, anneal over ~5%, evaluate, then
decide) is a better way to buy more than committing up front.

**Why 376/500.** These are absolute epoch counts, not fractions. Reusing 188/250 at 1200 epochs
would put 84% of the run in the linear tail. Scaling 2× with the horizon preserves the original
shape: poly decay against a 500-epoch reference for 376 epochs, then linear over the remaining 824.

**Known problem, documented in the script.** At ~531 s/epoch this is ~7.4 days against a 7-day
slurm limit. **Expect one resume.** The script takes `RESUME=<last.ckpt>` and does not delete `$OUT`
when set.

**Stakes.** High — it is the run.

---

## D-A6 — EMA left OFF in the long run

**Decision.** `EMA_DECAY=0.0`.

**Why.** `EMACallback.on_validation_epoch_end` runs a **second full pass** over the 1500-patch
manifest to log `val_dice_ema` beside `val_dice`, and that cost was never measured (estimated
+60–100 s per validation). Enabling an unmeasured cost on a 7-day run is the wrong trade. Measure
it first, or make the EMA pass run every Nth validation.

**Reversible.** One variable.

**Stakes.** Low-medium — EMA plausibly buys a few hundred epochs' worth of averaging, so it is worth
enabling once its cost is known.

---

## D-A7 — `LR=0.01` is a placeholder in the long-run script

**Not a decision — a flag.** The script ships with the inherited nnU-Net batch-2 value, which is
very likely wrong at 6 distinct patches per step. **Step 5a (three 60-epoch probes at 0.005 / 0.01 /
0.03 compared on the fixed manifest) has not been run.** The script says so in a comment at the
variable. Run the probe and replace the value before launching.

---

## Standing constraints honoured

- `configs/default.json` untouched — the 600-epoch baseline stays reproducible.
- No permanent `tests/` folder (R16); verification scripts written, run, reported, deleted.
- Every file under 200 LOC except `fit.py`, which the human explicitly allowed over.
- Explicit paths staged, never `git add -A` (that caused a mixed commit last session).
- Everything pushed — the container's SSH key works, so nothing waits on a manual push.
