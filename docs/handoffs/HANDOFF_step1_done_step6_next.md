# Handoff — Step 1+2 shipped; Steps 4 and 6 in flight

**Written 2026-08-05.** Audience: an agent with no access to the session that produced this.
Everything needed is here. Read `docs/handoffs/HANDOFF_training_overhaul.md` (the parent plan) and
`docs/handoffs/PLAN_step1_valset.md` (the Step 1+2 spec, including its own correction section)
before acting.

**Status in one line:** the measurement apparatus is built and has produced a decisive diagnosis.
The model does not use the click to select *which* lesion to segment. Step 6 is the fix.

---

## 0. Environment gotchas — read before running anything

| Gotcha | Fix |
|---|---|
| Container env vars point at `/nanounet_data/...`, which **does not exist**. Every nanoUNet CLI dies on startup. | `export NANOUNET_RAW=/nnunet_data/NanoUNet_raw NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed NANOUNET_RESULTS=/nnunet_data/NanoUNet_results` in **every** shell. |
| `/nanoUNet` is **not persistent**; `/nnunet_data` is. | Commit early. Anything uncommitted is lost. |
| The container **cannot push** (no GitHub credentials). | Commit here; the human pushes from their own shell. Tell them each time something is waiting. |
| Repo-root `scripts/slurm_*.sh` are **stale copies** with a broken `nnUNet_raw` path. | The real ones live at `/home/nielsrocholl/SLURM/jobs/nanoUNet/prompt-robustness/`. Ignore the repo-root ones. |
| This box is a **shared A100-40GB**. Batch 12 OOMs; the production H200 is bigger. | Use `--batch-size 6` for local validation work and say so when reporting. |
| `fit.py` is 212 LOC, over the 200 limit. | The human has explicitly allowed it to run slightly over if the code is better for it. Do not contort it. |

---

## 1. What was built and verified

All committed on `main`. Seven commits from this session, ending at `3a96a8e`.

| Commit | Content |
|---|---|
| `6ccd8ae` | removed permanent `tests/` (style rule R16) |
| `5aca661` | the Step 1+2 plan |
| `2eb2fe1` | `nanounet_build_splits` + the new split applied |
| `cb72e00` | plan correction (D14/D15, see §1.2) |
| `5b21ec3` | `nanounet_build_valset` + `valset.py` / `valset_alloc.py` / `valset_build.py` |
| `2c2b975` | manifest wiring, `val_metrics.py`, per-scenario + per-cohort metrics |
| `3a96a8e` | `val_prompt_agreement_clicked` |

### 1.1 New split — live on disk

`/nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/splits_final.json` is now a **1-element**
list (one split, `--fold 0` only). Old 5-fold file backed up beside it as
`splits_final.backup-20260805-142839.json`.

**4984 train / 882 val = 15.0%, balanced within every source dataset** (was a dataset-blind 5-fold
that drifted 13.0–24.6% per cohort). 17 cohorts, `d010`…`d027`; **`d026` is deliberately absent —
that raw dataset was corrupted.** Do not "fix" it.

### 1.2 Fixed validation manifest — live on disk

```
/nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/valset_1500.json         (0.4 MB)
/nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/valset_1500.targets.npz  (0.25 MB)
```

1500 patches pinned by (case, bbox, **both** click draws). Rebuild: 214 s, byte-identical from the
seed. Four mutually exclusive scenarios:

| Scenario | n | Tests | Correct output |
|---|---|---|---|
| `all_clicked` | 599 | normal accuracy | all lesions |
| `lesion_free_decoy` | 375 | does it invent lesions? | empty |
| `subset_clicked` | 300 | **does it segment only the clicked lesion?** | clicked subset only |
| `none_clicked` | 226 | does it stay quiet? | empty |

**`subset_clicked` covers 14 of 17 cohorts.** `d014`/`d016`/`d020` have **zero** val cases with ≥2
lesion instances — every case is single-lesion, so a strict subset is impossible at any patch
position. It is allocated by multi-lesion supply, capped at 30% per cohort (decision D14 in the
plan). Per-cohort metrics are therefore restricted to `all_clicked` rows (D15), so cohorts are
compared on identical difficulty rather than on different scenario mixes.

Everything expensive is **offline**: `cc3d`, the clicked-subset targets (packed bits in the `.npz`),
both prompt draws, the click-inside flags. `nanounet/data/valset.py` does pure tensor work.
**Do not add `cc3d` or EDT to the validation path.**

### 1.3 Verification results

| Check | Result |
|---|---|
| Two `Trainer.validate` runs, same manifest + checkpoint | **identical on all 98 metrics** |
| Legacy path (no `--val-manifest`) | unchanged, 11 metrics, no manifest keys leaked |
| **GPU utilisation during validation** | **100%** |
| Validation wall time | 155 s / 1500 patches (batch 6, 8 workers, shared A100) |
| Train epoch | 454 s (1000 iters, batch 6, `prompts_per_patch` 2) |

Partial reads work: `open_case` yields **lazy** blosc2 arrays and `crop_patch` slices them, so a
patch decompresses ~14% of a volume's chunks (173 ms vs 424 ms full read, 2.4× faster). An earlier
claim that every patch decompresses a whole volume was **wrong** — do not repeat it.

---

## 2. The diagnosis — this is the important part

Measured on `Dataset999_Merged_..._f0_h200/checkpoints/best-epoch=570-val_dice=0.8030.ckpt`.

### 2.1 The model does not use the click to select a lesion

```
val/subset_clicked/val_dice_vs_all_lesions      0.7382
val/subset_clicked/val_dice_vs_clicked_subset   0.4673
val/subset_clicked/val_selectivity_margin      -0.2709
```

Three lesions in view, one clicked. The output matches **"segment everything"** 27 Dice points
better than "segment what was pointed at". This is the parent handoff's *strategy (b)* hypothesis,
confirmed. **Step 6 gate condition 2 is met.**

Corroborated by the silence test:

```
val/none_clicked/val_pred_fg        0.0196   lesions visible, nothing clicked
val/lesion_free_decoy/val_pred_fg   0.0018   no lesion present
```

11× more foreground when unclicked lesions are present than on genuinely empty tissue. It is
segmenting lesions nobody pointed at. On truly empty tissue it is well behaved.

### 2.2 The consistency loss has a degenerate solution — leading hypothesis

Timeline that matters: `/nnunet_data/prompt_sensitivity/FINDINGS.md` measured the **pre-consistency**
model (`Dataset114_..._finetune_dwb`) and found click-placement instability — σ_A 0.211, pairwise
prediction agreement P_A 0.588. Its recommendation was "retrain with prompt augmentation + a
prompt-consistency loss". **That was then implemented** (`--consistency-weight 0.02`,
`--prompts-per-patch 2`) and produced the current `f0_h200` model.

Current model: `val_prompt_agreement` (clicked rows) **0.87**, up from 0.588.

**The hypothesis: the consistency term was satisfied the cheap way.** It rewards two different
clicks producing the *same* mask. A model that ignores the click satisfies it perfectly at zero
cost. The parent handoff already warns of this (§1: "it can be satisfied by ignoring the click
entirely") and `lightning_module.py:136-138` calls it prompt collapse. The two numbers fit that
story exactly: agreement up **because** the click matters less, and selectivity at −0.27.

Not proven. `val_prompt_gap` is 0.12, so the click is not entirely ignored. Treat as the leading
hypothesis, not a finding.

**Do NOT conclude the instability from FINDINGS.md is solved.** The two measurements perturb the
click by very different amounts: the sweep sampled centre / deep / **6 border** positions across the
whole lesion, while `val_prompt_agreement` uses only registration-error jitter (σ ≈ 6 vox/axis).
Local stability under small jitter says nothing about centre-vs-border. Different model, different
data, different geometry too.

### 2.3 Both symptoms, one fix

Instance-conditional targets (Step 6) address both: two clicks on the same lesion get the **same**
target, so invariance becomes *supervised* rather than penalised — removing the degenerate shortcut
in 2.2 — and clicking one of three lesions gets a one-lesion target, which trains selectivity.

The two cheap alternatives are already **ruled out by measurement** (FINDINGS.md): decoupling the
patch from the click recovers 0.007 of 0.211; ensembling clicks buys 0.017.

### 2.4 Per-cohort — d025 is the outlier

| cohort | n | val_dice | agreement |
|---|---|---|---|
| d020 WORC_CRLM | 25 | 0.927 | 0.962 |
| d022 KiTS23 | 49 | 0.904 | 0.952 |
| … | | | |
| d014 MSD_Colon | 29 | 0.641 | 0.819 |
| **d025 RUMC_Bone** | 22 | **0.516** | **0.616** |

d025 is far worst on both. Independently corroborated — FINDINGS.md also found Skeleton worst
(0.39). Two independent measurements agree, so it is real.

Other numbers: aggregate `val_dice` 0.7961 (**not** comparable to the historical 0.8030 — different
val cases and different patch composition; `val/all_clicked/val_dice` 0.8390 is the analogue),
`val_dice_weighted` 0.8176, `val_prompt_gap` 0.0819, click inside 0.836 vs outside 0.759.

Quote `val_prompt_agreement_clicked`, **not** `val_prompt_agreement`: the flat version includes 601
rows (`none_clicked`, `lesion_free_decoy`) where both draws get an identical input and score a
trivial 1.0, inflating it to 0.92.

---

## 3. State of play — updated 2026-08-05, end of session 1

### Shipped and committed

| Item | Commit | Note |
|---|---|---|
| balanced 15% split | `2eb2fe1` | applied to disk, old file backed up |
| fixed val manifest + builder | `5b21ec3` | `valset_1500.json` on disk |
| manifest wiring + metric layers | `2c2b975` | 98 metrics, determinism verified |
| `val_prompt_agreement_clicked` | `3a96a8e` | the flat headline was inflated by 601 trivial rows |
| `--val-every-n-epochs` | `afc4a9f` | **human decided: 1500 patches every 2 epochs** |
| Step 6 benchmark result | `f1956a3` | the GPU gate is cleared, see §4 |
| Step 6 plan | `5d4961f` | `docs/handoffs/PLAN_step6_instance_targets.md` |

**Human pushes; this container has no GitHub credentials.** Check `git status` for unpushed commits
and say so explicitly.

### Session 1 end state — Steps 4 and 6

**Step 4 — DONE and verified.** Commit `2450134` (`--warmup-epochs`, `--ema-decay`, `--monitor`).

| Check | Result |
|---|---|
| `--warmup-epochs 0` vs pre-change LR curve | byte-identical, 600 steps, both schedulers |
| `--warmup-epochs 10` | linear ramp then unshifted curve |
| EMA train-step cost | -0.17% / +1.67% across two runs (bound 2%) — inside noise |
| `best-epoch=570` still loads | 0 missing/unexpected keys |

`--monitor` replaces `val_dice_macro if init_weights else val_dice`. That selector was
structurally blind to false positives (macro averages over foreground-bearing rows only; every FP
that matters lives on the rows with none), which is why the d013 finetune's `best-*.ckpt` pointed at
an epoch where pooled and prompt-ablated Dice were at their worst.

**UNMEASURED, do before the long run:** `EMACallback.on_validation_epoch_end` runs a **second full
pass over the val set** with the shadow weights to log `val_dice_ema` alongside `val_dice` (swapping
around the existing loop would replace `val_dice` rather than add to it). At 1500 patches that is an
estimated +60-100 s per validation, never measured. EMA is opt-in (`--ema-decay 0` default), and the
comparison is only needed while deciding whether to trust EMA — consider logging it every Nth
validation.

**Step 6 — code written, checks C1-C6 PASSED, not finished.**

The code is committed inside `fb6f957` (see the git-hygiene note below), files:
`nanounet/data/instance_target.py`, `nanounet/data/sampling.py`, `nanounet/config.py`,
`configs/instance_conditional.json`.

| Check | Result |
|---|---|
| C1 flag absent ⇒ byte-identical to pre-edit `build_patch` | **250 patches, 0 mismatches** |
| C2 `pos=1.0` no-op except unmapped components | 24/250 patches differ, 342652 voxels |
| C3 realised keep fraction at `pos=0.8` | **0.8104** (target 0.80 ± 0.02) |
| C4 both prompt variants share one target | 500 pairs, 0 mismatches |
| C5 target fg ⊆ real seg fg | 300 patches, 0 violations |
| C6 patches with a suppressed visible lesion | 81/300 = **0.270** |

Still outstanding for Step 6:
1. **Throughput measurement** `build_patch` flag off vs on, mean + p95 over >=100 patches. The
   isolated benchmark predicts +1.2%; the >95% GPU rule makes this a hard gate. NOT YET RUN.
2. **C7 + dropout rebalance (human asked for this).** C2 shows the target also loses lesion
   fragments clipped by the patch boundary — lesions whose centroid is outside the patch are never
   clicked, so they are correctly background, but that is suppression *on top of* click dropout.
   C3 (per-lesion) and C2 (per-patch) are different units and cannot be added. Measure at
   `pos = 1.0` over >=250 patches:
   `X = foreground voxels removed by boundary clipping / total foreground voxels`
   then set `pos = 1 - (0.20 - X)` so total suppression lands at ~20%. Caveat to state when
   reporting: dropout removes whole lesions (a clean "not yours" signal) while clipping removes
   fragments, so they may not deserve equal weight — give the human the number and let them decide.
3. `docs/reference/instance_targets.md` was never written.
4. The `wip/UNVERIFIED` label on the code was lost to a git mishap; treat `fb6f957`'s code half as
   verified only up to C1-C6 above.

### Git hygiene — a mistake made this session, do not repeat

`git add -A` was run while a subagent had work in progress, sweeping unverified Step 6 code into a
docs commit (`fb6f957`). The attempted fix — splitting that commit — rewrote history the human had
already pushed, and the next `git push` was rejected as non-fast-forward. Resolved by resetting to
`origin/main` and cherry-picking only the genuinely new commit on top (verified the trees were
identical first, so nothing was lost). **Stage explicit paths, never `-A`, while an agent is
running. Never rewrite a commit that may have been pushed.**

### In flight when session 1 ended

| Step | State | Files it owns |
|---|---|---|
| **Step 6** — instance-conditional targets | agent mid-verification; code on disk, docs not written | `data/instance_target.py`, `data/sampling.py`, `config.py`, `configs/instance_conditional.json` |
| **Step 4** — warmup / EMA / monitor fix | agent dispatched | `model/lr_schedule.py`, `train/ema.py`, `train/fit.py`, `cli/train_parser.py` |

Neither was verified by the orchestrator before the session ended. **Check both yourself** — read the
diffs, re-run the acceptance checks in `PLAN_step6_instance_targets.md` §6a, and do not trust an
agent summary (§5).

### Parked by the human, deliberately

**Step 3 — cohort-weighted sampling.** The *code* can be built any time (`config.py` CohortConfig,
new `nanounet/data/cohorts.py`, `patch_iterable.py:120`), but the **actual per-dataset weights are
deferred to the last step before the training run**. Do not pick them yourself.

Inputs for that decision when it comes:

| Cohort | train cases | share | val_dice | agreement |
|---|---|---|---|---|
| d013 Longitudinal | 456 | 9.1% | 0.826 | 0.755 |
| d025 RUMC_Bone | 128 | 2.6% | **0.516** | **0.616** |
| d014 MSD_Colon | 107 | 2.1% | 0.641 | 0.819 |
| d010 CECT | 917 | 18.4% | 0.821 | 0.902 |

d013 is the deployment target; d025 is the worst performer and independently confirmed worst in
`/nnunet_data/prompt_sensitivity/FINDINGS.md` (Skeleton 0.39). Both are candidates for
oversampling. The parent handoff's "0.25 for d013" is an illustration, not a decision.

### To finish Step 6 completely

1. Agent finishes implementation + acceptance checks C1-C6 and the throughput measurement.
2. **Orchestrator reviews the diff**, specifically the three silent traps in
   `PLAN_step6_instance_targets.md` §7.
3. Commit; human pushes.
4. **Update the real slurm script** at
   `/home/nielsrocholl/SLURM/jobs/nanoUNet/prompt-robustness/slurm_supervised_999_h200.sh`:
   - `--config configs/instance_conditional.json`
   - `--val-manifest ${LOCAL_PREP}/${DS_FOLDER}/valset_1500.json`
   - `--val-every-n-epochs 2`
   - add `--include "valset_1500*"` to the `rclone` whitelist near line 91. **Without this the
     manifest is never staged to the compute node and training fails at startup.**
5. **Short probe, ~60-100 epochs.** Step 6 changes the objective, so get evidence before spending
   days. Success criteria in `PLAN_step6_instance_targets.md` §6c; the headline is
   `val/subset_clicked/val_selectivity_margin` going from **-0.2709** to positive while
   `val/all_clicked/val_dice` holds at ~0.839.

### Remaining order

Step 6 (probe) -> Step 3 code -> **human picks cohort weights** -> Step 5a LR probe -> Step 5b long
run. Step 5 is where all the GPU time goes (~8-12 days); everything before it is hours.

Step 5a is three 60-epoch probes at `lr` in {0.005, 0.01, 0.03} compared on the fixed manifest --
this comparison is only readable *because* of Step 1. Step 5b is 1200-1800 epochs, no finetune
stage, and `--stretched-k`/`--stretched-ref` **must be re-tuned** for the new horizon (they are
absolute epoch counts; reusing 188/250 at 1800 epochs puts almost the whole run in the linear tail).
Optional and reserved for the human: a warmup-stable-decay schedule, which would allow branching and
annealing a checkpoint to decide whether to continue rather than committing to the horizon up front.

## 4. Then Step 6 — start with the benchmark, not the code

Parent handoff §Step 6 is authoritative. Both gate conditions are met (§2 above), so the human has
approved proceeding. **Sequencing is non-negotiable:**

> benchmark the chosen instance-resolution mechanism *in isolation*, against the >95% GPU gate,
> **before** writing any of the four coupled changes.

Step 6 needs per-lesion identity for every **training** patch, on the hot path, every step — unlike
Step 1, whose instance maps are offline. Three options, ranked in the parent handoff §2 ("Live risk:
on-demand cc3d"). Two facts from this session make the benchmark cheap:

- **`bboxes_zyx` is already on disk for every case** (verified 200/200 sampled), so option 2 needs
  no re-preprocessing. This removes the feared multi-day data job.
- **Crop-sized `cc3d` is demonstrably fast** — the manifest builder ran 1500 of them inside 214 s
  wall, including full-volume reads. Option 3 is not obviously unaffordable.

### BENCHMARK RESULT — measured 2026-08-05, the gate is CLEARED

Measured over 39 real training patches drawn from the val split, against the real `build_patch`
cost on the same box:

| Path | mean | p95 | vs `build_patch` | vs per-patch CPU budget @16 workers |
|---|---|---|---|---|
| `build_patch` (baseline, dominated by the blosc2 crop read) | 477.1 ms | 1388.3 ms | — | — |
| **Option 3 — `cc3d` on the crop** + `np.isin` mask | **5.7 ms** | 8.0 ms | **1.2%** | **0.24%** |
| Option 2 — bbox lookup from the sidecars | 0.3 ms | 0.7 ms | 0.1% | 0.01% |

Budget derivation: the GPU consumes 3 patches per 454 ms step (batch 6, `prompts_per_patch` 2), so
6.6 patches/s; with 16 dataloader workers each worker has ~2421 ms per patch before it starves the
GPU. Option 3 uses 0.24% of that.

**Choose option 3.** It is the parent handoff's "last resort", but the measurement says it costs
~200x less than the budget, and it is the only option that is *exact*: no bbox overlap
approximation (option 2 is wrong where two lesions' bboxes intersect), no extra storage, no
re-preprocessing of the 542 GB pool (option 1). The handoff's warning not to assume it was cheap was
right; it was checked, and it is cheap.

Note `build_patch` p95 is 1388 ms — the per-patch cost is dominated by IO variance, not compute, so
5.7 ms is inside the noise of what the path already costs.

Judge Step 6 **only** on the Step 1 strata — especially `val/subset_clicked/val_selectivity_margin`
and `val/none_clicked/val_pred_fg`. Training loss **will rise** relative to the old objective; that
is expected and documented, not a regression.

Reserved for the human, per the parent handoff: the click **dropout rate**, and whether the
consistency term is **re-scoped or dropped**. §2.2 above is direct evidence for that decision.

After Step 6: Steps 3, 4, 5 (cohort-weighted sampling, warmup/EMA/monitor fix, LR probe + long run).

---

## 5. Working notes

- **Subagents:** two of three implementation agents stopped mid-task leaving a detached background
  process and reported nothing useful. Verify their work yourself; do not trust the summary.
- One agent **correctly stopped** rather than guess when it found the plan unsatisfiable (the
  `subset_clicked` / single-lesion-cohort conflict). Instruct agents to do that; it was right.
- Errors found in **this session's own plan**, both mine: a per-cohort scenario quota that was
  mathematically impossible, and an exact-global-total assertion that per-cohort rounding cannot
  satisfy. Expect the plan to be wrong somewhere and check rather than force the code to comply.
- The manifest must be added to the `rclone --include` whitelist in the real slurm scripts
  (`slurm_supervised_999_h200.sh`, around line 91) or training will not find it on the compute node.
- No permanent `tests/` folder (R16). Write throwaway scripts in the scratchpad, run, delete.
