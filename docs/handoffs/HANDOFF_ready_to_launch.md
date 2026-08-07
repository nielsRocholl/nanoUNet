# Handoff — everything is built; the 1200-epoch run is ready to launch

**Updated 2026-08-07.** Audience: an agent or human with no access to the sessions that produced
this. Read this file first; `docs/handoffs/HANDOFF_training_overhaul.md` is the original plan and is
now largely historical.

**Status in one line:** the measurement apparatus, the objective fix, and the production recipe are
all built, verified and pushed. What remains is GPU time.

---

## 0. Environment — this bites every fresh container

| Gotcha | Fix |
|---|---|
| `NANOUNET_*` env vars point at `/nanounet_data`, which does not exist | `export NANOUNET_RAW=/nnunet_data/NanoUNet_raw NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed NANOUNET_RESULTS=/nnunet_data/NanoUNet_results` in **every** shell |
| Console scripts missing | `python3 -m pip install -e . --no-deps --no-build-isolation` |
| `git push` fails — remote resets to HTTPS | `git remote set-url origin git@github.com:nielsRocholl/nanoUNet.git`; the container's SSH key authenticates |
| `/nanoUNet` is **not persistent**; `/nnunet_data` is | Commit and push early. Write run outputs under `/nnunet_data`, never the scratchpad — that is how two runs were lost |
| Local box is a single A100-40GB | batch 12 OOMs; use `--batch-size 6` locally and say so |

`nnUNet_*` env vars are **not needed** — nanounet never imports nnunetv2 and reads only `NANOUNET_*`.

---

## 1. The problem, and what was found

The model segments a lesion when a user clicks it. **The click was not selecting anything.**

Measured on the fixed validation set with the 600-epoch model
(`..._f0_h200/checkpoints/best-epoch=570-val_dice=0.8030.ckpt`):

```
val/subset_clicked/val_dice_vs_all_lesions      0.7382
val/subset_clicked/val_dice_vs_clicked_subset   0.4673
val/subset_clicked/val_selectivity_margin      -0.2709
val/none_clicked/val_pred_fg                    0.0196   (vs 0.0018 on genuinely empty tissue)
```

Click one of three lesions and the output matches *"segment all three"* 27 Dice points better than
*"segment the one you clicked"*. The cause: the old training target was **every annotated lesion in
the patch, regardless of clicks**, so ignoring the click cost nothing.

---

## 2. What is built (all pushed)

| Piece | Where |
|---|---|
| Balanced 15% split, one fold | `nanounet_build_splits`; `splits_final.json` is a **1-element** list |
| Fixed 1500-patch val manifest, 4 scenarios | `nanounet_build_valset`; `valset_1500.json` + `.targets.npz` |
| Per-scenario + per-cohort metrics (99 of them) | `nanounet/train/val_metrics.py`, `--val-manifest` |
| Click-conditional targets | `nanounet/data/instance_target.py`, `sampling.instance_targets` |
| Cohort-weighted sampling | `nanounet/data/cohorts.py`, `sampling.cohorts` |
| LR warmup, EMA, explicit `--monitor` | `lr_schedule.py`, `train/ema.py`, `fit.py` |
| Always-on CSV logger | `cli/train.py` -> `<out>/metrics/version_*/metrics.csv` |

### The four validation scenarios

| Scenario | n | Tests | Correct output |
|---|---|---|---|
| `all_clicked` | 599 | normal accuracy | all lesions |
| `lesion_free_decoy` | 375 | does it invent lesions? | empty |
| `subset_clicked` | 300 | **does it segment only the clicked lesion?** | clicked subset only |
| `none_clicked` | 226 | does it stay quiet? | empty |

`subset_clicked` covers 14 of 17 cohorts — `d014`/`d016`/`d020` are single-lesion datasets, so a
strict subset is impossible there. Per-cohort metrics are restricted to `all_clicked` rows so
cohorts are compared on identical difficulty.

---

## 3. The membership fix — the most important code change

An earlier version of Step 6 decided lesion membership by whether a lesion's **centroid** fell in
the patch. That was wrong in a way that would have broken deployment:

- A large lesion spans several patches. In the neighbouring ones its centroid is outside, so the old
  rule made all of it **background** — measured at **13.69% of all foreground voxels**.
- `nanounet/infer/border_expand.py` finishes such a lesion by placing another patch wherever the
  prediction **touches a patch face**. Training the model to suppress those faces breaks that path.
- `nanounet/infer/longi_row.py:37-39` **clamps a click into any patch that contains none**, so at
  inference such a patch always gets a prompt. Training dropped the click and called it background —
  the exact opposite.

**Fixed in `96d4242`:** membership is by **voxel overlap** (components mapped to their parent lesion
via the sidecar `bboxes_zyx`), and a kept lesion whose displaced click left the patch gets its click
**placed on its own tissue** rather than dropped.

Verified: at `pos = 1.0` the target is identical to the old objective (**0.0000** of foreground
removed, was 0.1369) and **0/170** patches with foreground have zero clicks.

---

## 4. Probe results — read before interpreting any early metric

Three 50-epoch probes from MAE on a free A100 (batch 6, 250 iters/epoch), scored on the manifest.

| | B: new objective | C: new objective, **no consistency term** |
|---|---|---|
| `val_dice` | **0.3073** | 0.1751 |
| `val/all_clicked/val_dice` | **0.3392** | 0.1906 |
| `val_prompt_gap` | 0.0001 | **0.0000** |
| `val_prompt_agreement_clicked` | 0.9931 | 0.9990 |

**Two conclusions:**

1. **For the first ~50 epochs from MAE the model ignores the click entirely** (`val_prompt_gap` ~ 0).
   This happens **with and without** the consistency term, so it is a **training stage**, not a
   defect — a net must learn what a lesion looks like before a click can help it. The 600-epoch
   model reached a gap of 0.082. **Do not kill the run over an early flat prompt gap.**
2. **Removing the consistency term is strictly worse** (0.175 vs 0.307). It stays at 0.02.

A control run on the *old* objective from MAE was started twice and never finished. It would say
what the new objective costs at equal budget. **The human decided to skip it.**

**Warm-started probes from earlier sessions are void** — warm-starting measures how fast a model
unlearns its old solution, not what the new objective produces, and does not match the production
recipe. Artefacts kept at `/nnunet_data/NanoUNet_results/nanounet/step6_ab_probe/`.

**GPU gate: closed.** Median utilisation **98%** over a full 20-epoch run (1666 samples), peak
memory 25.9/40 GB.

---

## 5. The production run — ready

`/home/nielsrocholl/SLURM/jobs/nanoUNet/prompt-robustness/slurm_supervised_999_h200.sh`
(mirrored to `scripts/` in the repo).

```bash
sbatch /home/nielsrocholl/SLURM/jobs/nanoUNet/prompt-robustness/slurm_supervised_999_h200.sh
# ~7.4 days against a 7-day limit -> expect ONE resume:
RESUME=<out>/checkpoints/last.ckpt sbatch <same script>
```

| Setting | Value | Why |
|---|---|---|
| Objective | `configs/longrun.json`, `instance_targets`, `pos 0.80` | 20% of lesions deliberately unclicked; boundary clipping now contributes 0% |
| Mixture | cohort weights, site-balanced | Liver was 40% (over 7 datasets), lung 20%. Now liver 22%, lung 14%, d013 9->25%, bone 2.6->6%, colon 2.1->4%. Max 2.7x |
| Validation | fixed manifest, every 2 epochs | Noise removed at source, so half the cadence with 2.5x the patches costs ~3% not ~18% |
| Epochs | 1200 | 600-epoch curve still rising with no knee |
| Schedule | `stretched_tail_poly`, k 376 / ref 500 | **Absolute** epoch counts. 188/250 would put 84% of the run in the linear tail. 376/500 gives 824 epochs of slow decay |
| Warmup | 10 epochs | None existed |
| LR | 0.01 | Probe skipped by the human: U-Nets are LR-robust and 0.01 trained the current model |
| EMA | 0.999 | Shadow weights are free and ride in the same `.ckpt`; the `val_dice_ema` diagnostic runs every 25th validation (`EMA_VAL_EVERY`) |
| Monitor | `val_dice` | Replaces a silent switch to `val_dice_macro`, which is blind to false positives |

### Preflight checks in the script (all verified against live data)

manifest + `.npz` staged · `splits_final.json` has exactly 1 entry · sidecars carry `volume_vox`
**and** `bboxes_zyx` · `--include "valset_1500*"` in the rclone copy (without it the job dies
*after* staging 543 GB).

### No-leakage verification

| # | Check | Result |
|---|---|---|
| L1-L2 | train/val disjoint, union = all 5866 | PASS |
| L3-L4 | all 663 manifest cases are val; none are train | PASS |
| L5 | fold is `0`, **not** `"all"` (which returns the same list as both) | PASS |
| L6 | `--only-prefix` unset | PASS |
| L7 | cohort sampler never draws a val case (60k draws) | PASS |
| L8 | val loader is the manifest, 1500 patches | PASS |

All 24 training flags validated against the parser — no unknown or stale ones.

### What to expect, so nobody panics

- **`val_dice` will sit below the old run** and train loss will be higher: the target no longer
  contains unclicked lesions. Judge on the per-scenario metrics.
- **`val_prompt_gap` ~ 0 for the first ~50 epochs** — a stage, see §4.

### Success criteria (fixed manifest)

| Metric | Baseline | Target |
|---|---|---|
| `val/subset_clicked/val_selectivity_margin` | -0.2709 | **positive** |
| `val/subset_clicked/val_dice_vs_clicked_subset` | 0.4673 | **up** (the un-confounded one) |
| `val/none_clicked/val_pred_fg` | 0.0196 | -> 0 |
| `val/all_clicked/val_dice` | 0.8390 | hold |

Failure mode to watch: over-suppression shows as falling `all_clicked` while `none_clicked` looks
excellent. If that happens, `click_modes.pos` is the lever.

---

## 6. Open items

| Item | Status |
|---|---|
| Old-objective control from MAE | **Skipped by the human** |
| LR probe (3 x 60 epochs) | **Skipped by the human** — `lr 0.01` retained |
| WSD schedule | **Rejected by the human** |
| Score raw vs EMA weights at the end | Do this when the run finishes: `Trainer.validate` on the manifest, ~140 s each |
| `docs/handoffs/DECISIONS_autonomous_session.md` | 7 decisions taken without the human; D-A2 and D-A3 are now superseded by later measurements. Worth a read, not blocking |

---

## 7. Working notes

- **Score checkpoints, do not trust the progress bar.** `--no-wandb` used to discard all 99 metrics;
  a CSVLogger is now always attached, but a `Trainer.validate` pass on the manifest (~140 s) is
  still how you get a full per-scenario dump for a specific checkpoint.
- **Single-epoch snapshots are noisy.** Validation itself is deterministic (fixed manifest), but the
  model moves a lot between epochs early on. Prefer the CSV curve over any one point.
- Subagents in this series repeatedly stopped mid-task leaving detached processes, and reported
  results that did not survive checking. Verify their work.
- Both plans in this series contained a real error found during implementation (an unsatisfiable
  per-cohort quota; a wrong rebalancing formula). Expect the plan to be wrong somewhere.
- No permanent `tests/` folder (R16): write, run, report, delete.
