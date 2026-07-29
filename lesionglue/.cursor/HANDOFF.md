# Handoff — Round 12 in progress

**Last updated:** 2026-07-29 13:15 UTC. Update this file whenever state changes.
**Supersedes** the 2026-06-09 handoff, which was stale and wrong in several load-bearing ways (§7).

---

## 0. TL;DR for a fresh agent

**Phase:** Round 12. Thesis: this problem is **measurement-, label- and representation-limited, not
capacity-limited**, so R12 does not touch the matcher. Read, in order:
1. `.cursor/plans/round12_measurement_features_data.md` — the plan (stages A–G, coding philosophy §1)
2. `.cursor/plans/round12_findings.md` — measured results so far
3. `.cursor/rules/nanochat-style.mdc` — mandatory coding style
4. this file

**Right now:** a 5-fold selector bake-off is training (§3). Nothing is blocked on a human.

**Bootstrap command** (paths are NOT the defaults — see §2):
```bash
cd /root/lesion-tracking && export PYTHONPATH=.
python3 -c "from tracking.config import load_config; print(load_config('configs/base.json'))"
```

---

## 1. Why R12 looks nothing like R7–R11

Verified from git: commit `8c40711` ("final version") **reverted Round 11 entirely** — deleted
`tracking/consistency.py`, `configs/geo_{on,off}.json`, `scripts/round{10,11}.sh`, reverted
`CACHE_TAG` to `v5_l0`. HEAD before R12 was functionally identical to r9_base.

| Round | Lever | Outcome |
|---|---|---|
| R6 | **data** (node-drop aug) | the win |
| R7 / R7.1 | architecture | regressed, rolled back |
| R8 | loss/arch | rolled back in R9 |
| R9 | **measurement** (k-fold CV) | kept; proved configs tie within noise |
| R10 | architecture | rolled back |
| R11 | architecture | **rolled back** |

Four straight architecture rounds produced nothing. **Do not propose another matcher change.**

---

## 2. Environment — the defaults in the code are WRONG

`tracking/common.py` has `DATASET_ROOT = /nnunet_data/unprocessed-universal-lesion-segmentation`,
which **does not exist**. Every CLI must be given explicit paths:

| what | path | note |
|---|---|---|
| dataset root | `/nnunet_data/Longitudinal-CT` | pass as `--root` |
| graph cache | `/nnunet_data/lesion_tracking/cache` | pass as `--cache`; **persistent** |
| runs | `/nnunet_data/lesion_tracking/runs/...` | **persistent** |
| repo | `/root/lesion-tracking` | **NOT persistent — push to main** |

**Only `/nnunet_data` survives this session.** Commit and push anything that matters.

**The cache was rebuilt this session** from pre-existing per-patient graphs at
`derivatives/graph-based-tracking/processed/cv/{train,val,test}/*.pt`, avoiding ~1 h of
preprocessing. Contents: **train 224 / val 28 / test 29** graphs, `CACHE_TAG=v5_l0`,
node feat dim 1387, cross edge dim 27. Verified to load without triggering `process()`.
Seed script (scratchpad only, not in repo — recreate from `dataset.py:125-132` if needed):
loads each `.pt`, asserts `assert_graph_feat`, writes `{split}_v5_l0{,_meta}.pt` via
`InMemoryDataset.save`.

### Hardware (measured, not assumed)
- **RTX 2080 Ti, 11 GB, Turing → fp16 only, NO bf16.**
- 16 CPUs, 251 GB RAM. **Box is SHARED** — `radiom_remote`, `nninteractive-server`, a `slide2vec`
  DDP job all run here. Be a good citizen.
- **Training is CPU/Python-bound, NOT GPU-bound:** peak GPU **703 MB**, utilization **3%** for a
  single run. Bottleneck is the per-graph Python loop in `MatcherModule._loss` (Sinkhorn + InfoNCE
  per graph → thousands of tiny CUDA launches) plus CPU augmentation.
- Measured: `num_workers=0` → 1172 ms/step; `num_workers=4` → 794 ms/step (8000 steps ≈ 106 min).
- **Run folds CONCURRENTLY, but cap at 4.** `tracking/cli/cv.py` spawns folds *sequentially*, so
  launch `train.py --fold N` in parallel instead.
- **GPU MEMORY TRAP (learned the hard way — fold 2 OOMed at epoch 28 with 5 concurrent).**
  `torch.cuda.max_memory_allocated()` reported **703 MB** for one run over 60 steps, but the real
  per-process footprint is **1.2–1.8 GB**: the caching allocator reserves well beyond peak-allocated,
  and the largest graphs (86×95 lesions → big dense cross-edge tensors) exceed a short sample.
  Other users hold **~3.2 GB** of the 10.57 GB. **Budget ~1.8 GB/run against ~7.3 GB usable → 4 max.**
  Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to cut fragmentation.
- **Real throughput with 4-5 concurrent is ~0.5 steps/s per fold, not 0.87** → **~4.3 h** for 8000
  steps, not the 2.6 h a startup reading suggests. Measure over epochs, not the first progress line.
- **Do NOT count running folds with `pgrep -fc 'tracking/cli/train.py'`** — it also matches dataloader
  workers (returns ~21 for 4 folds). Key off `fold_*/fold_metrics.json`, written only on completion.
- Available future speedup (deliberate, not casual): vectorize the per-graph Sinkhorn/InfoNCE loops.
  Pure performance, no semantics — but touches `sinkhorn.py`, which plan §10 rules out of scope.

---

## 3. What is running right now

**5-fold selector bake-off (Stage B.3).** Launched 2026-07-29 ~13:00 UTC, ETA ~3.1 h (≈16:10 UTC).

```
RUNS=/nnunet_data/lesion_tracking/runs/r12_selector
for f in 0 1 2 3 4; do
  PYTHONPATH=. nohup python3 tracking/cli/train.py --config configs/base.json \
    --out $RUNS/fold_$f --fold $f \
    --root /nnunet_data/Longitudinal-CT --cache /nnunet_data/lesion_tracking/cache \
    --no-early-stop > $RUNS/fold_$f.log 2>&1 &
done
```
PIDs at launch: 1980280–1980284. Progress check:
```bash
for f in 0 1 2 3 4; do tail -c 1200 $RUNS/fold_$f.log | tr '\r' '\n' | grep -E "^Epoch" | tail -1; done
```
~320 epochs per fold (8000 steps / ~25 steps-per-epoch). `--no-early-stop` is deliberate: we want the
full curve to compare selectors.

**Each fold writes 4 checkpoints** into `$RUNS/fold_$f/`: `best.ckpt` (EMA monitor),
`best_raw.ckpt` (raw monitor), `swa_plateau.ckpt` (plateau weight average), `last.ckpt`,
plus `fold_metrics.json`.

**Status 2026-07-29 ~14:00 UTC:** folds 0/1/3/4 alive at ~epoch 68/320. **Fold 2 OOMed at epoch 28**
(5 concurrent did not fit — see §2 GPU MEMORY TRAP). A detached watcher
(`$RUNS/requeue_fold2.sh`, launched with `setsid`) polls for the first `fold_*/fold_metrics.json`
and then relaunches fold 2 alone with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Check it with `pgrep -af requeue_fold2.sh`; result lands in `$RUNS/requeue.status`.

**If these runs died:** relaunch the block above, **but no more than 4 at once**. Folds are
independent and idempotent (`train.py` clears stray ckpts but preserves the four named ones).

---

## 4. Findings so far (all verified — details in `round12_findings.md`)

**D1 — checkpoint selection is broken; worth ~2 pp, free.** `CKPT_MONITOR="val_match_score_ema"` is a
causal EWMA (beta=0.3). An EWMA still converging upward toward a plateau peaks at the END of it, so
`save_top_k=1, mode="max"` degenerates into "save last". Confirmed in 6/6 W&B runs: `disappeared`
and `newly_appearing` peak at step 34–150 and have decayed 3–9 pp by the selected step, and they
carry **50%** of `val_match_score` (weights 0.5 / 0.25 / 0.25).
> **Do NOT "fix" this by stopping early.** At step ~34 `unchanged_split` ≈ 0.07 — the model matches
> nothing, so disappeared/newly are ~1.0 *trivially*. That early peak is a degenerate null model.
> The real overfit is only the LATE segment (unchanged_split flat while disappeared decays). Fix =
> select at plateau / average across it. R9's "the dustbin memorizes" reading was half wrong.

**D2 — ZERO native SPLIT events.** `topology_class` ∈ {UNCHANGED 2506, DISAPPEARING 1407,
NEWLYAPPEARING 559, MERGING 166 rows / 38 groups}. Splits elsewhere are **synthetic reversed
merges**. So `val_acc_unchanged_split` is ~98.5% plain UNCHANGED accuracy. R10 and R11 both attacked
a ceiling defined partly by manufactured labels.

**D3 — registration dependence is REAL but UNIFORM, not per-node. Gate A was reformulated.**
The original framing ("~30% of GT positions are registration-imputed, stratify by it") is **wrong at
the node level** and was corrected by measurement: of **3,815 graph-eligible BL nodes, 0 lack a real
`cog_bl`**; every FU node has a real `cog_fu`. So the "imputed" stratum is empty (n=0).
`clickfix_report.csv`'s `n_bl_filled`/`n_fu_filled` count registration guesses handed to the
**missing** side of NEWLYAPPEARING / DISAPPEARED rows — those never become graph nodes.

The actual dependence: `graph.py:123` computes the descriptor at the **real annotated** `cog_bl`,
but `graph.py:124` sets **every** BL node's `pos` from `cog_propagated`, a registration projection.
Registration error is therefore a *global* property of the BL geometry channel — which is the same
conclusion the R11 critique reached, and consistent with R11 failing empirically.

**Reformulated Gate A** (see `round12_findings.md` §A.3.4): stratify by *case-level* registration
quality from `derivatives/registration_error_table.json` →
`excluded.{backend}.case_level_failure_patients` (26 `original` / 45 `unigradicon`) plus
`clickfix_report.csv` `status != "ok"` or `n_sanity_bad > 0`. Union of flagged patients **with
cached graphs**: 33/224 train, 5/28 val, 2/29 test.
> **Ordering consequence: Stage B.2 is now a PREREQUISITE for Gate A**, not an independent step —
> the 28-patient global val split cannot answer it; only pooled out-of-fold per-patient metrics can.

`registration_error_table.json` also holds measured per-lesion offsets binned by lesion size (the
empirical basis for `PROP_SIGMA` in `tracking/common.py`) — useful if a continuous stratifier is ever
wanted instead of a binary flag.

**D4 — "radiomics is slow" is FALSE.** No pyradiomics in the repo at all. `descriptor_l0` is 4 scales
× 7³ = 1372 trilinear HU samples — microseconds/lesion. The bottleneck is I/O: full-volume
`nib.load().get_fdata()` over 48 GB of `.nii.gz` + per-lesion full-3D-mask scans in `mask_stats`.
**nanoUNet features would make preprocessing SLOWER, not faster** (same volume loads + a CNN forward).
Speed fix = Stage C (memmap + bbox). nanoUNet is an **accuracy-only** bet, and R8's MAE descriptor
already lost to L0 (0.919 vs 0.946).

**D5 — `configs/base.json` didn't exist.** FIXED (`git mv` from `r9_base.json`).

**D6 (new) — `_dom_fu` discards 10% of all transitions.** `graph.py:57-62` keeps only the dominant
`img_id_fu`; line 100 drops the rest. 335 transitions exist, 306 are viable, **281 graphs are
built**. Recovering the 25 unused viable transitions is **+8.9% training data** for free.
Scheduled as Stage A.5, deliberately AFTER the selector bake-off so effects don't confound.

**19 patients are silently skipped** (16 train / 2 val / 1 test): 14 `EMPTY_FU` (complete
responders — every lesion disappeared), 2 `EMPTY_BL` (registration failure: UNCHANGED rows exist but
`cog_propagated` is None for all), 3 dominant-FU casualties. IDs listed in `round12_findings.md`.

**Real headline number is ~0.91–0.94, not 0.946.** Retrieved: `r9_base_final` 0.9138; mergesplit
folds 0.8968 / 0.9443 / 0.9590. Fold spread is enormous (`unchanged_split` 0.833→0.957). Fold
identity dominates variance — which is exactly why R10/R11 read as flat.

---

## 5. Next actions, in order

1. **[blocked ~3 h] Stage B.3 selector bake-off.** When folds finish, evaluate all three selectors
   per fold and pick the winner:
   ```bash
   for f in 0 1 2 3 4; do for s in best best_raw swa_plateau; do
     PYTHONPATH=. python3 tracking/cli/eval.py --ckpt $RUNS/fold_$f/$s.ckpt --split val \
       --root /nnunet_data/Longitudinal-CT --cache /nnunet_data/lesion_tracking/cache
   done; done
   ```
   **CAUTION:** `eval.py --split val` evaluates the *global* val split (28 graphs), NOT the fold's
   held-out patients. For a correct per-fold number the fold's own val set must be used — check
   `tracking/train/datamodule.py` fold wiring and, if `eval.py` cannot target a fold, read the
   numbers from each fold's `fold_metrics.json` / W&B history instead. **Do not report fold results
   from the global val split.**
2. **Stage B.2 FIRST** (it now gates step 3) — per-patient out-of-fold metrics + patient-level
   bootstrap CI (`bootstrap_match_score` in `tracking/data/splits.py`). Each patient is in exactly
   one fold's val side, so concatenating per-patient records across the 5 folds gives ONE
   out-of-fold score per patient over the whole 224-patient pool. Replaces the statistically wrong
   "non-overlapping mean±std" rule with paired per-patient deltas + a 10k patient bootstrap.
3. **Gate A — registration-quality stratification** (needs step 2). Compare pooled OOF per-patient
   `match_score` on the **33 flagged** vs **191 clean** training-pool patients (§4 D3).
   Use a fold checkpoint, NOT `models/best.ckpt` (trained under the `final` split scheme → leaks
   against `cv/val`). If flagged patients are materially worse, report a registration-clean subset
   alongside the full number and leave geometry modelling dead.
4. **Stage A.5** — recover the 25 unused viable transitions (+8.9% data).
   **Leakage guard:** a patient contributing 2 graphs must have BOTH in the same CV fold.
   `splits.py::fold_map` keys on pid, so verify the datamodule filters by `pid`, not index.
5. Stage C (I/O speed), D (same-type negatives in InfoNCE), E (nanoUNet descriptor), F (time
   reversal), G (single test gate). See plan.

---

## 6. Code changed this session (all pushed to `origin/main`)

| commit | what |
|---|---|
| `7597568` | Round 12 plan |
| `7fc8fd1` | `configs/base.json` repair + `round12_findings.md` |
| `2a744b0` | Stage B.1: three selector checkpoints |

**Stage B.1 detail** (`tracking/train/module.py`, `tracking/cli/train.py`):
- `SWA_BAND=0.01`, `SWA_MIN_UPDATES=5` at module top.
- Plateau SWA updates when `raw >= self._val_score_peak - SWA_BAND`. **Keys off `_val_score_peak`,
  not `_best_raw_score`** — the latter only updates in an `elif` branch and is NOT the raw peak.
- **The SWA model is held in a plain list (`self._swa`), never assigned as a submodule.** This keeps
  it out of `state_dict`. Registering it broke `load_from_checkpoint` for *every* pre-R12 checkpoint
  (verified against `models/best.ckpt`) — that regression was caught and fixed. Do not "clean this
  up" into a normal attribute.
- `on_train_end` writes `swa_plateau.ckpt`, temporarily swapping SWA weights into both `matcher` and
  `ema_matcher.module` so eval picks them up regardless of its `use_ema` flag; raises if
  `_swa_updates < SWA_MIN_UPDATES` (a run that never plateaued is broken, not skippable).
- `train.py` gained a second `ModelCheckpoint` on raw `val_match_score`, and its stray-ckpt cleanup
  now preserves `best_raw.ckpt` and `swa_plateau.ckpt` (it previously would have deleted both).

**In flight, uncommitted:** `tracking/data/provenance.py` + `tracking/cli/audit.py` (Stage A.2/A.3),
being written by a subagent. Review before committing.

---

## 7. Corrections to the previous handoff — do not trust the old doc

| old claim | reality |
|---|---|
| "r9_base ~0.946 `val_match_score_ema`" | ~0.91–0.94; 0.946 was one favourable fold, not a CV mean |
| "`configs/base.json` + `tracking/config.py`" | `base.json` did not exist until this session |
| "Round 10 pivots to data" | R10 and R11 both went architecture, and both were reverted |
| "L0 = radiomics" | no pyradiomics anywhere; it is a 1372-point HU sampling grid |
| `DATASET_ROOT` default | points at a nonexistent path |
| "24/25 folds completed" | no `cv_summary.json` exists anywhere on disk |

---

## 8. Do NOT

- Propose or implement another matcher-architecture change (§1).
- "Fix" the disappeared/newly decay by stopping training early (§4 D1).
- Register `swa_matcher` as a submodule (§6).
- Use `models/best.ckpt` to evaluate against `cv/val` — split-scheme leak (§5.2).
- Trust `technical.md` (stale) or the pre-2026-07-29 `HANDOFF.md`.
- Touch the **test** split before Stage G.
- Report split-class accuracy as if it were real — there are no real splits (§4 D2).
- Assume the code's default `--root` / `--cache` work (§2).
