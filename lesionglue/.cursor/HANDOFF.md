# Handoff — Round 9 (CV harness + matcher/dustbin)

Date: 2026-06-08. Plan: `.cursor/plans/round9_cv_matcher_dustbin_c68d185b.plan.md`.

## TL;DR
R9 **code is done + smoke-verified**. Remaining work is **GPU runs**: the 5-fold CV ablation
sweep, then threshold freeze + a single test-set gate. Driver script:
`scripts/round9.sh`. Run from repo root, always `PYTHONPATH=.`, interpreter is `python3`
(no `python` in the container).

## Why R9 exists (diagnosis)
- R6->R8 net gain (~1pp) is **inside the noise floor** of the tiny fixed val set, so single-split
  deltas are untrustworthy. -> Phase 0 introduces patient-level k-fold CV (mean±std).
- Two R8 knobs were shipped without ablation and are suspected harmful -> Phase 1 rolls them back.
- Two real ceilings: dustbin overfits late (`disappeared`/`newly` decay during training) and
  `unchanged_split` stuck ~0.90 (same-anatomy discrimination) -> Phase 2 matcher/dustbin changes.
- Stay on L0 descriptor (MAE/yerebakan underperform). No cache rebuild.

## What changed this session (all uncommitted)
New files:
- `tracking/data/splits.py` — seeded patient-level fold map (no BL/FU leakage), `aggregate_cv_folds` (mean±std).
- `tracking/matchability.py` — `RowMatchability` (LightGlue-style dustbin head) + `row_dust_marginals`.
- `tracking/cli/cv.py` — procedural k-fold loop -> per-fold `fold_metrics.json` -> `cv_summary.json`. Per-fold resume (skips folds with `fold_metrics.json`).
- `scripts/round9.sh` — full ablation sweep + ranking; `RUN_FINAL=1` does retrain best + dust_tau sweep + test gate.

Modified:
- `tracking/cli/train.py` — `--fold/--n-folds/--cv-seed`, `CKPT_MONITOR=val_match_score_ema`, writes `fold_metrics.json`.
  **Rollback defaults**: `hard_pair_w=0.0` (was 0.2), `desc_norm=False` (was True).
  **Ablation-speed defaults**: `max_steps=8000` (was 40000), `val_check_steps=250` (was 500), `early_stop_patience=8` (was 20).
- `tracking/train/datamodule.py` — fold-aware: pools cached train+val, splits by patient; train-only augment.
- `tracking/train/module.py` — logs `val_match_score_ema` (EWMA over val checks), tracks peak-raw + best sub-accs at best-EMA.
- `tracking/matcher.py` — `RowMatchability` dustbin, `nn.Bilinear` identity term in the pair head, `edge_cross_attn` flag (default OFF), `ModelConfig.desc_norm=False`.
- `tracking/cli/report.py` — defaults synced (`hard_pair_w=0.0`, `desc_norm=False`).
- `tracking/report.py` — re-exports `aggregate_cv_folds`/`load_cv_summary`.
- `tracking/cli/eval.py` — `--dust-tau` override (already existed) used by the tau sweep.

## Contracts to respect
- `MatcherOutput(pair, dust_bl, dust_fu, z_bl, z_fu)` is unchanged. Keep it that way.
- Checkpoint monitor metric is `val_match_score_ema` (max). Best ckpt = `<out>/best.ckpt`.
- Held-out **test** split is the final gate — touch ONCE, only in the `RUN_FINAL` stage.
- nanochat-style is law: `.cursor/rules/nanochat-style.mdc`. `module.py` is at the 200-LOC ceiling — if you edit it, split on a concept boundary, don't blow the cap.

## Step clock (so estimates make sense)
- ~25 steps/epoch (216-patient train pool, bs 8). `max_steps=8000` ≈ 320 epochs.
- R8 overfit by ~200 epochs (~5k steps); 8k gives buffer + early stop. EMA-best ckpt captures the peak.
- 3080 Ti: ~30–42 min/fold -> 5-fold/config ~2.5–3.5 h -> full 5-config sweep ~13–17 h.
  NOTE: new R9 dustbin may overfit slower; if `val_match_score_ema` still climbing at 8k in `r9_base`, bump `max_steps`.

## Experiment matrix (scripts/round9.sh)
Configs (5-fold each, rolled-back R9 defaults unless noted):
- `r8_baseline` = `--hard-pair-w 0.2 --desc-norm` (Phase-0 reference + Phase-1 bundle-ON arm)
- `r9_base` = defaults (rollback OFF + new matcher) — Phase-1 OFF arm
- `r9_nce_0.1 / r9_nce_0.2 / r9_nce_0.5` = `--nce-w` sweep (Phase-2c; 0.3 == r9_base)
Then it ranks by mean `val_match_score_ema` and writes `runs/round9/best_config.txt`.

## How to run (cluster, in-container)
```bash
cd /home/nielsrocholl/projects/git_projects/lesion-tracking
export PYTHONPATH=.
export RUNS=/nnunet_data/lesion_tracking/runs/round9   # persistent mounted dir -> resumable across jobs
bash scripts/round9.sh                                  # CV sweep + ranking (stops before test gate)
# after inspecting the ranking:
RUN_FINAL=1 bash scripts/round9.sh                      # retrain best + dust_tau sweep + TEST gate (once)
```
SLURM wrapper resources: `--qos=high --gpus-per-task=1 --cpus-per-task=12 --mem-per-gpu=24G --time=07:00:00`,
container image `dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest`,
mount `/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data`.
7h < full sweep -> just resubmit the same job; `cv.py` skips finished folds. Keep `RUNS` on `/nnunet_data`.

## Acceptance / targets
- R9 5-fold mean `val_match_score` with CI separated above `r8_baseline`.
- `unchanged_split` mean +>=2pp.
- `disappeared`/`newly` stop decaying during training (peak ≈ near-final) — the dustbin fix working.

## Open risks / gotchas
- `dust_tau` sweep in `RUN_FINAL` uses the original static val split (eval.py has no `--fold`), not per-fold CV val.
  Fine for a threshold pick; if you want strict CV-tau, add `--fold` support to `eval.py`.
- The `edge_cross_attn` branch uses a global softmax over all edges (not per-receiving-node) — questionable
  semantics, default OFF, experimental. Only enable if 2b alone underwhelms, and fix the masking first.
- Nothing is committed. `git status` lists all R9 changes. Decide commit vs keep-dirty.

## Pending todos (from the plan)
- `baseline_cv` (run `r8_baseline` 5-fold) — pending
- `rollback_knobs` (confirm `r9_base` vs `r8_baseline` on CV) — pending
- `nce_recheck` (nce_w sweep) — pending
- `tau_freeze_test` (dust_tau argmax -> retrain -> test once) — pending
Code todos (`kfold_splits`, `ema_best_monitor`, `dustbin_matchability`, `matching_bilinear`) — done.
