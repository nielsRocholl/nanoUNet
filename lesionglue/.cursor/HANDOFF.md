# Handoff — Round 13 one-retrain (paused)

**Last updated:** 2026-08-27 12:27 CEST. Continue from `.cursor/plans/round13_maximize_matcher_performance.md`.

## State

Code for Phase A is on `main` (this commit). **Training has not started.** Preprocess of `cache_v8_regions` was started on a slow node (~14/240 train patients in 11 min) and should be **restarted on a faster node**, not waited out.

Do **not** overwrite:

- `/nnunet_data/lesion_tracking/cache_v7`
- `/nnunet_data/lesion_tracking/cache` (v5/v6)
- `/nnunet_data/lesion_tracking/runs/h60_r9`
- `/nnunet_data/lesion_tracking/runs/v7_complete`
- `/nnunet_data/lesion_tracking/runs/v7_nodp_complete`

New artifacts only:

- cache: `/nnunet_data/lesion_tracking/cache_v8_regions`
- run: `/nnunet_data/lesion_tracking/runs/r13_one_retrain/`

## Done

- Explicit EMA: `MatcherModule.set_eval_weights`; `eval.py` / `oof.py` / `audit.py` default to EMA (`--no-ema` to disable).
- `eval.py --dust-tau` is repeatable; `--out` writes JSON with selected tau.
- No-fold train: `--seed`, `--max-steps`; refuse existing `*.ckpt` in `--out`; always write `last.ckpt`.
- `build_hetero_data` returns one graph per `img_id_fu` (body region, not extra timepoint). Staging stores a list per pid.
- `configs/complete.json`: `drop_dp=false`, `intra=complete`, `max_steps=7400`, `dust_tau=0.10`.

## Blocked / next

1. Kill any leftover preprocess PID on the slow node if it is still writing `cache_v8_regions`.
2. Diagnose why `--jobs 16` was ~1.3 patients/min (expected minutes total). Measure dataloader throughput before/after.
3. Rebuild `cache_v8_regions` from scratch on the fast node. Gates: 306 graphs total, 247 train∪val, 59 test; holdout IDs absent from fit; finite features; one forward through `v7_complete/last.ckpt`.
4. **One** train only:

```bash
cd /lesion-tracking && export PYTHONPATH=.
python3 tracking/cli/train.py \
  --config configs/complete.json \
  --root /nnunet_data/Longitudinal-CT \
  --cache /nnunet_data/lesion_tracking/cache_v8_regions \
  --out /nnunet_data/lesion_tracking/runs/r13_one_retrain/final_seed0 \
  --seed 0 --max-steps 7400 --no-early-stop \
  --wandb --wandb-run-name r13-one-retrain-v8-s0
```

5. Tau sweep old vs new on **cache_v7 test** (identical event set). Deploy new only if match score and persistent counts are ≥ old; disappeared/new may drop by at most 1 each. Else keep `v7_complete`.
6. Write `runs/r13_one_retrain/final_manifest.json`. Style: `/nanochat-style`. Budget: **one** optimizer run.
