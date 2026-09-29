# Handoff — Round 13 complete

**Last updated:** 2026-08-27 13:52 CEST. Plan: `.cursor/plans/round13_maximize_matcher_performance.md`.

## Decision

**Keep `v7_complete`.** One retrain on `cache_v8_regions` (247 fit graphs, 7400 steps, seed 0) did not pass the precommitted common-set gate.

| | tau | match | persist | disappeared | new |
|---|---:|---:|---:|---:|---:|
| old `v7_complete` EMA | 0.125 | **0.970133** | 443/460 | 185/189 | 80/82 |
| new `final_seed0` EMA | 0.075 | 0.968810 | 443/460 | 184/189 | 80/82 |

Match score dropped 0.0013. Persist tied; disappeared −1 (allowed); newly-appearing 0. Gate requires match **and** persist ≥ old → reject.

Deployed: `/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` (SHA256 `94f16e6f…f2a98f`), EMA, `dust_tau=0.125`.
Manifest: `/nnunet_data/lesion_tracking/runs/r13_one_retrain/final_manifest.json`.

v8-only coverage of the same deployed ckpt (59 graphs, not comparable to 57-graph score): match 0.970241, persist 445/462.

## Artifacts (do not overwrite v7 / h60)

- cache: `/nnunet_data/lesion_tracking/cache_v8_regions` — 199+48+59=306 graphs
- candidate (not deployed): `runs/r13_one_retrain/final_seed0/last.ckpt`
- wandb: https://wandb.ai/hyper-alignment/lesion-tracking/runs/t9sj9zt4

## What was slow about `--jobs 16`

CIFS (`/nnunet_data`) + OpenMP oversubscription: 16 workers × default `torch` thread count (24 on this cgroup) thrashed gzip NIfTI decompress. Tornadus: ~1.3 patients/min. Arceus with `OMP/MKL/OPENBLAS_NUM_THREADS=1` in each worker: ~10 patients/min (30 min for all splits). One test case (`f2fc990265`, 922 MB nii.gz) took ~7 min alone. Dataloader: v7 33.8 ms/batch vs v8 36.3 ms/batch (ratio 1.075). Pin lives in `dataset.py` `_limit_threads`.

## Code on `main`

HEAD after this commit: `_limit_threads` in `dataset.py`; inference defaults (`DEPLOYED_CKPT`, `DEPLOYED_DUST_TAU=0.125`, hungarian, EMA) in `common.py` / `lesion_track` / `eval`.
