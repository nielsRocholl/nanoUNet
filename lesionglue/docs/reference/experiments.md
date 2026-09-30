# Deployed matcher, experiments and round 9

## Deployed matcher history

| Checkpoint | Holdout match | Status |
|---|---|---|
| `/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt` (L0, complete intra, EMA, hungarian, `dust_tau=0.125`) | **0.9701** (cache_v7 test, 57 graphs) | deployed |
| R13 v8-region retrain (extra-region graphs) | 0.9688 on the common set | lost the gate, not deployed |
| `h60_r9/best.ckpt` | 0.9453 | older baseline |

Node features are L0 descriptors (`DESC_DIM=1372`); see [technical.md](../technical.md).

## Encoding cost (keep L0)

Measured on pid `16a5cdae36` (90 lesions):

| Stage | Cost |
|---|---|
| L0 + mask_stats | 33341 ms CPU |
| `build_mask_graph` | 42107 ms wall |
| `track()` GPU forward | 19.2 ms |

The GPU is not the bottleneck, so L0 stays. An encoder GAP variant was skipped (the hook was over 40 LOC).

## Round 9 script

`lesionglue/scripts/round9.sh` runs CV, then optionally a final retrain, a `dust_tau` sweep and the test gate.

```bash
export RUNS=runs/round9          # optional; default runs/round9
export CONFIG=lesionglue/configs/base.json  # optional
bash lesionglue/scripts/round9.sh
RUN_FINAL=1 bash lesionglue/scripts/round9.sh   # retrain on full train+val, sweep dust_tau on val, eval test once
```

Cluster: `lesionglue/scripts/lesion-round9-cv.sh` (SLURM; sets `RUNS` on `/nnunet_data`).
The per-stage commands it wraps are `lesionglue_cv`, `lesionglue_train`, `lesionglue_eval --dust-tau ...` ([train.md](../steps/train.md), [eval.md](../steps/eval.md)).
