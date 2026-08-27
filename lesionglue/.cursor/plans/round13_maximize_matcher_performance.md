# Round 13 — One-Retrain Performance Finish

**Status:** implementation plan; replaces the multi-run OOF plan.  
**Repository baseline:** `/lesion-tracking` at `48da24a`.  
**Hard budget:** at most **one optimizer-training run**, approximately 30 minutes. Forward checks, preprocessing, and evaluation do not count; no debug/smoke training is allowed.

**Coding contract:** before editing either `/lesion-tracking` or `/nanoUNet`, read and strictly obey `/nanochat-style` (`/lesion-tracking/.cursor/skills/nanochat-style/SKILL.md`). It applies to every code, config, CLI, script, test, and documentation change. If a change cannot comply, stop and redesign it.

## 1. Objective and honest limitation

Finish the best practical matcher today without an architecture search:

- keep registered geometry: `drop_dp=false`;
- keep complete intra attention: `intra="complete"`;
- keep `type_mask=false`;
- use EMA explicitly;
- calibrate `dust_tau` by inference only;
- spend the single retraining on the only high-prior data lever: 25 viable body-region graphs currently discarded by `_dom_fu`.

One retraining cannot provide clean OOF model selection. The existing holdout has already informed EMA and `dust_tau`; final comparison is therefore an **operational rollback check**, not an unbiased performance claim.

Known reference on the 56 valid holdout graphs:

- v7_complete raw, `tau=0.20`: match score `0.964869`;
- v7_complete EMA, `tau=0.20`: `0.968687`;
- v7_complete EMA, `tau=0.10`: `0.975190` (development estimate, not final evidence).

## 2. Explicitly removed from this round

Do not run CV, complete-vs-kNN, horizon sweeps, registration-tail augmentation, extra seeds, SWA selection, loss sweeps, or ensembles. Do not train `drop_dp` or `type_mask`. Do not run `scripts/round9.sh`; it expects `best.ckpt` and tunes on global val.

There is exactly one candidate training:

| Candidate | Cache | Steps | Seed | Models |
|-----------|-------|------:|-----:|-------:|
| geo+complete + all body-region graphs | `cache_v8_regions` | 7400 | 0 | 1 |

If preflight fails, keep the existing v7_complete model and use **zero** retrainings. Never spend the training slot on a compromised cache or configuration.

## 3. Phase A — no-training corrections and preflight

### A1. Make EMA explicit

Files: `tracking/train/module.py`, `tracking/cli/eval.py`, `tracking/cli/audit.py`, `README.md`.

1. Add non-checkpoint state `self._eval_use_ema: bool | None = None`.
2. Add `set_eval_weights(use_ema: bool)`; requesting absent EMA raises instead of falling back.
3. `_val_net()` honors `_eval_use_ema` when set; in-training validation retains `_ema_ready()`.
4. Add `--no-ema` to `eval.py`; EMA remains the default. Route `audit.py` through the same choice.
5. Make `--dust-tau` repeatable and add `--out`; write every threshold’s counts/metrics plus the selected tau. Threshold sweeps rerun inference only, never training.

Acceptance: after loading `/nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt`, `global_step` may be zero but explicit EMA evaluation must still use `ema_matcher`.

### A2. Guarantee the only final checkpoint

File: `tracking/cli/train.py`.

1. Add optional `--seed` and `--max-steps`; apply before `seed_all` and `dump_config`.
2. After no-fold `trainer.fit`, explicitly save `out/"last.ckpt"` and assert it exists.
3. Write `fold_metrics.json` with `val_disabled=true`, `selector="last"`, and no fake validation scores.
4. Refuse an output directory already containing a checkpoint; the error must show a new literal `--out` path.

### A3. Recover the discarded body-region graphs

Files: `tracking/data/graph.py`, `tracking/data/dataset.py`, `tracking/data/staging.py`, `tracking/cli/preprocess.py`, `README.md`.

`img_id_fu` does **not** mean another longitudinal follow-up. Some baseline/follow-up visits are stored as multiple body-region volumes. The current code keeps only the region containing the most lesion rows.

1. Remove `_dom_fu`; parse each patient once and build one graph for every sorted follow-up body-region volume (`img_id_fu`).
2. Return `list[HeteroData]` from `build_hetero_data`; skip only regions with an empty BL or FU side.
3. Store `data.pid`, `data.img_id_fu_used`, and `data.graph_id=f"{pid}_{img_id_fu:02d}"`.
4. Stage one list per patient at `{pid}.pt`; flatten during collation. Save empty lists so `--resume` is exact.
5. Keep `CACHE_TAG="v7_native"` because the tensor schema is unchanged. Isolate cardinality with a new cache root:
   `/nnunet_data/lesion_tracking/cache_v8_regions`.

Build:

```bash
cd /lesion-tracking
export PYTHONPATH=.
python3 tracking/cli/preprocess.py \
  --split all --jobs 16 \
  --root /nnunet_data/Longitudinal-CT \
  --cache /nnunet_data/lesion_tracking/cache_v8_regions
```

Hard preflight gates:

- total graphs: `306`;
- non-holdout train∪val graphs: `247`;
- holdout graphs: `59`;
- every graph has finite features and expected dimensions;
- every holdout patient ID is absent from train∪val;
- one batched forward pass through the existing v7_complete checkpoint succeeds;
- dataloader throughput is measured before/after; reject the change if GPU input time regresses materially.

Temporary checks must perform no optimizer step and must be deleted after validation.

## 4. Phase B — the single training run

Create `configs/complete.json` from `configs/nodp_complete.json`, changing only:

```json
{
  "drop_dp": false,
  "intra": "complete",
  "type_mask": false,
  "max_steps": 7400,
  "seed": 0,
  "dust_tau": 0.10
}
```

All omitted fields remain byte-for-byte identical to `configs/nodp_complete.json`. Do not change augmentation, optimizer, losses, width, depth, or batch size. h60_r9 continued to about epoch 330, but its deployed best checkpoint is epoch 239/step 5500. v7_complete is epoch 217/step 6050 on 224 graphs and its NCE loss was still declining. With 247 graphs and batch size 8, 7400 steps is about 239 epochs: modestly longer than v7 and equal to the deployed h60 exposure, without copying non-deployed late h60 epochs.

Run once:

```bash
cd /lesion-tracking
export PYTHONPATH=.
python3 tracking/cli/train.py \
  --config configs/complete.json \
  --root /nnunet_data/Longitudinal-CT \
  --cache /nnunet_data/lesion_tracking/cache_v8_regions \
  --out /nnunet_data/lesion_tracking/runs/r13_one_retrain/final_seed0 \
  --seed 0 --max-steps 7400 --no-early-stop \
  --wandb --wandb-run-name r13-one-retrain-v8-s0
```

This command must omit `--fold`, so training uses every non-holdout graph and no validation. Do not launch another training command. If the process fails, diagnose first; restarting requires explicit user approval.

Post-train assertions:

- `last.ckpt` exists and loads;
- resolved `config.json` matches the table above;
- checkpoint contains EMA state;
- losses stayed finite;
- training duration and step throughput are recorded.

## 5. Phase C — inference-only threshold calibration and rollback

Evaluate old and new checkpoints with EMA on the same old v7 test cache. Sweep `tau ∈ {0.05,0.075,0.10,0.125,0.15,0.18,0.20,0.22,0.25}` for each checkpoint and select the highest match score; exact ties choose the value closest to `0.10`. This keeps the event set identical and requires no retraining.

```bash
python3 tracking/cli/eval.py \
  --ckpt /nnunet_data/lesion_tracking/runs/v7_complete/last.ckpt \
  --split test --dust-tau 0.05 --dust-tau 0.075 --dust-tau 0.10 --dust-tau 0.125 \
  --dust-tau 0.15 --dust-tau 0.18 --dust-tau 0.20 --dust-tau 0.22 --dust-tau 0.25 \
  --cache /nnunet_data/lesion_tracking/cache_v7 \
  --out /nnunet_data/lesion_tracking/runs/r13_one_retrain/old_common.json

python3 tracking/cli/eval.py \
  --ckpt /nnunet_data/lesion_tracking/runs/r13_one_retrain/final_seed0/last.ckpt \
  --split test --dust-tau 0.05 --dust-tau 0.075 --dust-tau 0.10 --dust-tau 0.125 \
  --dust-tau 0.15 --dust-tau 0.18 --dust-tau 0.20 --dust-tau 0.22 --dust-tau 0.25 \
  --cache /nnunet_data/lesion_tracking/cache_v7 \
  --out /nnunet_data/lesion_tracking/runs/r13_one_retrain/new_common.json
```

Precommitted operational gate:

- deploy the new model only if common-set match score and persistent correct count are both at least the old model’s;
- disappeared and newly-appearing correct counts may each decrease by at most one;
- otherwise keep v7_complete. No threshold adjustment, second seed, or second training is allowed.

After this decision, evaluate the deployed checkpoint once on `cache_v8_regions` to report 59-graph coverage. Never compare its absolute score directly with the old 57-graph score.

## 6. Finish

Write `/nnunet_data/lesion_tracking/runs/r13_one_retrain/final_manifest.json` with:

- deployed checkpoint and SHA256;
- fallback checkpoint;
- git commit;
- exact config and cache root;
- graph counts, seed, steps, EMA, tau;
- old/new common-set metrics;
- training duration and throughput;
- `holdout_status="already_opened_development_set"`.

Update the handoff and README in the same change. Final outcome is either:

1. improved v8 model from **one** training; or
2. unchanged v7_complete EMA model from **zero** trainings.

Expected wall time: implementation and cache build ≤2 hours, training ≈40 minutes, evaluation and handoff ≤30 minutes.
