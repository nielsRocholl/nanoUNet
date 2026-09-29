---
name: round8-step-hardneg-geometry
overview: "Remove batch-size confounds with step-based training, EMA by update count, graph-scoped InfoNCE, descriptor normalization, hard local negatives, and an eval-time dust threshold override. Stage 2 v6 geometry features are gated on the no-cache Round 8 smoke result."
todos:
  - id: plan_file
    content: "Create this .cursor plan copy before implementation."
    status: completed
  - id: step_training
    content: "tracking/cli/train.py: add max_steps, val_check_steps, warmup_steps, val_batch_size, and step-mode Trainer wiring."
    status: completed
  - id: ema_steps
    content: "tracking/train/module.py: replace epoch-gated EMA with ema_start_step and update-count validation gating."
    status: completed
  - id: nce_scope
    content: "tracking/train/match_utils.py + module.py: add graph-scoped InfoNCE default with batch scope as explicit ablation."
    status: completed
  - id: hard_negatives
    content: "tracking/train/match_utils.py + module.py: add nearest same-type hard local BCE with hard_pair_w and hard_k."
    status: completed
  - id: desc_norm
    content: "tracking/matcher.py + module/train CLI: add descriptor LayerNorm default with --no-desc-norm ablation."
    status: completed
  - id: tau_eval
    content: "tracking/cli/eval.py: add --dust-tau override for post-train threshold sweeps."
    status: completed
  - id: smoke
    content: "Run static checks and, when feasible, a 1500-step l0 smoke with val_batch_size=1."
    status: pending
  - id: stage2_geometry
    content: "Only after smoke passes: bump cache tag to v6 and add PROP_SIGMA-normalized cross-edge geometry channels."
    status: pending
isProject: false
---

# Round 8: Step-Based Training + Ambiguity-Focused Matching

## Summary

- Goal: improve the matcher by removing batch-size confounds, emphasizing ambiguous local negatives, normalizing descriptor usage, and later adding registration-uncertainty edge features.
- Style is hard-gated by `.cursor/rules/nanochat-style.mdc`: direct dataclasses/argparse, no factories, no registries, no `utils/`, no silent fallbacks, and touched files should end under 200 LOC.
- Mangotree MCP resources were not exposed during planning, so this plan is based on local repo inspection.

## Key Changes

### Training clock and validation hygiene

- In `tracking/cli/train.py`, add defaults: `max_steps=40000`, `val_check_steps=500`, `warmup_steps=1000`, `val_batch_size=1`, `early_stop_patience=20`.
- If `max_steps > 0`, run Lightning with `max_steps`, `max_epochs=-1`, `val_check_interval=val_check_steps`, and a step-based warmup+cosine scheduler.
- If `--max-steps 0`, preserve legacy epoch mode.
- Add `val_batch_size` to `MatcherDataModule`; train loader uses `batch_size`, val loader uses `val_batch_size`.

### EMA by update count

- Replace epoch-gated EMA with `ema_start_step=1000`.
- Update EMA only once `global_step >= ema_start_step`; validation uses EMA only after that same step threshold.
- Rename CLI flag to `--ema-start-step`; remove epoch semantics from code.

### Batch-size-controlled representation loss

- Add `nce_scope` choices `graph` and `batch`; default `graph`.
- `graph` computes InfoNCE independently per patient graph and averages graph losses, so batch size no longer changes the negative pool.
- Keep `batch` as an explicit ablation path for the old behavior.

### Hard local negative loss

- Add `hard_pair_w=0.2` and `hard_k=4`.
- Per graph, select all positive BL-FU edges plus the nearest `hard_k` negative FU candidates per BL row, preferring same-type negatives before filling by distance.
- Compute BCE on that selected set and add it to total loss as `hard_pair_w * hard_pair_loss`; keep existing focal pair loss unchanged.

### Descriptor normalization

- Add `desc_norm=True` to `ModelConfig` and `MatcherModule`; expose `--no-desc-norm`.
- In `NodeEncoder`, apply `LayerNorm(desc_dim)` to the descriptor block before concatenating stats and lesion-type embedding.
- No cache rebuild required for this part.

### Threshold tuning

- Add `--dust-tau` override to `tracking/cli/eval.py`.
- Post-train sweep: `0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35`; choose the val `val_match_score` winner for test/inference.

### Stage 2 geometry cache upgrade

- Bump cache tag from `v5_{mode}` to `v6_{mode}` only when adding geometry features.
- Extend `CROSS_DIM` from `27` to `31` by appending `dp / sigma_mm` as 3 channels and `mahalanobis_distance / 5` as 1 channel, where `sigma_mm = PROP_SIGMA * sp_fu`.
- Update both CSV-backed and mask-backed graph builders plus train-time augmentation so `cross_attr` always receives `sp_fu`.
- Treat v6 checkpoints and v5 checkpoints as incompatible because edge_attr dimensionality changes.

## Test Plan

- Static checks: instantiate `l0`, `mae`, and `yerebakan` datasets; assert node dims, edge dims, `feat_mode`, and cache tags match expectations.
- Smoke: train `l0` for `--max-steps 1500 --val-check-steps 250 --val-batch-size 1`; require finite losses and no collapse in `val_match_score`.
- Batch-size fairness: run batch sizes `1` and `8` for the same `4000` optimizer steps; compare curves by `trainer/global_step`, not epoch.
- Ablations: baseline step-mode, `--nce-scope batch`, `--no-desc-norm`, `--hard-pair-w 0`, then v6 geometry.
- Full target: `val_match_score >= 0.94`, `val_acc_unchanged_split >= 0.89`, `val_acc_disappeared >= 0.96`, `val_acc_newly_appearing >= 0.96`.

## Assumptions

- First implementation patch includes protocol, EMA, NCE scope, hard negatives, descriptor norm, and eval tau override.
- Stage 2 v6 geometry lands only after the no-cache Round 8 patch passes smoke.
- Feature-mode ensembling and topology-aware merge-rescue decoding are deferred until after the v6 geometry result is known.
