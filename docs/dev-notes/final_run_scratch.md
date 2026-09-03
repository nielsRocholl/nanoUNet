# Final-run scratch (2026-09-03)

Product: promptable 3D CT lesion seg, longitudinal. d013 = melanoma, many small mets.
Do not judge d013 on pooled/global Dice. Last Dataset999 instance run: wandb `ekkxcgi6`, FT `i4q2oamw`.

## Loss (settled)

Plans set `batch_dice: false`. Dice = mean of per-row Dice. Each of 12 rows (6 patches × 2 prompts) equal.
CE = `nn.CrossEntropyLoss` mean over voxels → large lesions still dominate the CE half. Not worth changing.
`cc_dc_ce`: tried, ~4× step time (CPU cc3d + Voronoi in the hot step). Hard NO.
CC-Dice only differs from per-patch Dice when ONE target contains several CCs of very different size.
Prompt-centred crop + `instance_targets` (often 1 kept lesion) → that case is uncommon on the mixed pool.
d013 FOV can pack several mets; still cannot pay 4×. Per-patch Dice already equalizes a small-only patch vs a large-only patch.

## Last run, right numbers

Instance 1200ep (`longrun.json`, d013 forced 25%, MAE init, dc_ce):
- pooled val_dice 0.69 (best 0.71 @ ep 1005). Wrong headline for melanoma.
- val_dice_macro 0.55. Per-row, closer.
- val/tag/small 0.62 end (best 0.83 @ 791, then decay). Mixed-pool small, not d013-only.
- val/all_clicked 0.77 (old semantic 600ep was 0.84; harder objective, expected).
- selectivity_margin −0.23 (target: positive). Did not flip. vs_clicked_subset 0.36 (wanted up from 0.47).
- val/cohort/d013 pooled 0.35 @ end, n=41, swings 0.33–0.84. Noise. Ignore.
- Same ckpt on FT’s 400-patch d013 all_clicked @ ep1: **0.76**.

d013-only FT 80ep (adamw 1e-5, only-prefix):
- all_clicked 0.76 → 0.74 (best 0.78 @ 23).
- small-tag 0.27 → 0.35. Large 0.63 → 0.59. Click-inside 0.77, outside 0.45.
- selectivity −0.08 → +0.02. Prompt gap 0.07 → 0.26.
- Long d013-only FT (500ep) previously specialised: val_dice 0.70→0.61. Do not repeat.

## Data 900 vs 999

Patient-level split fixed. d026 106 empty quarantined. 64 genuine zero-lesion kept (25 are d013 timepoints).
site_balanced: d013 = 1/11 = **9.1%** (last run forced 25%).
`configs/longrun.json` cohort keys are Dataset999 — missing d028–d031, has d010. **Will crash on 900.**
No 900 valset, no `*_weights.json`, no 900 MAE. 999 MAE exists (250ep, recon 0.107).

## Recipe

SSL 900 (train patients) → supervised 900 site_balanced + instance_targets + empirical prompts + dc_ce
→ short mixed FT, d013 oversampled, other cohorts remain as regularisers. Not `--only-prefix`. Not cc_dc_ce.
Watch: tag/small, macro, click_outside, selectivity, decoy pred_fg. Not pooled d013.
