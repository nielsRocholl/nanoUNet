# Fixed validation set

Replaces the per-epoch random val sample with a seeded, offline 1500-patch manifest split across 4
prompt scenarios, so per-scenario/per-cohort Dice curves aren't drowned in resampling noise.

## Commands

```bash
nanounet_build_splits -d 999 --plans nnUNetResEncUNetLPlans_h200_smallpv --val-frac 0.15 --force
```

```bash
nanounet_build_valset -d 999 --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --config configs/default.json \
  --out /nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/valset_1500.json
```

```bash
nanounet_train -d 999 -f 0 --plans nnUNetResEncUNetLPlans_h200_smallpv --config configs/default.json \
  --val-manifest /nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/valset_1500.json
```

## `nanounet_build_splits` arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d, --dataset_id` | int | required | Dataset ID (e.g. 999) |
| `--plans` | str | required | Plans identifier, no `.json` |
| `--val-frac` | float | 0.15 | Validation share, applied per source dataset |
| `--seed` | int | 12345 | RNG seed, recorded in the printed table |
| `--force` | flag | off | Required to overwrite an existing `splits_final.json` |

Writes a single train/val split, balanced **within each source dataset** (a plain KFold on a
17-cohort merged pool drifts to 13-25% val per cohort). The old `splits_final.json`, if present, is
backed up to `splits_final.backup-<timestamp>.json`, never silently overwritten.

## `nanounet_build_valset` arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d, --dataset_id` | int | required | Dataset ID (e.g. 999) |
| `--plans` | str | required | Plans identifier, no `.json` |
| `--config` | str | required | ROI/prompt JSON (e.g. `configs/default.json`) |
| `--out` | str | required | Manifest output path (`.json`) |
| `--n-patches` | int | 1500 | Total validation patches |
| `--floor` | int | 40 | Minimum patches per source dataset |
| `--mix` | str | `0.40,0.25,0.20,0.15` | Scenario shares: `all_clicked, lesion_free_decoy, subset_clicked, none_clicked` |
| `--seed` | int | 1234 | RNG seed, recorded in the manifest |
| `--max-tries` | int | 60 | Rejection-sampling attempts per patch before giving up |

Everything expensive (`cc3d` connected components, both prompt draws, clicked-subset targets,
click-inside flags) runs **once, offline**, here. Nothing at validation time recomputes or
randomises.

## Scenarios

| Scenario | What it tests | Scored against |
|---|---|---|
| `all_clicked` | Every lesion in the patch clicked | full `seg` (Dice) |
| `subset_clicked` | A strict subset of a multi-lesion patch's lesions clicked | clicked-subset target AND full `seg` — the gap is `val_selectivity_margin` |
| `none_clicked` | Lesion(s) present, nothing clicked | predicted-foreground fraction (correct output is empty; Dice is undefined) |
| `lesion_free_decoy` | No lesion in the patch, one false-positive click | predicted-foreground fraction |

`subset_clicked` requires >=2 lesion instances in a patch, so single-lesion-per-case cohorts
(`d014`, `d016`, `d020`) get **zero** `subset_clicked` patches by construction — their budget is
redistributed to the other 3 scenarios within the same cohort. Per-cohort metrics are therefore
computed over `all_clicked` rows only, so they compare model quality, not scenario difficulty.

## Manifest schema

One JSON file plus a `.targets.npz` sidecar (packed-bit clicked-subset masks, `.json` -> same path
with `.targets.npz`).

| Field | Meaning |
|---|---|
| `bbox` | `[[z0,z1],[y0,y1],[x0,x1]]`, half-open, preprocessed case frame |
| `clicks_zyx` / `clicks2_zyx` | Patch-local coords, post-displacement, draw 1 / draw 2 |
| `n_false_pos` | Trailing decoy count in each click list |
| `size_bucket` | `small`/`large` by the largest in-patch lesion's voxel volume |
| `click_inside` | 1/0/-1, majority vote of draw-1 clicks landing on foreground |
| `subset_target_index` | Row into the `.npz` sidecar, or `-1` |
| `draws_matched` | 1 if draw 1 and draw 2 kept the same lesion-click count (displacement can drop a click from one draw only) |

`cohort_weights` in the header are the **true** case-count proportions (train+val), used to
re-weight headline aggregates and undo the per-cohort `--floor` skew.

## Logged metrics (`--val-manifest` runs only)

Beyond the legacy `val_dice` / `val_prompt_agreement` family (always logged, unchanged formulas):

- `val/{scenario}/*` — full metric set per scenario (`val_dice`, `val_dice_macro`, `val_pred_fg`,
  `val_prompt_agreement`, `val_prompt_agreement_matched`, `val_dice_prompt_ablated`,
  `val_prompt_gap`, `n`).
- `val/subset_clicked/val_dice_vs_clicked_subset`, `val_dice_vs_all_lesions`,
  `val_selectivity_margin` (= the first minus the second; negative means the model ignores the
  click).
- `val/cohort/{name}/val_dice`, `val_prompt_agreement`, `n` — `all_clicked` rows only (D15).
- `val/tag/{click_inside,click_outside,small,large}/val_dice`.
- `val_dice_weighted`, `val_prompt_agreement_weighted` — cohort metrics re-weighted by
  `cohort_weights`.
- `val_prompt_agreement_matched` — global, rows with `draws_matched==1` only.

A bucket with zero rows logs `NaN`, never a fabricated `0.0`.

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `splits_final.json already exists` | `nanounet_build_splits` without `--force` | Add `--force` (old file is backed up) |
| `No validation manifest at <path>` | `--val-manifest` path missing | Run `nanounet_build_valset` first |
| `<path> has schema N, this build expects 1` | Stale manifest format | Rebuild with `nanounet_build_valset` |
| `subset_clicked target ... exceeds available capacity` | `--mix` asks for more selectivity patches than multi-lesion cohorts can supply | Lower the `subset_clicked` share or raise `--n-patches` |
| `Could not fill scenario '...'` | Cohort ran out of eligible patches within `--max-tries` | Lower that scenario's `--mix` share, or raise `--max-tries` |
