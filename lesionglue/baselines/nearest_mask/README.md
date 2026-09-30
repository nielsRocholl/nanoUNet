# Nearest-Mask Distance Baseline

This is an isolated raw-data baseline. It does not use cached PyG graphs, model checkpoints, or code under `lesionglue/cli`.

For every eligible baseline lesion, it queries the nearest foreground voxel in the follow-up instance mask for the same `img_id_fu`. The predicted match is the instance label of that closest voxel. Multiple baseline lesions may choose the same follow-up lesion. The baseline predicts `-1` only when the matching follow-up mask volume has no candidate foreground instances.

## Usage

```bash
.venv/bin/python lesionglue/baselines/nearest_mask/run.py \
  --root "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/Longitudinal_CT_v2" \
  --split val \
  --out lesionglue/baselines/nearest_mask/outputs/val
```

Options:

- `--split train|val|test|all`
- `--limit N` for a smoke test on the first `N` patients in each split
- `--graph-compatible` to filter each patient to the dominant `img_id_fu`, matching the current graph preprocessing behavior

## Output

`rows.csv` contains one row per eligible baseline lesion:

- `pid`
- `img_id_fu`
- `bl_lesion_id`
- `topology`
- `gt_fu_lesion_id`
- `pred_fu_lesion_id`
- `distance_mm`
- `correct`
- `status`

`summary.json` and `summary.csv` contain split-level metrics:

- `linkable_acc`
- `disappeared_acc`
- `newly_appearing_acc`
- `row_acc_all_bl`
- `match_score`
- distance summaries for correct and incorrect predictions


Real-data smoke test:

```bash
.venv/bin/python lesionglue/baselines/nearest_mask/run.py \
  --root "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/Longitudinal_CT_v2" \
  --split val \
  --limit 1 \
  --out /tmp/nearest_mask_baseline
```
