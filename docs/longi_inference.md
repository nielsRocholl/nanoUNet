# Two-stream longitudinal inference

For `--longi` finetune checkpoints, pass the baseline scan and pre-propagation BL centroids
so the model receives a co-located BL stream (identity fallback when a partner is absent).

```bash
nanounet_predict \
  -i followup.nii.gz \
  -o out/pred \
  -m "$MODEL_DIR" \
  --ckpt finetune/best-epoch=412-val_dice_macro=0.6649.ckpt \
  --points fu_points.json \
  --baseline-image baseline.nii.gz \
  --baseline-points bl_partners.json \
  --inference-mode clustered --merge max --border-expand
```

- `--baseline-image`: sibling BL `.nii.gz`, preprocessed with the same `run_case` path as FU.
- `--baseline-points`: JSON with the same `points` list format as `--points`; one BL partner
  per FU prompt (use `null` entries for new lesions with no baseline partner). Length must
  match the FU points list.

Without both flags, inference stays single-stream (compatible with stage-2 checkpoints).
The net auto-detects longi from `net.dwb.*` keys in the checkpoint; pass `--longi` to force it.

External eval (`strat_eval.py`) should supply BL image + pre-propagation BL centroids as partners.
