# Track (after predict)

Turn binary nanoUNet predictions + click JSON into instance masks, then call `tracking.infer.track`. Seg stays in nanoUNet; this CLI does not re-run predict.

Requires `pip install -e /lesion-tracking`.

## Command

```bash
nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/CASE_00.nii.gz \
  --bl-pred /tmp/preds/CASE_bl.nii.gz --bl-clicks CASE_bl.json \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/CASE_00.nii.gz \
  --fu-pred /tmp/preds/CASE.nii.gz --fu-clicks CASE.json \
  --propagated /nnunet_data/Longitudinal-CT/meta/CASE.csv \
  --track-ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt \
  --decode dense --out /tmp/preds/CASE_matches.csv
```

`--decode` omitted on a TTY → interactive table. Non-TTY must pass `--decode`.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-img` `--fu-img` | path | required | Native CT NIfTI |
| `--bl-pred` `--fu-pred` | path | required | Binary FG predictions from `nanounet_predict` |
| `--bl-clicks` `--fu-clicks` | path | required | nanoUNet click JSON (`points[].name` = lesion_id, `point` = `[x,y,z]`) |
| `--propagated` | path | required | Meta CSV (`cog_propagated`), slim `lesion_id,z,y,x`, or FU-frame click JSON. Not `inputsTrBL` native clicks. |
| `--track-ckpt` | path | required | Lightning matcher ckpt (`h60_r9/best.ckpt`) |
| `--out` | path | required | Match CSV next to preds |
| `--decode` | choice | unset | `dense` / `sinkhorn` / `hungarian` (see `lesion_track --help`) |
| `--thresh` | float | 0.5 | Dense pair cutoff |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` |

Output columns: `bl_lesion_id, fu_lesion_id, pair_prob, decode`.

## Inputs / outputs

**In:** binary pred NIfTIs, click JSON, CTs, propagated centroids, matcher ckpt.

**Out:** match CSV. Instance conversion is in-memory (temp NIfTI); click-on-FG owns the cc3d-18 component.

## Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tracking is not installed` | lesion-tracking not on the env | `pip install -e /lesion-tracking` |
| `No --decode given and stdin is not a TTY` | Batch job omitted decode | `--decode dense` |
| `No checkpoint at …` | Train not finished / wrong path | `--track-ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt` |
