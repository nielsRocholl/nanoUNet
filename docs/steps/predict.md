# Predict

Prompt-driven GPU-batched inference over a dataset folder or a single case. Points are native scanner voxels `(x,y,z)`; mapping to preprocessed space is automatic.

Default engine: clustered covering tiles, face-grid expand, max-merge, TTA from `nano_config.json`. `--gt-dir` scores LongiSeg DSC/NSD/LDR from written niftis after export drains.

## Command

Dataset mode (folder of scans + sibling JSON):

```bash
nanounet_predict \
  -i /nnunet_data/Longitudinal-CT/inputsTrFU \
  -o /tmp/preds \
  -m /nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_instance_1200ep \
  --ckpt last.ckpt \
  --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv \
  --gt-dir /nnunet_data/Longitudinal-CT/targetsTrFU \
  --metrics-out /tmp/preds/metrics
```

Single case:

```bash
nanounet_predict -i case.nii.gz -o seg.nii.gz --points case.json \
  -m /path/to/run --ckpt last.ckpt
```

Then track both timepoints (writes `{bl,fu}.nii.gz` + `matches.csv`): [track.md](track.md) / `nanounet_segtrack`.

Centered mode (one patch per click):

```bash
nanounet_predict -i case.nii.gz -o seg.nii.gz --points case.json \
  -m /path/to/run --ckpt last.ckpt --inference-mode centered
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-i`, `--input` | str | (required) | Folder (dataset) or single `.nii.gz` |
| `-o`, `--output` | str | (required) | Output folder (dataset) or single `.nii.gz` |
| `-m`, `--model-dir` | str | (required) | Run dir with `plans.json`, `dataset.json`, `nano_config.json`, checkpoint |
| `--ckpt` | str | auto | Basename or path; auto = `last.ckpt`, tried as-is then `<model-dir>/`, `checkpoints/`, `finetune/` |
| `--ema` | flag | off | Load `callbacks/EMACallback/shadow` from the same `.ckpt` instead of raw `net.*`. Empty/missing shadow is an error. Use a different `-o` than the raw run. |
| `--points` | str | none | Points JSON (**single mode only**) |
| `--no-prompt-encode` | flag | off | Zero the 2 prompt channels |
| `--no-border-expand` | flag | off | Disable large-lesion face-grid expand (on by default) |
| `--max-border-extra` | int | `16` | Max extra grid tiles per click cluster |
| `--tta` / `--disable-tta` | flag | from config | Force test-time augmentation on / off |
| `--batch-size` | int | `8` | GPU patch cap per forward; clamped to free VRAM, never auto-raised |
| `--num-workers` | int | `1` | Prefetch threads. Depth-1: one case on GPU, one preprocessing (32G cgroup) |
| `--cluster-margin-frac` | float | `0.1` | Cluster bbox margin as fraction of patch size |
| `--inference-mode` | choice | `clustered` | `clustered` (covering tiles) \| `centered` (one tile per click) |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` (falls back if unavailable) |
| `--no-amp` | flag | off | Disable autocast (fp32) |
| `--overwrite` | flag | off | Re-run cases whose output exists |
| `--patients-csv` | str | none | CSV `patient` column; keep `<id>_*.nii.gz` whose prefix matches |
| `--gt-dir` | str | none | Instance-labeled native GT folder (same stems as `-i`). Enables scoring. |
| `--metrics-out` | str | none | Write `{stem}.json` and `{stem}.csv`. Requires `--gt-dir`. |

Points JSON format: `{"points": [{"name": "1", "point": [x, y, z]}, ...]}`. Empty `points` → all-background output. `name` is the lesion id (integer); required for `--gt-dir` scoring.

## Engine

Not a sliding window. Clicks pack into the fewest covering tiles (`clustered`). `--inference-mode centered` is the same path with one tile per click. Face-grid expand grows a per-cluster lattice where predicted FG hits a tile face (`--no-border-expand` disables it). Overlaps max-merge. Native `.nii.gz` is a per-tile nearest paste, not a full-volume logit resample.

TTA cat size is probed from free VRAM (no flag). `--batch-size` is a cap the engine may clamp further. Folder mode prefetches **one** next case (depth-1; `--num-workers` default 1 — this Docker cgroup is 32G). Dim `[i/n] case` prints when preprocess **starts**; the green timing line is after GPU+export; with `--gt-dir`, Dice vol / DSC / NSD / LDR print on the next line. Omit `--overwrite` to skip existing preds (those still score immediately).

## Checkpoint selection

`--ckpt last.ckpt` (or any basename) is tried as a path, then `-m/<name>`, `-m/checkpoints/<name>`, `-m/finetune/<name>`. Omit `--ckpt` for `last.ckpt`. Default weights are raw `net.*`. `--ema` loads the EMA shadow from that same file (`last.ckpt` = end-of-run shadow; `best-*.ckpt` = that epoch's shadow). Do not overwrite a raw `-o`. Train with `--ema-decay 0.999` or the shadow is empty.

## Inputs / outputs

**Inputs**

- Model run dir (`plans.json`, `dataset.json`, `nano_config.json`, checkpoint)
- Scans (`.nii.gz`) and points JSON (dataset: sibling `<name>.json`; single: `--points`)
- Optional `--patients-csv` (`patient` column) and `--gt-dir` instance masks (same stems as `-i`)

**Outputs**

- Dataset: `<out>/<case>.nii.gz` per input scan
- Single: `-o` segmentation file
- `--gt-dir`: per-case Dice vol / DSC / NSD / LDR after each export; summary panel at the end. **Dice vol** = whole-volume FG (`P>0` vs `G>0`; empty∩empty → 1). **DSC / NSD (1 mm) / LDR (IoU>0.1)** = named GT instance vs pred cc3d-18 (click voxel if on pred, else the component with max overlap vs that instance). Empty GT dropped; empty pred → 0. Headline is case-mean then mean-over-cases. `--metrics-out` writes `{stem}.json` + `{stem}.csv`

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `missing points JSON for: …` | Dataset mode without sibling JSON | Add `<basename>.json` next to each scan |
| `single mode requires --points` | Single `.nii.gz` without points | Pass `--points` |
| Missing checkpoint | Wrong `--ckpt` or incomplete train | Verify path under `checkpoints/` or `finetune/` |
| `No EMA shadow in checkpoint` | `--ema` on a ckpt with `--ema-decay 0` or no callback block | Drop `--ema`, or train with `--ema-decay 0.999` |
| CUDA unavailable | No GPU | Use `--device cpu` or `mps` |
| `--metrics-out was set without --gt-dir` | `--metrics-out` without scoring GT | Pass `--gt-dir` (instance labels, same stems as `-i`) |
| `GT at '…' looks binary` | Union/binary masks in `--gt-dir` | Use instance-labeled `targetsTrFU` (voxel value = lesion_id) |
| `no cases match --patients-csv` | CSV ids do not match `-i` stem prefixes | Use `test_patients.csv`; ids are `03b90eb112`, not `03b90eb112_00` |
| `Killed` (no traceback) | Host OOM / 32G cgroup; too many inflight volumes | Rerun without `--overwrite` (skips written preds). Default `--num-workers 1` |

## Interactive / embed (library)

Not a CLI flag. Radiom remote interactive session calls these in-process:

| Function | Module | Description |
|----------|--------|-------------|
| `predict_case_logits` | `nanounet.infer.predict.case` | Clustered/centered patch forward → `(logits, tiles)` |
| `patch_logits_to_native_seg` | `nanounet.infer.export.tiles` | Argmax patch → per-tile native paste |
| `native_seg_to_nifti_bytes` | `nanounet.infer.export.tiles` | Gzip NIfTI bytes from native seg + `props["sitk_stuff"]` |

See also [radiom_embed_api.md](../dev-notes/radiom_embed_api.md).
