# Graph QC viewer

Dash + Cytoscape viewer for one cached patient graph: BL and FU lesion nodes, cross edges, click a node or edge for its
features. Needs the cached graphs from `lesionglue_preprocess`; serves on `127.0.0.1` and prints the URL.

## Command

```bash
lesionglue_qc --case 16a5cdae36 --split val --root /nnunet_data/Longitudinal-CT --port 8050
```

Then open `http://127.0.0.1:8050`. The layout dropdown offers `preset (mm + spread)`, `cose`, `breadthfirst`, `circle`.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--case` | str | required | Patient id to show; a trailing `_NN` image index is stripped |
| `--split` | choice | `val` | `train`, `val` or `test`: which cached split to look the patient up in |
| `--cache` | path | `CACHE_ROOT` (`/nnunet_data/lesion_tracking/cache`) | Cached graph root (output of `lesionglue_preprocess`) |
| `--root` | path | `DATASET_ROOT` (`/nnunet_data/Longitudinal-CT`) | Longitudinal-CT root passed to the graph dataset |
| `--port` | int | 8050 | Local port for the Dash server on `127.0.0.1` |

## Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/lesion_tracking/cache/processed/val_v8_native.pt` | PyG cache | `lesionglue_preprocess` |
| `http://127.0.0.1:8050` | Dash web page (nothing is written to disk) | this step |

## Common errors

| Message starts with | Fix |
|---|---|
| `pid=` | Patient is not in that split: try another `--split`, or rerun `lesionglue_preprocess` for it |
| `No tracking split at` | `lesionglue_split --root /nnunet_data/Longitudinal-CT --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out lesionglue/configs/split.json` |
| `zero graphs for split=` | Split has no patients: rerun `lesionglue_preprocess` |
