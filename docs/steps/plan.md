# Planning knobs

The ResEnc planner chooses patch size, batch size, and network topology. These are set at **preprocess** time via `nanounet_preprocess`, not in [configs/default.json](../reference/config.md).

## Command

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncTiny --patch-vol small --gpu-memory-gb 24 -np 4
```

Cluster CPU preprocess, large GPU train:

```bash
nanounet_preprocess -d 001 --planner nnUNetPlannerResEncL --gpu-memory-gb 80 --patch-vol medium -np 8
```

## Arguments (planning-related)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--planner` | str | `nnUNetPlannerResEncL` | Network scale: **Tiny** (~1–5M params) vs **L** / **XL** (8–40+ GB VRAM targets) |
| `--patch-vol` | choice | `large` | Starting isotropic edge before aniso split and VRAM shrink: `small` 128, `medium` 192, `large` 256, `xlarge` 320 |
| `--gpu-memory-gb` | float | none | VRAM budget the planner targets; use the **GPU you train on**, not a random CPU node |
| `--plans-name` | str | auto | Override output plans basename; required with `--skip-plan` |
| `--skip-plan` | flag | off | Reuse existing plans; skip planner step |

Implementation: [`planner_resenc.py`](../../nanounet/plan/resenc/planner_resenc.py) may **shrink** the patch if footprint × network width exceeds VRAM, or **enlarge** if memory allows.

## Inputs / outputs

**Inputs**

- `dataset_fingerprint.json` from fingerprint step
- `--patch-vol` preset and optional `--gpu-memory-gb`

**Outputs**

- `<plans>.json` — definitive patch size, `batch_size`, ResEnc topology for train/predict

## Planner presets

| Planner | Typical use |
|---------|-------------|
| `nnUNetPlannerResEncTiny` | Laptop / smoke tests; pair with `--patch-vol small` |
| `nnUNetPlannerResEncL` | Default cluster training |
| `nnUNetPlannerResEncXL` | Maximum capacity when VRAM allows |

Use the **same planner family** for preprocess and train (`--plans` must match the generated JSON basename).

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| Plans / train mismatch | Trained with different planner than preprocess | Re-preprocess or pass matching `--plans` basename |
| Patch smaller than expected | VRAM shrink loop after `--patch-vol` | Raise `--gpu-memory-gb` or accept planner output; see [patch_size.md](../reference/patch_size.md) |
| `--skip-plan` without name | Missing `--plans-name` | Pass existing plans basename |

## Further reading

- [Patch size playbook](../reference/patch_size.md) — FOV vs lesion scale, dual-scale cohorts, inference tile overlap
