# Planning knobs

Patch size, batch size, and network topology are set at **preprocess** time via `nanounet_preprocess`, not in [nanounet/configs/default.json](../reference/config.md).

**Foundation mode (default) bypasses the planner.** The nnFoundationCNN checkpoint fixes the ResEnc-L topology
(6 stages, 3×3×3 kernels) and its recommended 192³ patch, so the plan is written from constants
([`foundation_plan.py`](../../plan/resenc/foundation_plan.py)): normalization is per-image Z-score, `batch_size` comes
from the VRAM estimator at `--gpu-memory-gb` rounded down to even (141 GB → 12), and `spacing_mode: z_only` makes
each case's thickest axis resample to `--target-z` mm (rule: `argmax(spacing)` if it exceeds 1.25× axis 0, else axis 0).
The plan's `spacing` is only a nominal median; the real per-case spacing is stored in each case's properties
(`spacing_after_resampling`). `--planner`, `--patch-vol` and the patch-shrink loop apply only with `--no-foundation`.

## Command

```bash
nanounet_preprocess -d 001 --no-foundation --planner nnUNetPlannerResEncTiny --patch-vol small --gpu-memory-gb 24 -np 4
```

Cluster CPU preprocess, large GPU train:

```bash
nanounet_preprocess -d 001 --no-foundation --planner nnUNetPlannerResEncL --gpu-memory-gb 80 --patch-vol medium -np 8
```

## Arguments (planning-related)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no-foundation` | flag | off | Plan with the ResEnc planner at the dataset median spacing instead of the fixed nnFoundationCNN plan |
| `--target-z` | float | `1.0` | Foundation mode: target mm of each case's thickest axis |
| `--planner` | str | `nnUNetPlannerResEncL` | Network scale: **Tiny** (~1–5M params) vs **L** / **XL** (8–40+ GB VRAM targets) |
| `--patch-vol` | choice | `large` | Starting isotropic edge before aniso split and VRAM shrink: `small` 128, `medium` 192, `large` 256, `xlarge` 320 |
| `--gpu-memory-gb` | float | none | VRAM budget the planner targets; use the **GPU you train on**, not a random CPU node |
| `--plans-name` | str | auto | Override output plans basename; required with `--skip-plan` |
| `--skip-plan` | flag | off | Reuse existing plans; skip planner step |

Implementation: [`planner_resenc.py`](../../plan/resenc/planner_resenc.py) may **shrink** the patch if footprint × network width exceeds VRAM, or **enlarge** if memory allows.

## Inputs / outputs

**Inputs**

- `dataset_fingerprint.json` from fingerprint step
- `--patch-vol` preset and optional `--gpu-memory-gb`

**Outputs**

- `<plans>.json` — definitive patch size, `batch_size`, ResEnc topology for train/predict

## Planner presets (`--no-foundation` only)

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
