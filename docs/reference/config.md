# ROI / prompt configuration

Training and inference read a JSON config (default [`configs/default.json`](../../configs/default.json)) parsed by [`nanounet/config.py`](../../nanounet/config.py) into frozen dataclasses.

Passed to supervised training via `nanounet_train --config <path>`. Copied to the run dir as `nano_config.json` for predict.

## Example

```bash
nanounet_train -d 001 -f 0 --plans nnUNetResEncUNetLPlans --config configs/default.json
```

## Top-level sections

| Section | Dataclass | Purpose |
|---------|-----------|---------|
| `prompt` | `PromptConfig` | Point encoding for training / validation |
| `sampling` | `SamplingConfig` | Patch sampling, click modes, large-lesion extras |
| `inference` | `InferenceConfig` | Sliding-window step size, default TTA |
| `validation` | `ValidationConfig` | Optional; fraction of no-lesion validation crops |

---

## `prompt`

| Field | Type | Default in `default.json` | Description |
|-------|------|----------------------------|-------------|
| `point_radius_vox` | int | `2` | Radius of binary / EDT prompt disk in voxels |
| `encoding` | `"binary"` \| `"edt"` | `"edt"` | Prompt channel encoding |
| `validation_use_prompt` | bool | `true` | Apply prompts during validation (not just train) |
| `prompt_intensity_scale` | float | `0.5` | Scale prompt peak; must be in `(0, 1]` |

---

## `sampling`

| Field | Type | Default in `default.json` | Description |
|-------|------|----------------------------|-------------|
| `fg_patch_prob` | float | `0.67` | Probability of foreground-centred patch vs random background |
| `click_modes.pos` | float | `1.0` | Probability of jittered centroid prompt |
| `click_modes.drop` | float | `0.0` | Probability of omitting prompt (no-click training) |
| `n_false_pos` | int or `[min, max]` | `[1, 1]` | Count range of decoy false-positive clicks per patch |
| `false_pos_min_dist_vox` | int | `50` | Minimum voxel distance of decoys from true foreground |
| `false_pos_probability` | float | `0.2` | Probability of adding false-positive decoys |
| `large_lesion.K` | int or `[min, max]` | `2` | Extra centroid samples for large lesions |
| `large_lesion.K_min` | int | `1` | Minimum extra samples |
| `large_lesion.K_max` | int | `4` | Maximum extra samples |
| `large_lesion.max_extra` | int | `0` | Cap on additional large-lesion patches |
| `propagated.sigma_per_axis` | `[sx, sy, sz]` | `[2.75, 5.19, 5.40]` | Gaussian jitter sigmas for propagated clicks (voxels) |
| `propagated.max_vox` | float | `34.0` | Max jitter magnitude (voxels) |

### `click_modes` constraint

**`click_modes.pos + click_modes.drop` must equal `1.0`** (tolerance `1e-5`). Parser raises `ValueError` otherwise.

Example valid modes:

```json
"click_modes": { "pos": 0.8, "drop": 0.2 }
```

There is no separate `neg` mode — drop covers no-prompt training.

If `propagated` is omitted, defaults `(2.75, 5.19, 5.40)` and `max_vox: 34.0` apply.

---

## `inference`

| Field | Type | Default in `default.json` | Description |
|-------|------|----------------------------|-------------|
| `tile_step_size` | float | `0.75` | Sliding-window stride as fraction of patch size |
| `disable_tta_default` | bool | `false` | When `true`, predict disables TTA unless `--tta` passed |

Smaller `tile_step_size` → more overlap → higher compute. Interacts with planner patch size; see [patch_size.md](patch_size.md).

---

## `validation`

| Field | Type | Default if omitted | Description |
|-------|------|-------------------|-------------|
| `no_lesion_frac` | float | `0.3` | Fraction of validation batches sampled from background-only patches |

Must be in `[0, 1]`.

---

## What is *not* in this config

Patch size, batch size, and network topology come from the **plans JSON** produced at preprocess time — not from this file. See [steps/plan.md](../steps/plan.md).

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `click_modes.pos + click_modes.drop must sum to 1` | Invalid mode weights | Adjust so sum is exactly 1 |
| `prompt_intensity_scale in (0,1]` | Scale ≤ 0 or > 1 | Use e.g. `0.5` |
| `encoding` ValueError | Not `binary` or `edt` | Fix typo |
| Config not found | Wrong `--config` path | Relative paths try cwd then repo root |
