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
| `false_pos_probability` | float | `0.05` | Probability of adding a false-positive decoy click |
| `large_lesion.K` | int or `[min, max]` | `2` | Extra centroid samples for large lesions |
| `large_lesion.K_min` | int | `1` | Minimum extra samples |
| `large_lesion.K_max` | int | `4` | Maximum extra samples |
| `large_lesion.max_extra` | int | `0` | Cap on additional large-lesion patches |
| `propagated.mode` | `"gaussian"` \| `"empirical"` | `"empirical"` | How the propagated-click offset is drawn |
| `propagated.error_table` | str | `/nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json` | Path to the measured registration-error table (mode=`empirical` only) |
| `propagated.backends` | `[str, ...]` | `["original", "unigradicon"]` | Registration backends to draw offsets from (mode=`empirical`) |
| `propagated.sigma_per_axis` | `[sz, sy, sx]` | `[5.95, 6.39, 5.93]` | Gaussian jitter sigmas, mode=`gaussian` (voxels) |
| `propagated.max_vox` | float | `34.0` | Max jitter magnitude, mode=`gaussian` only (voxels) |

### `false_pos_probability`

A synthetic false-positive decoy models a spurious click on empty tissue, which does not occur at
deployment — detection is handled upstream and every click refers to a real lesion. The genuine
negative is the disappeared lesion (35% of propagated clicks), which is already present in the data
via the empirical click model. `false_pos_probability` is kept small (`0.05`) only as a residual
robustness margin: 42% of real lesions have a neighbour closer than 30 voxels, so any decoy sampled
far from foreground sits further away than a typical real neighbour ever would.

### `click_modes` constraint

**`click_modes.pos + click_modes.drop` must equal `1.0`** (tolerance `1e-5`). Parser raises `ValueError` otherwise.

Example valid modes:

```json
"click_modes": { "pos": 0.8, "drop": 0.2 }
```

There is no separate `neg` mode — drop covers no-prompt training.

### `propagated` modes

`mode: "empirical"` (default) draws real registration-error offsets from the measured table,
size-matched to each lesion's equivalent-sphere diameter (`volume_vox` from the centroid sidecar).
No magnitude clip -- the table is already outlier-filtered. At startup the table is validated to
exist, parse, and have a non-empty offset pool for every `(size bin, backend)` pair in `backends`;
otherwise the config load raises naming the fix: `python3 scripts/measure_registration_error.py`.

`mode: "gaussian"` keeps the legacy Gaussian jitter (`sigma_per_axis`, clipped to `max_vox`).

If `propagated` is omitted entirely, `mode: "empirical"` with the defaults above applies.

---

## `inference`

| Field | Type | Default in `default.json` | Description |
|-------|------|----------------------------|-------------|
| `tile_step_size` | float | `0.5` | Expand-grid stride as a fraction of patch size (0.5 = half-patch neighbours) |
| `disable_tta_default` | bool | `false` | When `true`, predict disables TTA unless `--tta` passed |

Smaller `tile_step_size` → more overlap → more expand tiles. Not a sliding-window over the volume; see [predict.md](../steps/predict.md).

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
