# Plan — Step 6: instance-conditional targets

**Audience:** a coding agent with no access to the session that produced this. Read
`docs/handoffs/HANDOFF_training_overhaul.md` §Step 6 (the parent spec) and
`docs/handoffs/HANDOFF_step1_done_step6_next.md` (session state, diagnosis, benchmark) first.

**Job in one sentence:** make the training target a function of the click — foreground only for
lesion instances that received a click — so "no click ⇒ no segmentation" is trained rather than
hoped for.

**Why now:** measured on the current checkpoint, `val/subset_clicked/val_selectivity_margin` is
**−0.2709**. Click one of three lesions and the output matches "segment all three" 27 Dice points
better than "segment the one you clicked". The model does not treat the click as selecting a lesion.

**Style:** `.claude/skills/nanochat-style` is binding. <200 LOC per file, no `utils/`, no ABCs, no
factories, errors that name the fix, no fallbacks for missing data.

---

## 0. Decisions already made — do not reopen

| # | Decision | Source |
|---|---|---|
| S1 | Unclicked instances map to **background**, never ignore-label. | Parent handoff, settled. Ignore-label would exclude those voxels from the loss and never teach suppression. |
| S2 | Click dropout **`pos = 0.8`** (20% of in-patch annotated lesions go unclicked). | Human, this session. |
| S3 | The consistency term is **re-scoped, not dropped**: kept set drawn once per patch, only jitter varies. | Human, this session. |
| S4 | Instance resolution is **`cc3d` on the crop at sample time** (parent handoff option 3). | Benchmarked: 5.7 ms mean / 8.0 ms p95 = 1.2% of the 477 ms `build_patch` baseline, 0.24% of the per-patch CPU budget at 16 workers. Exact, no extra storage, no re-preprocessing. |
| S5 | Enabled by a **config flag**, default off, so `configs/default.json` runs stay bit-identical. | Old runs must remain reproducible. |

### Two findings from reading the code that change the parent spec

**F1 — the parent handoff's change #2 ("per-variant targets") is NOT needed.** It was listed
because "once the kept click set varies per draw, each variant needs its own target". But change #4
(S3 above) draws the kept set **once per patch**, so both variants of a pair share one target.
`collate_patches` keeps reusing a single target per item and needs **no structural change**.
Step 6 is three coupled changes, not four. Do not restructure `collate_patches`.

**F2 — no config schema change is needed.** `nanounet/config.py:115` already enforces
`click_modes.pos + click_modes.drop == 1`, so `pos = 0.8` simply means `drop = 0.2`. The
"dead branch" at `sampling.py:54` is gated on `cm.drop == 0.0`, which is exactly `pos == 1.0`.
Setting the two values enables it. Do not invent a new dropout field.

---

## 1. The mechanism

For one training patch, in `build_patch`:

1. Crop as today → `seg_crop` (binary, `-1` outside the volume).
2. **`cc3d` the crop once**: `lab, n = cc3d.connected_components((seg_crop > 0).astype(np.uint8), connectivity=26, return_N=True)`.
   Connectivity 26 matches `nanounet/prompt/centroids.py` (verified: it is `cc3d`'s 3D default).
3. `in_patch` = indices of **undisplaced** centroids inside the patch, via
   `filter_centroids_in_patch`. Undisplaced on purpose — the kept set must not depend on the random
   displacement, or the two variants would disagree.
4. **Draw the kept set ONCE per patch**: each `j in in_patch` is kept with probability
   `cfg.sampling.click_modes.pos`.
5. Map kept lesions to crop-local instance labels: for lesion `j`, look up `lab` at its
   **`seed_zyx[j]`** (patch-local), and at `centroids_zyx[j]` as a second probe. Collect every
   non-zero label found. Use the seed first: the plain centroid falls **outside its own lesion ~12%
   of the time** on concave shapes, which is precisely why `seed_zyx` (argmax-EDT, guaranteed
   interior) exists — see `centroids.py:1-7`.
6. **Target** = `np.isin(lab, kept_labels)`, cast back to the seg dtype, `-1` padding preserved
   where `seg_crop == -1` so `RemoveLabelTansform` still behaves.
7. Clicks: for each variant, displace **only the kept centroids** and filter into the patch, exactly
   as today. The shared decoy is unchanged.

### Why components with no matching lesion become background — and that is correct

A lesion whose centroid lies outside the patch is never clicked (`filter_centroids_in_patch` drops
it), so under the project rule it must be background. Its voxels form a `cc3d` component in the crop
that no kept lesion maps to, and step 6 excludes it. That is the rule working, not a bug.

### Why a click displaced out of the patch keeps its target

The kept set is drawn before displacement (step 3/4). If displacement pushes a kept lesion's click
outside the patch, the lesion stays **foreground** with no visible click. This is deliberate:

- it is the deployment-realistic case already measured by `val_dice_click_outside` (0.759 today);
- it keeps the pair's targets identical, which is what makes the re-scoped consistency term correct.

---

## 2. Files

| # | Path | Kind |
|---|---|---|
| 1 | `nanounet/data/instance_target.py` | **new** — cc3d, kept-set draw, target masking (~90 LOC) |
| 2 | `nanounet/data/sampling.py` | edit — call it from `build_patch`, pass the kept set to variants |
| 3 | `nanounet/config.py` | edit — `instance_targets: bool` on `SamplingConfig`, default `False` |
| 4 | `configs/instance_conditional.json` | **new** — `pos 0.8 / drop 0.2`, `instance_targets true` |
| 5 | `docs/steps/train.md` | edit — document the config field |
| 6 | `docs/reference/instance_targets.md` | **new** — the rule, the mechanism, the metrics to watch |

`patch_render.py`, `patch_iterable.py`, `lightning_module.py`, `losses.py`, `data_module.py`,
`fit.py`, and the whole validation path are **untouched**. If you find yourself editing one, stop:
either you have hit something this plan got wrong (report it) or you are doing more than the job.

---

## 3. `nanounet/data/instance_target.py`

```python
"""Click-conditional targets: foreground only for lesion instances that received a click.

The seg on disk is binary, so instance identity is recovered by cc3d on the CROP -- 5.7 ms mean
against a 477 ms build_patch baseline (measured, see HANDOFF_step1_done_step6_next.md), i.e. inside
the IO noise the path already carries. Full-volume connected components would not be.

The kept set is drawn ONCE per patch, before displacement, so every prompt variant of a patch shares
one target: that is what keeps the consistency term measuring click PLACEMENT rather than penalising
two variants for legitimately disagreeing about which lesion they were pointed at."""
```

Public surface — two functions, nothing else:

```python
def draw_kept(in_patch: list[int], keep_prob: float, rng) -> list[int]:
    """Which in-patch lesions keep their click. Drawn once per patch, on UNDISPLACED centroids, so
    the choice cannot depend on the random displacement (which differs per variant)."""
    if keep_prob >= 1.0:
        return list(in_patch)
    return [j for j in in_patch if rng.random() < keep_prob]


def clicked_target(seg_crop, lab, kept, seeds_local, cts_local) -> np.ndarray:
    """seg_crop masked down to the cc3d components of `kept`. Components no kept lesion maps to
    become background -- correct under the project rule: a lesion whose centroid is outside the
    patch never receives a click, so it must not be segmented. -1 padding is preserved."""
```

`clicked_target` implementation notes:

- Probe order per kept lesion `j`: `seeds_local[j]` first, then `cts_local[j]`. Accept any non-zero
  `lab` value found; a lesion may legitimately contribute zero labels if both probes fall outside
  the crop, and that is not an error.
- Build `out = np.where(np.isin(lab, kept_labels), 1, 0).astype(seg_crop.dtype)`, then restore
  padding: `out[seg_crop < 0] = -1`.
- `kept_labels` as a Python `set` → `np.isin(lab, list(kept_labels))`. With `n` typically 1–3 per
  crop (measured mean 1.5) this is trivial.
- Bounds-check probes with a plain `if 0 <= z < D and ...`; no try/except (R5).

The `cc3d` call itself lives in `build_patch` (it needs `seg_crop`, which `build_patch` owns) and is
passed in as `lab`. Keeping `instance_target.py` free of IO makes it directly testable.

---

## 4. `nanounet/data/sampling.py` — the edits

`build_patch` is 168 LOC today; these edits add ~18. If it would exceed 200, move the cc3d call and
the local-coordinate conversion into `instance_target.py` as a third function — do **not** exceed
the limit.

**4a.** After `data_crop, seg_crop, _patch_shape, pslc = crop_patch(...)` and after `fp = draw_false_pos(...)`:

```python
    seg_out = seg_crop
    kept = None
    if cfg.sampling.instance_targets:
        m = (seg_crop[0] if seg_crop.ndim == 4 else seg_crop)
        lab, _n = cc3d.connected_components((m > 0).astype(np.uint8), connectivity=26, return_N=True)
        in_patch = [j for j, c in enumerate(cts_global) if _inside(c, pslc)]
        # ONE kept-set draw for the whole patch: every variant must share the target, else the
        # consistency term would penalise two variants for holding different (correct) answers.
        kept = draw_kept(in_patch, cfg.sampling.click_modes.pos, rng)
        seeds = properties.get("seed_zyx") or cts_global
        seg_out = clicked_target(seg_crop, lab, kept, _to_local(seeds, pslc), _to_local(cts_global, pslc))
```

`_inside(c, pslc)` and `_to_local(pts, pslc)` are two 2-line module-level helpers in `sampling.py`;
`filter_centroids_in_patch` already returns patch-local coordinates but discards which index each
came from, which is exactly what step 5 needs — hence the local helpers rather than reuse.

**4b.** Pass `kept` through to the variants so only kept lesions are clicked:

```python
    variants = [
        points_variant(seg_crop, cts_global, pslc, cfg, force_zero_prompt, rng, True, volumes_vox, fp, kept)
        for _ in range(prompts_per_patch)
    ]
```

and the same for the `extra_rng` diagnostic variant.

**4c.** `points_variant` and `select_prompt_points` gain a trailing `kept: list[int] | None = None`.
In `select_prompt_points`, replace the dropout line:

```python
        inch = filter_centroids_in_patch(displaced, pslc)
        cm = cfg.sampling.click_modes
        kept_ = inch if cm.drop == 0.0 else [p for p in inch if rng.random() < cm.pos]
```

with a version that honours a caller-supplied set. When `kept` is given, the per-variant random draw
must **not** run — that is the whole point of S3:

```python
        if kept is None:
            inch = filter_centroids_in_patch(displaced, pslc)
            cm = cfg.sampling.click_modes
            kept_ = inch if cm.drop == 0.0 else [p for p in inch if rng.random() < cm.pos]
        else:
            # Kept set fixed per patch (instance_targets); displace ONLY those, then filter.
            kept_ = filter_centroids_in_patch([displaced[j] for j in kept], pslc)
```

Note `displaced` is indexed by the original centroid index, so `kept` indexes into it directly.

**4d.** Return `seg_out` instead of `seg_crop` as `"segmentation"`. `seg_crop` is still what is
passed to `draw_false_pos` and `click_inside_flags` — the decoy must avoid **real** foreground, and
the click-inside flag must report whether the click landed on a real lesion, not on the masked
target. Getting this backwards silently corrupts both.

**4e.** Update the module docstring: one sentence saying the target is click-conditional when
`sampling.instance_targets` is set, and that the kept set is drawn once per patch.

---

## 5. Config

**5a.** `nanounet/config.py`: add `instance_targets: bool = False` to `SamplingConfig`, loaded with
`bool(d.get("instance_targets", False))`. Absent ⇒ current behaviour, exactly.

**5b.** New `configs/instance_conditional.json` — a copy of `configs/default.json` with:

```json
  "sampling": {
    "fg_patch_prob": 0.67,
    "instance_targets": true,
    "click_modes": { "pos": 0.8, "drop": 0.2 },
    ...
  }
```

Everything else identical, including the `propagated` block and its absolute `error_table` path.

**Do not edit `configs/default.json`.** The 600-epoch baseline must stay reproducible.

---

## 6. Verification

### 6a. Correctness — before any training

Throwaway scripts in the scratchpad, deleted after (R16). Report numbers.

| # | Check |
|---|---|
| C1 | With `instance_targets` absent, `build_patch` output is **byte-identical** to the current code for a fixed seed, over ≥200 patches. This is the no-regression gate. |
| C2 | With `pos = 1.0` and `instance_targets true`, the target equals `seg_crop` wherever the seg has lesions whose centroid is in the patch — i.e. masking is a no-op when nothing is dropped, except for components no centroid maps to. Report how often that exception fires. |
| C3 | With `pos = 0.8`, over ≥500 patches: measure the realised fraction of in-patch lesions that keep their click. Must be 0.80 ± 0.02. |
| C4 | Both variants of a pair have **identical targets**, over ≥500 patches. If this ever fails, S3 is broken and the consistency term is now wrong. |
| C5 | Target foreground ⊆ `seg_crop > 0`, always. The mask may only ever remove voxels, never add. |
| C6 | Patches exist where a visible lesion is background in the target (that is the new training signal). Report the fraction — expect roughly `1 - 0.8 = 0.2` of in-patch lesions. |

### 6b. Throughput — the hard gate

**Average GPU utilisation must stay above 95%.** Measure before and after, same box, same bucket,
same batch size, ≥3 epochs, using `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` and
`epoch_wall_time_sec`. The benchmark predicts +1.2% on the per-patch path; confirm rather than
assume. If it drops below 95%, stop and report the measurement.

### 6c. The training run — what success looks like

Judge **only** on the Step 1 manifest strata. Baselines from the current checkpoint:

| Metric | Now | Target |
|---|---|---|
| `val/subset_clicked/val_selectivity_margin` | **−0.2709** | **positive** |
| `val/none_clicked/val_pred_fg` | 0.0196 | **→ 0** |
| `val_prompt_gap` | 0.0819 | **higher** — the click matters more |
| `val/all_clicked/val_dice` | 0.8390 | hold, do not regress |
| `val/lesion_free_decoy/val_pred_fg` | 0.0018 | hold |

**Training loss will rise relative to the old objective. That is expected, not a regression.** The
same lesion is now foreground in one patch and background in another; the task stays well-posed
because the input differs (the click channel), but the shortcut the model was using is gone.

Watch `val/all_clicked/val_dice` for the failure mode that matters: if suppression is over-learned,
it shows up as false negatives there while `none_clicked` looks great.

---

## 7. Do not do

- Do not restructure `collate_patches` for per-variant targets (F1 — not needed, and it would break
  the pairing the consistency term relies on).
- Do not add a new dropout config field (F2 — `pos`/`drop` already exist and are validated).
- Do not use ignore-label for unclicked instances (S1, settled).
- Do not let the kept set vary per variant (S3) — it silently makes the consistency term penalise
  correct behaviour, and nothing crashes.
- Do not run `cc3d` on the full volume, only on the crop.
- Do not pass the masked target where the real seg is required: `draw_false_pos` and
  `click_inside_flags` both need the **real** `seg_crop` (4d).
- Do not edit `configs/default.json`, the validation path, or the manifest.
- Do not create a permanent `tests/` folder (R16).
- If a file or field this plan describes does not exist or differs, **stop and report what you
  found**. Both previous plans in this series had a real error caught this way.
