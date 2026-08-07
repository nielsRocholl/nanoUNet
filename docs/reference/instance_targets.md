# Instance-conditional targets

Click-conditional segmentation targets: **foreground only for lesion instances that received a
click.** Off by default; enabled per-config, not per-flag.

## The rule

> Segment a lesion **if and only if** it has a corresponding click. The click is the hint.

| Lesion in patch | Click on it | Target |
|---|---|---|
| annotated | yes | foreground |
| annotated | no | **background** |
| tissue in the patch, lesion clicked elsewhere | yes (click moved into the patch) | **foreground** |
| nothing there (decoy click) | yes | background |

Unannotated lesions need no special handling: they are not in the segmentation at all, and the
"no click ⇒ background" clause covers them.

## Why

Measured on the 600-epoch checkpoint against the fixed validation manifest:

```
val/subset_clicked/val_dice_vs_all_lesions      0.7382
val/subset_clicked/val_dice_vs_clicked_subset   0.4673
val/subset_clicked/val_selectivity_margin      -0.2709
```

Click one of three lesions and the prediction matches *"segment everything"* 27 Dice points better
than *"segment what you pointed at"*. The old target — every annotated lesion in the patch,
regardless of clicks — never penalised that, so the shortcut was free.

## Enabling it

```json
"sampling": {
  "instance_targets": true,
  "click_modes": { "pos": 0.8, "drop": 0.2 }
}
```

```bash
nanounet_train -d 999 --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --config configs/instance_conditional.json \
  --val-manifest /nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/valset_1500.json \
  --val-every-n-epochs 2 --prompts-per-patch 2 --consistency-weight 0.02
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sampling.instance_targets` | bool | `false` | Mask the target down to clicked instances. Absent ⇒ current behaviour, byte-identical |
| `sampling.click_modes.pos` | float | `1.0` | Probability a lesion present in the patch keeps its click. `0.8` drops 20% of lesions, which is now also 20% of the suppression signal -- boundary clipping no longer contributes |
| `sampling.click_modes.drop` | float | `0.0` | Must satisfy `pos + drop == 1` (validated in `config.py`) |

`configs/default.json` is deliberately untouched, so the 600-epoch baseline stays reproducible.

## Mechanism

Per training patch, in `build_patch` (`nanounet/data/sampling.py`) via
`nanounet/data/instance_target.py`:

1. `cc3d.connected_components` on the **crop** (not the volume) -> instance labels.
2. Each component is mapped back to its **parent lesion** via the sidecar `bboxes_zyx`. A crop can
   split one lesion into several components, and every component is a subset of exactly one lesion
   (lesions are disjoint), so bbox containment of the component centroid recovers the parent, with
   nearest-lesion-centroid as the tie-break.
3. A lesion is **present in the patch if any of its voxels are** -- NOT if its centroid is. This is
   the load-bearing rule; see the two consequences below.
4. **Draw the kept set once per patch**, each present lesion kept with probability
   `click_modes.pos`. Drawn before displacement, so it cannot differ between prompt variants.
5. Every kept lesion **always gets a click**. If displacement threw its click out of the patch, the
   click is placed on the centroid of that lesion's largest component in this crop.
6. Target = the components of kept lesions only. `-1` padding preserved.

Cost: `cc3d` on a 96×160×160 crop is **5.7 ms** mean, ~1.9% of the IO-dominated per-patch budget.
Full-volume connected components would not be affordable.

### Two consequences worth knowing

**A kept lesion is never left unclicked, and never becomes background by accident.** Both rules
above exist because training previously contradicted inference:

- `nanounet/infer/border_expand.py` finishes a lesion that spans several patches by placing another
  patch wherever the prediction **touches a patch face**. That only fires if the model segments up
  to the edge. Centroid-based membership trained the opposite, suppressing **13.69% of all
  foreground voxels** (measured) and teaching "ignore anything near the patch boundary".
- `nanounet/infer/longi_row.py:37-39` clamps a click into any patch that contains none, so a patch
  with no in-bounds click still gets a prompt at inference. Dropping the click during training made
  that same situation mean "background".

After the fix, at `pos = 1.0` the target is **identical to the old all-lesion objective**
(0.0000 of foreground removed, verified over 250 patches) and **0 / 170** patches with foreground
have zero lesion clicks. The only tissue removed is what `click_modes.pos` deliberately removes.

## Interaction with the consistency term

`--consistency-weight` rewards two prompt draws producing the same mask. Under the **old** target
that had a free degenerate solution: ignore the click, and the term is satisfied at zero cost while
the segmentation loss is indifferent. That is the leading explanation for the current model's high
prompt agreement (0.87) alongside its −0.27 selectivity.

Instance-conditional targets close it — a click-ignoring model is now catastrophically wrong on the
segmentation loss. The term stays **re-scoped, not dropped**: the kept set is drawn once per patch,
so both variants share one target and the term measures click *placement* robustness only.

**Never let the kept set vary per variant.** Their correct outputs would differ, the term would
penalise correct behaviour, and nothing would crash.

## Metrics to watch

| Metric | Baseline | Target |
|---|---|---|
| `val/subset_clicked/val_selectivity_margin` | −0.2709 | **positive** |
| `val/none_clicked/val_pred_fg` | 0.0196 | → 0 |
| `val_prompt_gap` | 0.0819 | higher — the click matters more |
| `val/all_clicked/val_dice` | 0.8390 | hold, do not regress |
| `val/lesion_free_decoy/val_pred_fg` | 0.0018 | hold |

**Training loss will rise relative to the old objective. That is expected, not a regression.** The
same lesion is foreground in one patch and background in another; the task stays well-posed because
the input differs (the click channel), but the shortcut is gone.

The failure mode to watch for is over-suppression: it shows up as false negatives on
`val/all_clicked/val_dice` while `none_clicked` looks excellent.

See `docs/handoffs/PLAN_step6_instance_targets.md` for the full specification and acceptance checks.
