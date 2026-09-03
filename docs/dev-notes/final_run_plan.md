# nanoUNet Dataset900 final run — handoff plan

**Status:** approved, not implemented. Rewritten 2026-09-03 after verifying every claim against
current code, Dataset900 on disk, and the Claude Code session that produced it.
**Audience:** an agent with no prior context. Do not re-litigate §1 or §9. Do not edit until you
have read §0. Do not submit the SLURM job until the user says so.

**End deliverable of this work:** one submittable SLURM script whose commands (except the node-local
data copy) have been executed on this machine without error. Code + configs land in `/nanoUNet`;
the script lands at
`/home/nielsrocholl/SLURM/jobs/nanoUNet/prompt-robustness/slurm_final_900_h200.sh`.

---

## 0. Mission, constraints, non-goals

**Product.** Promptable 3D CT lesion segmentation for a longitudinal workflow: baseline clicks are
propagated by registration onto follow-up and condition the U-Net. The click is often wrong (~52%
land inside their lesion). The model must be selective (segment the clicked lesion, not its
neighbours), consistent under click jitter, and not drown small mets.

**This run.** One *script* on one H200: **SSL → supervised 1200 ep → short mixed finetune**.
Copying Dataset900 onto the node takes ~a day, so stages share one staged copy **per job**.
qos=vram wall is 7 days; the last 1200-ep supervised run alone was **~8 days** (wandb
`ekkxcgi6` `_wandb.runtime` 689776 s; last `epoch_wall_time_sec` 619.6). A single 7-day allocation
cannot finish MAE+1200+FT. The script must resume from NFS checkpoints; each resubmit re-stages
`/root` (ephemeral). One-copy only happens if the user gets a longer wall clock — ask before
raising `--time` past 7 days. No ablation budget.

**Do**

- Implement the code/config changes in §2–§5, in that order (channel count before any SSL).
- Battle-test every command except `rclone` copy (§7).
- Write the SLURM script in §8. Push to `origin/main` during work — the training container is
  ephemeral and sees Git, not your dirty tree.

**Do not**

- Train the longitudinal / DWT / DWB model. No `--longi`. No Dataset114. Still **update** the
  longi channel-count sites so inference and a future longi warm-start do not silently disagree
  with the new 2-channel supervised net.
- Reuse the Dataset999 MAE (`.../mae_pretrain/checkpoints/last.ckpt`, 250 ep, `val_recon_loss`
  0.1070). It was trained under the old case-level split and saw other timepoints of val patients.
- Re-preprocess, re-plan, or rebuild splits. Dataset900 is done (see below).
- Use `configs/longrun.json` (Dataset999 cohort keys: has `d010`, missing `d026,d028–d031` —
  `CohortSampler` will raise on 900).
- Use `configs/finetune_d013.json` (`click_modes.pos=1.0` — that is the old semantic objective).
- Use `--only-prefix`. A 500-ep d013-only finetune specialised (`val_dice` 0.70→0.61). A short
  80-ep d013-only run (wandb `i4q2oamw`) *did* help small-tag and selectivity; the mixed-FT recipe
  in §5b keeps that benefit and the other 20 cohorts as regularisers.
- Use `--loss cc_dc_ce`. Default `--loss dc_ce`.
- Change the prompt encoding (ball, EDT, `point_radius_vox=2`, `prompt_intensity_scale=0.5`).
- Commit unless the user asks. Notify when the SLURM script is written and commands have been
  smoke-tested.

**House rules.** `nanochat-style` skill: every touched file stays **<200 LOC** (current counts:
`lightning_module.py` 196, `config.py` 194, `val_metrics.py` 150, `fit.py` 213, `instance_target.py`
147, `patch_render.py` 136). Fail loud, no silent fallbacks, no `utils/`. Delete throwaway
verification scripts once they pass.

**Node.** Interactive session right now: H200, ~128 GB RAM, 38 CPUs — enough to smoke-test, not
enough for `--dl-bucket xl`. The submitted job requests the proven H200 layout (64 CPU, 200 GB)
from `slurm_supervised_999_h200.sh`. User said more CPUs can be reserved for the real run.

**Env (container sets none of these):**

```bash
export NANOUNET_RAW=/nnunet_data/NanoUNet_raw
export NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results
```

This host’s current process env points at **`/nanounet_data` (does not exist)**. Data is
`/nnunet_data` (two n’s). Smokes that skip the export will fail before they start.

Repo `/nanoUNet`, editable install; `/usr/local/bin/nanounet_*` resolve here.

### Dataset900_Merged — already built, do not re-investigate

```
path           /nnunet_data/NanoUNet_preprocessed/Dataset900_Merged
cases          5690 centroid sidecars (spot-check 30/30 have volume_vox + bboxes_zyx)
split          splits_final.json  ONE fold: 4833 train / 857 val, 0 id overlap
plans          nnUNetResEncUNetLPlans_h200_smallpv.json
data_identifier nnUNetPlans_3d_fullres          # NOT the plans name
patch          [64, 192, 192]   plans batch 10   spacing [2.5, 0.758, 0.768]
batch_dice     false
on disk        332 GB (331 GB in data_identifier/)
cohorts.json   site_balanced, 21 prefixes d011–d031, d013 weight 0.090909
raw            /nnunet_data/NanoUNet_raw/Dataset900_Merged/dataset.json  (numTraining 5690)
```

Preprocessed `dataset.json` still says `numTraining` **5796** (pre-quarantine). Ignore it.
`nanounet_train` reads RAW (`train.py` `dj_path = join(raw_dir(), ds, "dataset.json")`). Split
keys match the 5690 files on disk (0 missing). Do not copy the stale preprocessed json into the
stage; rclone includes below do not.

Already done previously: patient-grouped split (fixes leaked d013 + d029 patients), quarantine of
106 empty-label d026 cases, 64 genuine zero-lesion cases **kept**. Do not redo.

**Missing, you must create before submit:** valset, d013 `*_weights.json` (0 today; 999 had 426/537
d013). No Dataset900 MAE — that is stage 1 of the job.

---

## 1. Evidence base — why these changes, and only these

Numbers are from `metrics.csv` of
`Dataset999_Merged_..._h200_instance_1200ep` (wandb `ekkxcgi6`, 1200 ep) and a sampler simulation
on Dataset900 sidecars. Do not re-derive. Do not contradict without new measurement.

**A — the objective teaches the model to ignore the click.** Per fg patch, `click_modes.pos=0.8`:

| row teaches | share |
|---|---|
| segment every in-patch lesion (all kept) | **70.2%** |
| segment this one, not that one (≥1 kept AND ≥1 dropped) | **15.3%** |
| predict nothing (all dropped) | 14.5% |

**B — pooled `val_dice` is anti-correlated with selectivity, r = −0.571** (ep>200).
`--monitor val_dice` picked ep 1005, the **worst-selectivity checkpoint of the run**.

```
epoch   val_dice   small   selectivity   vs_subset   prompt_gap
   79     0.454    0.542      +0.122       0.455       0.076
  395     0.615    0.631      -0.159       0.298       0.228
 1005     0.712    0.667      -0.257       0.377       0.121   ← monitor
 1199     0.690    0.619      -0.231       0.363       0.130
```

`val/subset_clicked/val_selectivity_margin` = Dice(pred vs clicked subset) − Dice(pred vs ALL
lesions) on subset rows. Negative ⇒ prediction matches all lesions better than the clicked ones.

**C — FP is not the problem.** `val_fp` / `lesion_free_decoy/val_pred_fg` sit at 0.0017–0.0029 the
whole run. Inference is hard `argmax` at 0.5
(`nanounet/infer/roi_slices.py:38`, `patch_export.py:117`, `predict_case.py:194`).

**D — negative prompt channel is dead.** `pn = []` at `nanounet/data/sampling.py:43`, never
appended, returned empty at line 67. Decoys are appended to the **positive** list.

**E — `sampling.large_lesion` is parsed and never read.** All four configs + `config.py`.

**F — multi-lesion patches are common on the product cohort.** d013 median 2 in-patch lesions,
50.2% of its fg patches have ≥2 (d030 82.3%, d023 55.1%). The usual "cc_dc_ce is a no-op" argument
is false here; it is still rejected on cost.

**G — small-lesion Dice is partly a resampling ceiling.** Median z 2.5 mm; 45.5% of cases are
natively ≤1.5 mm and get resampled down. Do not chase this with an unablated loss.

**H — short mixed FT is worth running; long d013-only FT is not.** 80-ep d013-only instance FT:
`all_clicked` 0.76→0.74 (best 0.78 @ 23), small-tag 0.27→0.35, selectivity −0.08→+0.02, prompt_gap
0.07→0.26. 500-ep d013-only specialised. Hence: 80 ep, **no** `--only-prefix`, d013 upweighted
inside a full-pool mixture.

**I — prompt encoding is not the bottleneck.** `val_prompt_gap` reached 0.228 at ep 395 then decayed
to 0.130 — the encoding is learnable; the 70:15 row mixture unlearns it. Geometry: the r=2
voxel-isotropic ball is a rod in mm (±5.0 mm z vs ±1.5 mm in-plane). Registration error is
near-isotropic in *resampled voxel* space `[5.95, 6.39, 5.93]` and 3.1× anisotropic in mm, matching
the 3.3× spacing anisotropy. Kernels are voxel-isotropic, so the ball is the right shape. Radius 2
sits under median lesion radius (pool 4.6 vox, d013 3.8, LIDC 2.8); growing toward σ≈6 would make
the marker larger than the median d013/LIDC lesion and merge neighbours (42% closer than 30 vox).

---

## 2. Change 1 — remove the negative prompt channel

**Why:** Finding D. Sequencing: **before SSL** (input width 3→2). Invalidates Dataset999
checkpoints; we are not reusing them. MAE is unaffected: `nanounet/pretrain/module.py:58` builds
with `n_extra_in=0`, and `nanounet/model/mae_transfer.py` zero-pads the 1-ch stem to the target
width.

**Do not populate the channel with dropped-lesion clicks.** No negative clicks at deployment ⇒
the net would learn "no negative click ⇒ segment it". That is July-audit root cause #1.

### 2.1 Constant

In `nanounet/prompt/encoding.py`, next to the module docstring (rewrite the docstring to drop
"pos/neg pair for two channels"):

```python
N_PROMPT_CHANNELS = 1  # positive heatmap only; the unused negative channel was removed
```

Import this everywhere a `+ 2` currently means "two prompt channels". Do not leave a stray `+ 2`.

### 2.2 Exact edits

1. `nanounet/prompt/encoding.py` — delete `encode_points_to_heatmap_pair` entirely (lines 61–72).
   Callers use `encode_points_to_heatmap(...).unsqueeze(0)`.

2. `nanounet/model/network.py`
   - line 35: `n_extra_in: int = 2` → `n_extra_in: int = N_PROMPT_CHANNELS` (import the constant).
   - `build_net_longi` (not trained this run, still update): line 61–73 comment +
     `n_extra_in=2` → `n_extra_in=N_PROMPT_CHANNELS`, `n_stream = 1 + 2` → `1 + N_PROMPT_CHANNELS`.

3. `nanounet/train/patch_render.py`
   - import `encode_points_to_heatmap` not `_pair`.
   - `concat_variant_keypoints` line 27: drop `v["points_neg"]`. Concat `points_pos` then optional
     `bl_points_pos`.
   - `split_variant_keypoints` lines 37–40: drop `n_pn` / `pn`. Entry is
     `{"pp": pp, "n_fp": int(v.get("n_false_pos", 0))}` plus optional `bp`.
   - `render_variant` lines 80–89:
     ```python
     fu_hm = encode_points_to_heatmap(
         _point_list(entry["pp"]), shape, pr.point_radius_vox, pr.encoding, None,
         pr.prompt_intensity_scale,
     ).unsqueeze(0)
     ```
     Same for `bl_hm` from `entry["bp"]` (no empty-neg argument). `click_inside_flags` is unchanged
     (reads `pp` and `n_fp` only).

4. `nanounet/data/sampling.py`
   - `select_prompt_points` currently returns `(pp, pn, n_fp)` with `pn` always `[]`. Change the
     return type to `(pp, n_fp)`, delete `pn`, update the docstring (it still says
     "positive, negative").
   - `points_variant` (line 85–92): drop the `"points_neg"` key.

5. `nanounet/data/valset.py` lines 147–150 — drop `"points_neg"` from both variant dicts.

6. `nanounet/cli/build_valset.py` line 150 — `ci_entry` may drop `"pn"` (unused by
   `click_inside_flags`).

7. `nanounet/infer/longi_row.py`
   - `n_stream = n_img + 2` (line 33) → `n_img + N_PROMPT_CHANNELS`.
   - Replace both `encode_points_to_heatmap_pair(loc, [], ...)` with
     `encode_points_to_heatmap(loc, ...).unsqueeze(0)`.

8. `nanounet/infer/predict_case.py:71` and `nanounet/infer/predict_patch.py:37`
   - `n_stream = n_img + 2` → `n_img + N_PROMPT_CHANNELS`.
   **The previous draft of this plan missed both files. Skipping them silently breaks inference.**

9. `nanounet/train/lightning_module.py` lines 80–82. Current:
   ```python
   # supervised [CT, hm+, hm-], longi [FU_CT, FU_hm+, FU_hm-, BL_CT, BL_hm+, BL_hm-].
   self._prompt_ch = [1, 2, 4, 5] if longi else [1, 2]
   ```
   Replace with:
   ```python
   # supervised [CT, hm]; longi [FU_CT, FU_hm, BL_CT, BL_hm]. Used to zero prompts for val_prompt_gap.
   self._prompt_ch = [1, 3] if longi else [1]
   ```
   **Previous draft missed this. Ablation would zero a CT channel.** File is 196 LOC — this is a
   1-line change.

10. `nanounet/model/dwb.py` module docstring currently `x = [FU(3ch); BL(3ch)]`. After the
    channel drop each stream is 2ch (`1 CT + N_PROMPT_CHANNELS`). Update the docstring. No
    behaviour change (we do not train longi).

### 2.3 Dead `large_lesion` config (Finding E) — same commit

- `nanounet/config.py`: delete `LargeLesionConfig`, `_load_large`, the `large_lesion` field on
  `SamplingConfig`, and `ll = _require(d, "large_lesion")` plus `_load_large(ll)` in
  `_load_sampling`. After this, a leftover JSON key is ignored (unknown keys are not read).
- Delete the `"large_lesion"` block from `configs/default.json`, `configs/instance_conditional.json`,
  `configs/finetune_d013.json`, `configs/longrun.json`.

### 2.4 Verify

```bash
python3 -c "import nanounet.train.patch_render, nanounet.infer.longi_row, nanounet.infer.predict_case, nanounet.infer.predict_patch, nanounet.config, nanounet.train.lightning_module"
grep -rn "points_neg\|heatmap_pair\|large_lesion\|n_img + 2\|n_extra_in=2\|n_stream = 1 + 2\|FU(3ch)\|_prompt_ch = \[1, 2" nanounet/ configs/ --include='*.py' --include='*.json'
# must be empty (allow N_PROMPT_CHANNELS itself)
```

Then a 1-step supervised smoke (§7) confirming `net` first conv `in_channels == 2`.

---

## 3. Change 2 — conditional click-drop (never drop the only lesion)

**Why:** Finding A. Only training-behaviour change in this plan. Selectivity-teaching rows
**15.3% → 29.1%**; empty-row rate 14.5% → ~14.2%. Keep-all exposure 70.2% → 56.7% (may cost a
little `val/all_clicked/val_dice` — currently the strongest and least product-relevant metric).

Replace `draw_kept` in `nanounet/data/instance_target.py` (currently lines 127–132) with:

```python
def draw_kept(in_patch: list[int], keep_prob: float, rng: np.random.Generator) -> list[int]:
    """Which in-patch lesions keep their click. Drawn once per patch, on the UNDISPLACED lesion
    set, so the choice cannot depend on the random displacement (which differs per variant).

    On a patch holding 2+ lesions the draw is forced to a PROPER subset (at least one kept, at
    least one dropped). An independent per-lesion coin left only 15.3% of foreground patches
    teaching "segment this one, not that one" against 70.2% teaching "segment everything", and
    the model duly learned to ignore the click (selectivity -0.23 after 1200 epochs). Forcing the
    split on multi-lesion patches doubles that signal to 29.1% and costs nothing; single-lesion
    patches keep the plain coin so the "predict nothing" rate is unchanged.
    """
    if keep_prob >= 1.0:
        return list(in_patch)
    if len(in_patch) < 2:
        return [j for j in in_patch if rng.random() < keep_prob]
    order = rng.permutation(len(in_patch))
    n_keep = int(rng.integers(1, len(in_patch)))  # 1 .. n-1 inclusive; numpy high is exclusive
    return sorted(in_patch[i] for i in order[:n_keep])
```

- `keep_prob` is `cfg.sampling.click_modes.pos`; leave it **0.8** in configs.
- `instance_targets: true` must be set in the run config (§5).
- Return sorted so `kept_set` / `fallback` behave as today.

**Verify** with a scratch script (delete after): draw 20k fg patches through the real
`resolve_instance_target` / `draw_kept` on Dataset900 and confirm row-type shares land near
56.7 / 29.1 / 14.2, and that no n≥2 patch returns 0 or n kept. Point
`NANOUNET_PREPROCESSED` at the NFS tree; do not copy 332 GB for this.

---

## 4. Change 3 — save a selectivity-aware checkpoint alongside pooled best

**Why:** Finding B. Do **not** replace `--monitor val_dice`. Tested on the real run:

| monitor | picks | val_dice | all_clicked | selectivity |
|---|---|---|---|---|
| `val_dice` (keep) | ep1005 | 0.712 | 0.788 | **−0.257** |
| `all_clicked + sel` | ep339 | **0.595** | 0.685 | +0.013 |
| **`all_clicked + 0.5*sel`** | **ep377** | 0.667 | 0.768 | **−0.153** |

Weight 1.0 picks an undertrained net. Weight 0.5 is the measured compromise.

### 4.1 Log the composite — `nanounet/train/val_metrics.py`

The previous draft said `dice_all_clicked + 0.5*(dice_clicked-dice_all)` and then admitted
`dice_all_clicked` might not be in scope. It is not. At lines 122–126:

- `dice_clicked` / `dice_all` are subset-row Dice vs clicked-subset / vs all lesions.
- `val/all_clicked/val_dice` is `dice_s` inside the scenario loop (lines 99–112) when
  `s == "all_clicked"`.

Capture it in that loop, then log after line 126:

```python
    dice_all_clicked = float("nan")
    for si, s in enumerate(SCENARIOS):
        sel = scenario == si
        # ... existing logs ...
        if s in ("all_clicked", "subset_clicked"):
            dice_s = _dice_sel(tp, fp, fn, sel)
            if s == "all_clicked":
                dice_all_clicked = dice_s
            lm.log(f"val/{s}/val_dice", dice_s, **d)
            # ... rest of existing branch unchanged ...
    # ... existing agreement logs ...
    # after val_selectivity_margin:
    lm.log("val_prompt_score", dice_all_clicked + 0.5 * (dice_clicked - dice_all), **d)
```

`SCENARIOS` in `valset.py:37` is
`("all_clicked", "subset_clicked", "none_clicked", "lesion_free_decoy")` — `all_clicked` is first.
File stays well under 200 LOC.

### 4.2 Third ModelCheckpoint — `nanounet/train/fit.py` ~line 190

Keep the existing `best-{epoch}-{monitor}` (`save_top_k=2`) and `save_last`. Insert **before**
`EMACallback`:

```python
        ModelCheckpoint(
            dirpath=join(out, ckpt_dir),
            filename="bestsel-{epoch}-{val_prompt_score:.4f}",
            monitor="val_prompt_score",
            mode="max",
            save_top_k=1,
        ),
```

`--monitor val_dice` stays. This cannot affect training.

### 4.3 FT starting checkpoint

Default: `bestsel-*.ckpt`. Fall back to `best-*.ckpt` only if bestsel is missing or its `val_dice`
(from `metrics.csv` at that epoch, or the `best-*` filename is not the source — read
`metrics.csv`) is **< 0.60** (the undertrained regime of weight 1.0).

Do not write a new eval CLI. Optional extra (not blocking): `Trainer.validate` on both ckpts
against `valset_2000.json` (~10 min) and record both rows in the run notes.

### 4.4 Resume-safe `ckpt_dir` — `nanounet/cli/train.py` ~line 54

Today `ckpt_dir = "finetune" if args.init_weights else "checkpoints"`. FT resume cannot pass
`--init-weights` (`validate_train_args` forbids it), so a resumed FT would start writing into
`checkpoints/` and mix with supervised weights if `--out` were shared. Even with a separate FT
`--out`, resume would split `finetune/` vs `checkpoints/`. Fix:

```python
    ckpt_dir = "finetune" if args.init_weights else "checkpoints"
    if args.resume:
        ckpt_dir = os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
        if ckpt_dir not in ("checkpoints", "finetune"):
            raise ValueError(
                f"--resume must sit in a checkpoints/ or finetune/ directory, got {args.resume}"
            )
```

---

## 5. Configs and validation manifest

### 5a. `configs/longrun900.json` — write this exact file

No `cohorts` block: `CohortSampler` then loads
`Dataset900_Merged/cohorts.json` (already correct, d013=0.0909). Site-balanced vs the old forced
25% d013 was near-neutral for selectivity signal (24.2% vs 28.0%).

No `require_weights`. Default is `false`. 999 had weights for 426 d013 cases only; a global
`require_weights: true` would crash on the other 20 prefixes. Generate d013 sidecars (§6); other
cases stay uniform per-centroid.

No `large_lesion`. Do not change the prompt block.

```json
{
  "prompt": {
    "point_radius_vox": 2,
    "encoding": "edt",
    "validation_use_prompt": true,
    "prompt_intensity_scale": 0.5
  },
  "sampling": {
    "fg_patch_prob": 0.67,
    "instance_targets": true,
    "click_modes": { "pos": 0.8, "drop": 0.2 },
    "false_pos_probability": 0.05,
    "propagated": {
      "mode": "empirical",
      "error_table": "/nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json",
      "backends": ["original", "unigradicon"],
      "sigma_per_axis": [5.95, 6.39, 5.93],
      "max_vox": 34.0
    }
  },
  "inference": { "tile_step_size": 0.75, "disable_tta_default": false }
}
```

`load_config` this file after §2.3 or it will still `_require("large_lesion")`.

### 5b. `configs/finetune900_d013.json` — write this exact file

Same as 5a plus a `cohorts` override that **names all 21 Dataset900 prefixes** (d011–d031) and
sets d013=0.38, remainder rescaled from site-balanced. `fg_patch_prob` stays 0.67 (the old
finetune config dropped it to 0.55 and grew FPs). `click_modes` stay 0.8/0.2.
`instance_targets: true`.

```json
{
  "prompt": {
    "point_radius_vox": 2,
    "encoding": "edt",
    "validation_use_prompt": true,
    "prompt_intensity_scale": 0.5
  },
  "sampling": {
    "fg_patch_prob": 0.67,
    "instance_targets": true,
    "click_modes": { "pos": 0.8, "drop": 0.2 },
    "false_pos_probability": 0.05,
    "cohorts": {
      "d011": 0.009557, "d012": 0.010844, "d013": 0.38,    "d014": 0.062,
      "d015": 0.003162, "d016": 0.01484,  "d017": 0.006306, "d018": 0.062,
      "d019": 0.011255, "d020": 0.003735, "d021": 0.062,    "d022": 0.062,
      "d023": 0.006113, "d024": 0.045032, "d025": 0.062,    "d026": 0.000687,
      "d027": 0.025033, "d028": 0.046474, "d029": 0.002962, "d030": 0.062,
      "d031": 0.062
    },
    "propagated": {
      "mode": "empirical",
      "error_table": "/nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json",
      "backends": ["original", "unigradicon"],
      "sigma_per_axis": [5.95, 6.39, 5.93],
      "max_vox": 34.0
    }
  },
  "inference": { "tile_step_size": 0.75, "disable_tta_default": false },
  "validation": { "no_lesion_frac": 0.3 }
}
```

Sum of `cohorts` = 1.0. Do **not** pass `--only-prefix`.

### 5c. Validation manifest

`--mix` order is `MIX_ORDER` in `nanounet/cli/build_valset.py:36`:
`(all_clicked, lesion_free_decoy, subset_clicked, none_clicked)` — **not** `SCENARIOS` order.

```bash
export NANOUNET_RAW=/nnunet_data/NanoUNet_raw
export NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results

nanounet_build_valset -d 900 --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --config configs/longrun900.json \
  --out /nnunet_data/NanoUNet_preprocessed/Dataset900_Merged/valset_2000.json \
  --n-patches 2000 --mix 0.35,0.15,0.30,0.20 --seed 1234 --max-tries 120
```

Raises `subset_clicked` 0.20→0.30 (the selectivity scenario) and cuts `lesion_free_decoy` 0.25→0.15
(FP already 0.0017). `SMALL_LESION_MAX_VOX = 500` (`valset.py:39`). After build, **require**
`val/tag/small/n ≥ 350` from the composition table (previous run’s n=197 swung ±0.15). If
`subset_clicked` cannot fill, raise `--max-tries` first, do not silently lower its share.

`config_stamp` only hashes `sampling.propagated`, so building after deleting `large_lesion` is fine.
Both `valset_2000.json` and `valset_2000.targets.npz` must exist. Reuse this manifest for FT.

---

## 6. Pre-flight artifacts (NFS, before submit)

1. **Lesion-type weights** (d013 only):

```bash
nanounet_lesion_weights -d 900 --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --meta-dir /nnunet_data/Longitudinal-CT/meta
```

Default `--only-prefix d013_Longitudinal_CT_`. Report coverage (999 was 426/537 = 79%). All **240**
unique patient hashes in the 900 d013 case ids already have a CSV under `meta/` (300 CSVs total;
the extra 60 match `test_patients.csv`). Missing CSV for a train/val case must still crash (R12)
— do not wrap in try/except. Registration error table already exists at
`/nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json`.

2. **Valset** — §5c.

3. **Centroid keys** — already present (spot-checked). If a later case fails:
   `nanounet_preprocess -d 900 --sidecars-only`.

4. **Do not copy Dataset999 MAE into this run.**

---

## 7. Battle-test protocol (this node, no 332 GB copy)

Point env at NFS. Skip `rclone`. After §2–§5:

| # | Command | Passes when |
|---|---|---|
| 1 | grep in §2.4 | empty |
| 2 | `python3 -c "from nanounet.config import load_config; load_config('configs/longrun900.json'); load_config('configs/finetune900_d013.json')"` | no raise |
| 3 | `python3 -c` build_net on Dataset900 plans, print `next(net.parameters()).shape` wait — first conv: `net.encoder.stem.convs[0].conv.weight.shape[1] == 2` | 2 |
| 4 | `draw_kept` 20k-patch sim (§3) | ~56.7/29.1/14.2, never 0 or n on n≥2 |
| 5 | Parse the exact stage-1/2 argv (no fit): `from nanounet.cli.train_parser import build_train_parser, validate_train_args` + `parse_args([...])` | no raise; `mae_pretrain True`; `loss=='dc_ce'`; `mae_epochs==250`; `epochs==1200`; `iters_per_epoch==1000` |
| 6 | Same for FT argv: `init_weights` set, `optimizer=='adamw'`, `epochs==80`, `only_prefix is None`, **no** `--mae-pretrain`, **no** `--longi` | no raise |
| 7 | 1-epoch / 2-iter smoke (NFS IO is slow; that is fine): `--mae-pretrain --mae-epochs 1 --epochs 0` is illegal because supervised still runs. Instead run **two** smokes: (a) `nanounet_pretrain -d 900 --plans nnUNetResEncUNetLPlans_h200_smallpv --epochs 1 --iters-per-epoch 2 --val-iters 1 --batch-size 2 --dl-bucket s --no-wandb --out /tmp/mae_smoke` (b) `nanounet_train -d 900 --plans ... --config configs/longrun900.json --mae-ckpt <smoke last.ckpt> --epochs 1 --iters-per-epoch 2 --val-iters 1 --batch-size 2 --prompts-per-patch 2 --consistency-weight 0.02 --dl-bucket s --no-wandb --out /tmp/sup_smoke` — **without** `--val-manifest` so you do not need the 2000-patch set yet. Delete `/tmp/*_smoke` after. | both exit 0 |
| 8 | `nanounet_lesion_weights` (§6.1) on NFS | coverage printed |
| 9 | `nanounet_build_valset` (§5c) on NFS | 2000 patches, small n≥350, `.targets.npz` exists |

`--mae-iters-per-epoch` default is `None`, and `fit.py:43` then uses `--iters-per-epoch`. The
Dataset999 MAE was `epoch=249-step=250000.ckpt` = **250 ep × 1000 iters**. Therefore the real job
must pass `--iters-per-epoch 1000` (supervised) and **must not** pass `--mae-iters-per-epoch 250`
or MAE would be 4× shorter than the proven budget. Integrated `--mae-pretrain` with
`--mae-epochs 250 --iters-per-epoch 1000` matches 250k MAE steps.

Smoke 7a uses standalone `nanounet_pretrain` only to avoid a 1200-epoch parser default actually
trying to train; the **submitted** job uses integrated `nanounet_train --mae-pretrain` so MAE and
supervised share one process and one staged copy.

---

## 8. SLURM script — write this file

Path:
`/home/nielsrocholl/SLURM/jobs/nanoUNet/prompt-robustness/slurm_final_900_h200.sh`

Base guards, staging, and thread exports on
`/home/nielsrocholl/SLURM/jobs/nanoUNet/prompt-robustness/slurm_supervised_999_h200.sh`.
Integrated `--mae-pretrain` argv shape is already proven in
`/home/nielsrocholl/SLURM/jobs/nanoUNet/nanoUNet-pretrain-train.sh` (`--mae-pretrain --mae-epochs
250 --iters-per-epoch 1000` then supervised in one process). Copy that flag cluster, **not** that
file’s node (`dlc-arceus`), 600-ep horizon, stretched 188/250, `--dl-bucket l`, `NANOUNET_RAW=.../nnUNet_raw`,
or `nnUNet_*` exports.

Differences that matter:

- Dataset **900**, config `configs/longrun900.json`.
- **`--include "cohorts.json"`** in rclone. 999 overrode cohorts in JSON so it did not stage this
  file. 900 has no override — `CohortSampler` reads `cohorts.json` from the preprocessed root.
  Also `--include "valset_2000*"` (json + npz).
- Integrated `--mae-pretrain --mae-epochs 250` then 1200-ep supervised in **one** `nanounet_train`.
- Then a second `nanounet_train` in the same job for 80-ep mixed FT (`--init-weights` from
  `bestsel`/`best`, `--config configs/finetune900_d013.json`, separate `--out` so FT cannot clobber
  supervised checkpoints).
- No `--longi`. No Dataset999 MAE path. No `--only-prefix`.
- `--time=7-00:00:00` (qos=vram as used on slowpoke). **Do not write 4.04 days.** Arithmetic:
  `578 s/epoch × 1200 = 8.03 days`; wandb `ekkxcgi6` runtime **8.0 days**; last epoch wall 619.6 s
  would be 8.6 days if constant. The 999 header’s “7.4 days” was the right order of magnitude;
  “4.04 days” in the previous draft was half of 8.03 (a division error). Budget:
  copy ~1 d + MAE 250k/1200k ≈ 0.21× step count (~1 d if similar, less if cheaper) +
  supervised ~8 d + FT 80×1000 ≈ 0.07× (~0.5 d) ≈ **10–11 d**. Expect **at least one resume**.
  `--mem=200G` is this user’s reservation ceiling (`slurm_preprocess_merge_900.sh` header) — do
  not raise it.
- Request **64 CPU, 200G, 1 GPU, qos=vram, nodelist=dlc-slowpoke**, same container/mounts as the
  999 H200 script. Do not copy `WANDB_RUN_ID=ekkxcgi6`.
- **Do not copy 999’s `RESUME="${RESUME-$OUT/checkpoints/last.ckpt}"`.** That default made a bare
  `sbatch` resume an *existing* 999 run. This is a new 900 run. Default = §8.1 state machine
  (inspect whether `$OUT` checkpoints exist on NFS). Optional `FRESH=1` wipes `$OUT` and starts
  over. Missing `$SUP_LAST` on a claimed resume must `exit 1`, not fall through into a second
  MAE.

### 8.1 Resume state machine (implement exactly)

```
SUP_LAST=$OUT/checkpoints/last.ckpt
MAE_LAST=$OUT/mae_pretrain/checkpoints/last.ckpt
FT_LAST=$OUT_FT/finetune/last.ckpt

if [[ -f $FT_LAST ]]; then
  # FT in progress / done — skip SSL+supervised, resume FT
  FT_ARGS=(--resume "$FT_LAST")
  SKIP_MAIN=1
elif [[ -f $SUP_LAST ]]; then
  # supervised in progress — skip MAE (train.py already drops mae_ckpt when --resume is set)
  MAIN_ARGS=(--resume "$SUP_LAST")   # no --mae-pretrain
  SKIP_MAIN=0
elif [[ -f $MAE_LAST ]]; then
  MAIN_ARGS=(--mae-pretrain --mae-resume "$MAE_LAST")
  SKIP_MAIN=0
else
  MAIN_ARGS=(--mae-pretrain)
  SKIP_MAIN=0
fi
```

`--mae-resume` requires `--mae-pretrain` (`validate_train_args`). `--resume` + `--mae-pretrain` is
allowed; `train.py:82` passes `mae_ckpt_arg=None` when `sup_resume` is set, so a supervised
resume must **not** pass `--mae-pretrain` (otherwise it would re-run MAE if last.ckpt looks
unfinished — avoid the branch entirely).

Do not `rm -rf $OUT` unless `FRESH=1`. Never delete `$OUT_FT` if it exists — refuse overwrite
(copy that guard from `slurm_finetune_d013_instance_h200.sh`). Checkpoints live on NFS
(`$NANOUNET_RESULTS`) and survive the job; only `/root/NanoUNet_preprocessed` dies, so every
resubmit re-runs rclone then `--resume`.

### 8.2 Exact train commands

**Stage 1+2** (`nanounet_train`):

```
nanounet_train \
  -d 900 -f 0 \
  --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --config configs/longrun900.json \
  --val-manifest "$VAL_MANIFEST" \
  --val-every-n-epochs 2 \
  --out "$OUT" \
  --mae-pretrain \
  --mae-epochs 250 \
  --mae-lr 1e-2 \
  --mae-lr-schedule cosine_warm_restarts \
  --mae-cosine-t0 250 \
  --mae-mask-ratio 0.75 \
  --batch-size 12 \
  --epochs 1200 \
  --iters-per-epoch 1000 \
  --lr 0.01 \
  --warmup-epochs 10 \
  --ema-decay 0.999 \
  --monitor val_dice \
  --lr-schedule stretched_tail_poly \
  --stretched-k 376 \
  --stretched-ref 500 \
  --loss dc_ce \
  --prompts-per-patch 2 \
  --consistency-weight 0.02 \
  --dl-bucket xl \
  --dl-persistent-workers \
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset900_f0_ssl_sup_instance_1200ep"
```

On supervised resume replace `--mae-pretrain ...` with `--resume $SUP_LAST`.
On MAE resume keep `--mae-pretrain` and add `--mae-resume $MAE_LAST`.

`--batch-size 12` is rows; with `--prompts-per-patch 2` that is 6 distinct patches. Must stay
divisible. `--consistency-weight 0.02` requires `prompts_per_patch>=2`.

**Stage 3** — only if stage 1+2 reached 1200 ep (`pl_ckpt_stage_done` / `last.ckpt` epoch). Pick
init ckpt per §4.3. Separate out dir:

```
nanounet_train \
  -d 900 -f 0 \
  --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --config configs/finetune900_d013.json \
  --val-manifest "$VAL_MANIFEST" \
  --val-every-n-epochs 2 \
  --init-weights "$INIT_CKPT" \
  --out "$OUT_FT" \
  --batch-size 12 \
  --epochs 80 \
  --iters-per-epoch 1000 \
  --optimizer adamw --lr 1e-5 --wd 3e-5 --grad-clip 1.0 \
  --warmup-epochs 2 \
  --lr-schedule poly \
  --loss dc_ce \
  --prompts-per-patch 2 \
  --consistency-weight 0.02 \
  --consistency-warmup-epochs 0 \
  --ema-decay 0.999 \
  --monitor val_dice \
  --dl-bucket xl \
  --dl-persistent-workers \
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset900_f0_mixed_d013_ft_80ep"
```

FT resume: `--resume $OUT_FT/finetune/last.ckpt` instead of `--init-weights`, same `--out`.

**SBATCH header** (copy mounts/image from the 999 script):

```
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-900-final
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_900_final.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_900_final.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"
```

**Staging includes** (fail before rclone if remote valset/weights/splits/plans/cohorts missing):

```
--include "${PLANS_NAME}.json"
--include "splits_final.json"
--include "cohorts.json"
--include "valset_2000*"
--include "${DATA_ID}/**"
```

Fail if `splits_final.json` does not have length 1. Fail if 20 centroid sidecars lack
`volume_vox`/`bboxes_zyx`. Fail if no `d013_*_weights.json` under `$DATA_ID`. Fail if
`nanounet_train --help` is broken. Export `NANOUNET_PREPROCESSED=$LOCAL_PREP` after copy.
`NANOUNET_RAW` stays on the mount (`$STORAGE/NanoUNet_raw`).

`VAL_MANIFEST=${LOCAL_PREP}/Dataset900_Merged/valset_2000.json`.

`OUT=${STORAGE}/NanoUNet_results/nanounet/Dataset900_Merged_${PLANS_NAME}_f0_h200_final`
`OUT_FT=${OUT}_ft`

Write the full bash file; do not leave "…" in the submitted script. `set -euo pipefail`. After
success, `rm -rf` only the local stage dir, never `$OUT`.

### 8.3 What to watch (priority)

`val/tag/small/val_dice`, `val_dice_macro`, `val/tag/click_outside/val_dice`,
`val/subset_clicked/val_selectivity_margin`, `val/lesion_free_decoy/val_pred_fg`,
`val_prompt_score`. **Do not judge d013 on pooled `val/cohort/d013/val_dice`** — previous run’s
0.35 was n=41 noise; same ckpt scored 0.76 on 400 d013 patches.

Early `val_prompt_gap ~ 0` for ~50 ep from MAE is expected. Do not kill the run for it.

---

## 9. Post-run (not this session, not the SLURM script)

Sweep the fg probability threshold on `valset_2000.json`. `val_fp=0.0017` buys recall.
Thread the chosen threshold through the three `argmax` sites as an explicit config field.
Fully reversible; do not bake a new threshold into this training job.

---

## 10. Explicitly rejected — do not re-litigate

| Idea | Why not |
|---|---|
| `cc_dc_ce` | ~4× step time (`_cc_voronoi` = CPU EDT per component per DS scale). Unablated. d013 *would* benefit (50.2% multi-lesion) — rejected on cost/risk, not irrelevance. |
| Fill negative channel with dropped clicks | No neg clicks at deploy → "no neg ⇒ segment it". Selectivity worse. |
| Larger / physically-isotropic marker | Encoding is learnable (prompt_gap 0.228). Geometry in §1.I. |
| Radius scaled by baseline volume | Biases FU extent toward BL, damps the RECIST signal. |
| Tversky / focal | Unablated. §9 is the reversible operating-point shift. |
| Finer z-spacing | True small-lesion ceiling; means re-preprocess 332 GB. Violates step-time. |
| W4 target rule | Fallback click already mirrors expand-tile inference (`longi_row.py:37-39`). |
| Force d013 to 25% in supervised | Near-neutral vs site-balanced. FT is where d013 is upweighted. |
| `--only-prefix` / 500-ep d013 FT | Specialised. |
| `--longi` / DWT this job | User: not this run. |
| Reuse Dataset999 MAE | Split leak. |
| `require_weights: true` | Only d013 has a meta CSV. Would crash the other 20 prefixes. |
| `configs/longrun.json` / `finetune_d013.json` | Wrong cohort keys / `pos=1.0`. |
| Separate `nanounet_pretrain` job | Would copy 332 GB twice. Use integrated `--mae-pretrain`. |
| `--mae-iters-per-epoch 250` | Would cut MAE to 62.5k steps; proven run was 250k. |
| Assume 1200 ep fits in 4 days | 578×1200 = **8.03 days**; wandb runtime 8.0 d. qos 7 d ⇒ resume. |

---

## 11. Verification checklist

Before `sbatch`:

- [ ] grep in §2.4 is empty
- [ ] non-longi first conv in_channels == 2; smoke §7.7 exits 0
- [ ] `draw_kept` sim ~56.7 / 29.1 / 14.2
- [ ] `val_prompt_score` appears after a 1-ep smoke with `--val-manifest` (run this after the
      valset exists; can be `--epochs 1 --iters-per-epoch 2` on NFS)
- [ ] `bestsel-*.ckpt` would be configured (inspect `fit.py` callbacks if the 1-ep val did not
      fire the monitor yet)
- [ ] `configs/longrun900.json` and `finetune900_d013.json` load; CohortSampler against Dataset900
      keys (the FT file names all 21)
- [ ] `valset_2000.json` + `.targets.npz`; small n ≥ 350
- [ ] `d013_*_weights.json` present; coverage reported
- [ ] `cohorts.json` listed in rclone `--include`
- [ ] argv parse of both train commands (§7.5–7.6)
- [ ] no `--longi`, no `--only-prefix`, no `--loss cc_dc_ce`, no Dataset999 MAE path
- [ ] SLURM script exists at the path in §8; `bash -n` it
- [ ] smokes exported `NANOUNET_*=/nnunet_data/...` (not `/nanounet_data`)
- [ ] user has been notified; job is **not** submitted

Touched files <200 LOC. Throwaway scripts deleted. Ask before commit.

---

## 12. Segtrack scorer caveat

Audit 2026-08-29/31, `/nnunet_data/audits/segtrack_audit/`: the follow-up scorer deletes **376 of
687** predicted CCs before scoring; `--ema` defaulted off at inference; GT `cog_fu` leaked into
the matcher. Gates 1–2 were never implemented. **If this run is scored through that pipeline
unchanged, the number measures the pipeline, not the model.** Fix Gates 1–2
(`NEXT_AGENT_PROMPT.md` in that folder) before claiming the retrain helped.

---

## 13. Implementation order

1. §2 channel + dead config (code + four JSON configs).
2. §3 `draw_kept`.
3. §4 composite + ModelCheckpoint + `ckpt_dir` resume fix.
4. §5 write `longrun900.json` + `finetune900_d013.json`.
5. §7.1–7.7 smokes (no valset yet).
6. §6 weights, then §5c valset, then optional 1-ep `--val-manifest` smoke.
7. §8 write SLURM script, `bash -n`, argv parse.
8. Stop. Notify. Do not sbatch.
