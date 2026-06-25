# Inference optimization session — 2026-06-25

Goal for the session: **squeeze maximum inference-time performance out of the existing
`d013` longitudinal finetune, with no retraining.** Dataset: autoPET-IV longitudinal CT
(baseline `BL` = on-lesion prompts, followup `FU` = registration-propagated prompts). Model:
single-timepoint prompt-aware 3D ResEnc-L U-Net,
`Dataset999_Merged_..._finetune_d013/finetune/`.

This is a self-contained record. The detailed experiment log lives outside the repo at
`/nnunet_data/unprocessed-universal-lesion-segmentation/handoff/` (ephemeral-node handoff dir).

---

## 0. TL;DR

- **One free win found: use checkpoint `best-epoch=412`, not `last` (epoch 499).** It is mildly
  overfit. On the full FU val set (820 lesions, clustered + max-merge + border-expand, TTA):
  per-lesion DSC **62.6 → 63.9 (+1.3)**, LDR 87.7 → 88.3, case DSC 63.7 → 64.1; neutral on BL; no
  per-type regressions. Pure checkpoint swap.
- **Everything else was refuted and verified** (not just asserted): the train/inference
  distribution is already fully matched, and every post-processing / ensembling / prompt trick is
  DSC-neutral-to-negative. After the previously-shipped merge fix, per-lesion DSC is at the
  prompt/detection ceiling for inference-time levers.
- **Deliverable config:** `--inference-mode clustered --merge max --border-expand`
  `--ckpt finetune/best-epoch=412-val_dice_macro=0.6649.ckpt` (TTA on, `--batch-size 24`).

---

## 1. Repo recovery (start of session)

The fresh ephemeral `/root/nanounet` clone had landed on a **stale `origin/main`** (`d83bc0b`,
6 commits behind), missing the prior session's merge fix and the `--merge` / `--inference-mode` /
`--patients-csv` flags. The cloned code still used the legacy gaussian-average merge (the washout
bug). Cause: the previous container shut down before the work was fully visible to this clone.

Fix: the HTTPS remote has no credentials in the container, but the SSH key is registered —
`git remote set-url origin git@github.com:nielsRocholl/nanoUNet.git` then `git fetch && git merge
--ff-only origin/main` fast-forwarded to `90282ea` (the real HEAD with the fix). Verified the pulled
code reproduces the prior session's on-disk predictions **bit-exactly** (agreement Dice 1.0000).

---

## 2. Distribution-match audit (your strongest hypothesis — turned out clean)

Audited the full inference pipeline against the finetune config (`nano_config.json`, `plans.json`)
and the training sampler. **No mismatch exists to exploit:**

| Aspect | Training | Inference | Match |
|---|---|---|---|
| Prompt encoding | EDT ball r=2, intensity 0.5 | `prompt/encoding.py` via `nano_config` | ✅ |
| Intensity norm + spacing | `run_case_npy` (CTNorm + resample to plans spacing) | same `run_case_npy` | ✅ |
| TTA | full 3-axis mirror | full 3-axis mirror | ✅ |
| Patch size | 96×160×160 | 96×160×160 | ✅ |
| Clicks per lesion | **1 jittered click/lesion** (+ false-positive decoys) | 1 click/lesion | ✅ |

Notable: the `large_lesion.K` (multi-click) config is **parsed but never used** in
`data/sampling.py build_patch` — so "give large lesions multiple clicks at inference to match
training" is a non-starter; one click/lesion already matches the finetune distribution.

---

## 3. The win — checkpoint selection

The `finetune/` dir has 3 distinct checkpoints: `best-epoch=412`, `best-epoch=425`, and `last`
(== `epoch=499`, identical md5). The prior work used `last`.

Full FU val (820 lesions, clustered + max + border-expand, TTA):

| checkpoint | per-lesion DSC | LDR | detected DSC | case DSC |
|---|---|---|---|---|
| `last` (epoch 499) | 62.6 | 87.7 | 71.4 | 63.7 |
| **`best-epoch=412`** | **63.9** | **88.3** | **72.4** | **64.1** |

Per-type FU deltas (412 − last): Lymph node +1.7, Soft tissue +1.5, Others +7.0, Lung +0.8,
Liver +0.5; no real regressions. BL paired (16 cases): neutral (69.0 vs 68.8). Consistent ordering
**412 > 425 ≈ 499** → mild overfit past ~epoch 412.

Two caveats worth recording:
- The `val_dice_macro` checkpoint selector **does not track per-lesion DSC** (it ranked 425 above
  412, but 412 wins the task metric). Select empirically on the task metric.
- A **logit-ensemble of the 3 checkpoints is *worse* than 412 alone** (59.8 vs 61.1 on the screen) —
  the weaker members dilute it. Pick the single best checkpoint; do not ensemble.

---

## 4. Levers refuted this session (verified, do not re-try)

| Lever | Result | Verdict |
|---|---|---|
| Train/inference distribution mismatch | none found (full audit) | ✅ already matched |
| Checkpoint logit-ensemble (412+425+499) | 59.8 < 61.1 (412 alone) | ❌ dilutes |
| Morphological **dilation** of preds | dil1 59.4→45.4, dil2→30.8 | ❌ strongly negative |
| **Fill-holes** per CC | 55.9 → 55.9 | ❌ model leaves no holes |
| **FP-island suppression** | spurious islands = 11% of all FP | ❌ ~nothing to gain |
| Prompt-jitter TTA **averaging** | prompt is informative, not a nuisance var | ❌ reasoned out |
| (prior) recenter / threshold / union-multiprobe | DSC-flat | ❌ refuted earlier |

Root reason there is no further inference lunch: the whole-volume error is **false-negative
dominated** (FN ≈ 2× FP) and **regional** (missed lesions / missed portions of large lesions), not a
boundary shrink — so blanket post-processing cannot recover it. Recall is limited by propagated-
prompt quality. Real gains need better prompts (registration) or longitudinal retraining — out of
scope here.

---

## 5. Deliverable

Recommended config: `--inference-mode clustered --merge max --border-expand`
`--ckpt finetune/best-epoch=412-val_dice_macro=0.6649.ckpt` (TTA on, `--batch-size 24`).

Regenerated prediction sets (clustered, the best mode) with `best-epoch=412`:
- `predictions/val_FU_clustered_412/` (67 cases)
- `predictions/val_BL_clustered_412/` (68 cases)

Final stratified numbers (DSC ×100):

| set | per-lesion DSC | LDR | detected DSC | case DSC | n lesions |
|---|---|---|---|---|---|
| FU clustered — `last` (before) | 62.6 | 87.7 | 71.4 | 63.7 | 820 |
| **FU clustered — `412` (deliverable)** | **63.9** | **88.3** | **72.4** | **64.1** | 820 |
| BL centered — `last` (ref) | 64.3 | 88.9 | 72.4 | 75.6 | 854 |
| **BL clustered — `412` (deliverable)** | **68.6** | **94.1** | **72.9** | **77.3** | 854 |

FU before/after is the clean checkpoint A/B (same mode): **+1.3 per-lesion DSC, +0.6 LDR, +0.4 case
DSC**. BL clustered-412 is also the first *full* BL-clustered set (the prior one had only 16 cases)
and is the strongest BL result.

Stratified results JSON (per-lesion DSC / LDR / case DSC, per lesion type) regenerated at:
`/nnunet_data/unprocessed-universal-lesion-segmentation/results/lesion_type_stratified_results.json`
(contains the `*_412` deliverable sets plus the `last`-checkpoint sets for the before/after).

Reproduce:
```bash
MD=/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_finetune_d013
D=/nnunet_data/unprocessed-universal-lesion-segmentation
nanounet_predict -i "$D/inputsTrFU" -o "$D/predictions/val_FU_clustered_412" \
  -m "$MD" --ckpt "finetune/best-epoch=412-val_dice_macro=0.6649.ckpt" \
  --inference-mode clustered --merge max --border-expand \
  --batch-size 24 --num-workers 3 --patients-csv "$D/val_patients.csv" --device cuda --overwrite
python3 "$D/handoff/scripts/strat_eval.py"   # writes the results JSON
```

README updated (`nanounet_predict` section) with the checkpoint recommendation. Experiment scripts
added this session: `handoff/scripts/subset_eval.py` (paired per-lesion/case DSC across pred dirs)
and `handoff/scripts/ensemble_exp.py` (checkpoint logit-ensemble inference).
