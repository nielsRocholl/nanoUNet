# Longitudinal Difference-Weighting (DWB) for nanoUNet — design, flaws, and the main contender

**Status:** exploratory / under active discussion. A first two-stream finetune is training; this
document captures the reasoning, the flaws we found by inspecting real patches, and the design
we now believe is correct. Nothing here is final — it is the decision record.

---

## 1. Motivation

nanoUNet is a single-timepoint, promptable lesion segmenter. On the Longitudinal-CT (metastatic
melanoma) test set it already matches the published SOTA (LongiSeg / autoPET-IV winner, `arXiv-2605.23118v1`)
**without using any baseline image information** — the only longitudinal signal is a propagated
baseline centroid rendered as a prompt heatmap on the follow-up (FU) scan.

LongiSeg's thesis is that the baseline **appearance** (not just a point) resolves ambiguous
boundaries and improves detection, *if* you (a) feed the baseline through a **shared-weight
encoder** and (b) fuse at the skip connections with a **Difference Weighting Block (DWB)**. Their
ablation:

| setup | DSC |
|---|---|
| single timepoint | 55.8 |
| naive longitudinal concat, **from scratch** | 54.0 ← *worse than single-TP (cross-sectional collapse)* |
| naive concat + synthetic pretrain | 56.5 |
| **+ Difference Weighting** | **58.5** |

Decision (Niels): keep the existing 3-stage recipe (MAE SSL → supervised → finetune). Add the
two-stream + DWB **only at the finetune stage**. No synthetic pretraining.

---

## 2. The Difference Weighting Block

One encoder, two passes, one subtraction. With shared weights both timepoints land in the same
feature space, so subtracting them is meaningful.

```
s_BL = enc(baseline) ; s_FU = enc(followup)        # shared weights
at every skip level l:
    x' = x_FU + x_FU · InstNorm(x_FU − x_BL)        # implemented in nanounet/model/dwb.py
```

- `x_FU − x_BL` = features now minus features then. Stable anatomy cancels to ~0; change (grow/
  shrink/appear/disappear) is large. A change detector in **feature space**, not pixels.
- `InstNorm(·)` standardizes that difference per channel over the volume → a scale-free "how
  unusual is this change vs the rest" salience map.
- `x_FU · (…)` gates the follow-up features (amplify change, relatively damp stable regions);
  `+ x_FU` is a residual so nothing is destroyed.
- **Free fallback:** if `x_BL = x_FU` (no prior), the difference is 0, `InstNorm(0)=0`, gate `=0`,
  so `x' = x_FU` — exact single-timepoint behaviour. (InstanceNorm `affine=False` ⇒ DWB is
  parameter-free, so the stage-2 supervised checkpoint loads verbatim.)

Why DWB rather than naive concat: naive fusion lets the net ignore the baseline branch
("cross-sectional collapse"); DWB makes the baseline *structural* — you cannot ignore a term you
multiply by.

---

## 3. What is implemented now (the "anchor-coloc" baseline)

Files: `nanounet/model/dwb.py` (`LongiResEncUNet`), `nanounet/model/network.py` (`build_net_longi`),
`nanounet/data/sampling_longi.py` (`build_patch_longi`), `nanounet/plan/longi_pairs.py` +
`nanounet/cli/longi_pairs.py` (offline pairing), `nanounet/data/blosc2_dataset.py`
(`load_case_properties` merges `<id>_baseline.json`), `--longi` flag threaded through
`cli/train.py`, `train/lightning_module.py`, `train/data_module.py`.

**Patch construction (`build_patch_longi`):**
1. Sampler picks an **anchor** = one true FU centroid (stratified by lesion type via
   `*_weights.json`). Crop the FU patch around it.
2. FU prompt heatmap = **all** in-patch FU centroids, each jittered by the calibrated propagation
   offset (σ = (2.75, 5.19, 5.40) vox, cap τ = 34). Plus spurious clicks (unchanged machinery).
3. BL stream = the **co-located** baseline crop: shift the BL window so the anchor's paired
   baseline centroid `c_bl` lands at the same in-patch offset as the FU anchor. BL prompt = **one**
   click at the anchor, **no jitter** (the baseline location is known).
4. **Null baseline** (BL case, new lesion, background/spurious patch): duplicate the FU stream →
   identity DWB → cross-sectional.
5. Single 6-channel tensor `[FU_CT, FU_hm+, FU_hm−, BL_CT, BL_hm+, BL_hm−]`; the network splits it.
   This keeps the producer/queue/collate/augment pipeline untouched (GPU utilisation preserved).

**Correspondence (offline):** the meta-CSV (`<hash>.csv`, cols `cog_bl`, `cog_fu`, `lesion_type`,
native xyz) pairs baseline ↔ follow-up lesions by lesion id. `nanounet_longi_pairs` forward-maps
those into preprocessed voxel space (reusing `cog_to_preprocessed`) and writes
`<fu_id>_baseline.json` = `{baseline_case_id, pairs_zyx}` aligned to the FU centroid list. On the
Dataset013 run: 269 FU cases, 2029 centroids, 87.5 % paired, median match 0.99 vox.

---

## 4. Empirical signal so far

`val_dice_macro` for the two-stream finetune tracks the single-stream finetune at near **parity**
(~0.66), slightly under at matched epochs.

**This neither proves nor disproves the prior helps.** Two reasons it is uninformative:
- **Null-baseline dilution.** In d013 ~half the cases are baseline-timepoint (null baseline), plus
  every new lesion and background/spurious patch — all of which are *mathematically* single-stream.
  A large fraction of the val metric is therefore insensitive to DWB.
- **Collapse looks identical to parity.** LongiSeg's whole point is that an unprepared longitudinal
  net collapses to cross-sectional and *matches* the single-TP baseline. We skipped the
  collapse-preventer (synthetic pretrain) and warm-start from the single-stream net, which biases
  toward collapse. Parity is the predicted signature of "prior ignored," not "prior helping."

The aggregate val curve cannot distinguish "helping but diluted" from "ignored." The real arbiter
is a **centered, per-lesion test eval on the FU-with-baseline subset**, plus the ablation in §8.

---

## 5. Flaws found by inspecting real patches

We extracted real training-style patches (`/Users/.../Downloads/longi_viewer`, cases
`8c19f571e2_00`, `b34f33d0bc_00`) with three baseline variants — `samebbox`, `coloc`, `stamped` —
and read off the geometry. Findings:

### 5.1 The images are not registered — only the point is
DWB subtracts features voxel-wise, which is only meaningful where the same patch coordinate is the
same anatomy in both streams. We do **not** register the volumes; we only align a point. Baseline
and follow-up are separate acquisitions with **different coordinate frames** (e.g. b34f: FU volume
`(588,541,536)`, BL `(392,640,634)`; the *same* lesion 8 sits at FU `[281,220,209]` vs BL
`[280,258,263]`). Cropping BL at the FU window (`samebbox`) lands ~`[−1,+38,+54]` vox away — a
different part of the body. This is expected, and is *why* we need the pairing shift.

> DWB needs **local co-location at the lesion**, not global registration. This is exactly
> LongiSeg's setting — they also center VOIs on prompts and do not warp the images.

### 5.2 Anchor-only co-location: neighbours drift
The current `coloc` is one rigid shift that aligns the **anchor** exactly. A neighbour lesion N's
residual misalignment works out to:

```
(fu_N − fu_anchor) − (bl_N − bl_anchor)   =  the change in the anchor→N vector between scans
                                          =  the local relative deformation
```

The global offset cancels (good); only relative deformation remains. Measured (in-patch offset,
"BL lands at" vs "FU at"):

- **8c19** (gentle pose): anchor 0; neighbours drift **1–13 vox**. Anchor-only ≈ fine.
- **b34f** (large interscan shift): anchor 0; neighbours up to **id13 = 33**, **id29 = 37**,
  **id11 = 27**. Far lesions badly misaligned.

So Niels' intuition ("lesions don't move much, so a crop around the same lesion roughly aligns the
others") is **correct to first order** — exact for the anchor, good for nearby lesions, breaking
for distant/deformed ones.

### 5.3 Clustered inference breaks per-lesion co-location
The best-performing, most efficient inference mode is **clustered** (multiple prompts per patch,
one forward per cluster). But a single BL crop can only co-locate **one** lesion. For every other
prompt in the cluster, DWB subtracts the FU lesion against unrelated baseline anatomy → spurious
"change". Clustered and per-lesion baseline alignment are fundamentally in tension.

### 5.4 Stamping introduces out-of-distribution artifacts — rejected
To co-locate *all* lesions in one patch we considered **stamping**: paste each paired lesion's BL
neighbourhood at its FU position on a canvas. Inspecting it, the baseline becomes a collage of
boxes with seam discontinuities that never occur in a real CT. Worse: the encoder is **shared**
with the clean FU stream, so feeding it composites can pollute the FU encoding too. A CNN will key
on / be confused by these seams. **Stamping is rejected.** Co-location must come from a *real*
image, not a collage.

### 5.5 Train/inference mismatch on the anchor itself
The deepest flaw. At **inference** we do not know the true FU lesion centre; we center the patch on
the **propagated** point (BL centre pushed through registration into FU), which breathing/
deformation makes imperfect by an error **ε** (the same residual the jitter is calibrated from:
σ = (2.75, 5.19, 5.40) vox, cap 34). So at inference the BL lesion (known centre) and the FU lesion
sit **ε apart** even for the anchor.

But the current `build_patch_longi` co-locates the BL crop on the **true** FU centroid and jitters
only the *prompt heatmap*, not the crop alignment. So in **training** the anchor lesions are placed
**perfectly on top of each other** — DWB never practices the ε misalignment it will face at test.
Training is "too easy" on the image-alignment axis.

### 5.6 Alignment error budget (summary)
- **Anchor lesion:** misaligned by **ε** (propagation error). Small on average, occasionally large.
- **Neighbour lesion:** misaligned by **ε + relative deformation** (grows with distance).
- Nothing is perfectly aligned. The anchor is merely the *least* misaligned.

---

## 6. Options considered (ranked, artifact-aware)

| approach | baseline image | co-location | clustered? | artifacts | notes |
|---|---|---|---|---|---|
| **Warp BL→FU** with dense φ | real CT, resampled | all lesions + background | ✓ | none | most correct; needs dense deformation field in preprocessing; inherits registration error |
| **Centered + contiguous BL crop** (= LongiSeg) | real contiguous crop | per-lesion exact | ✗ (1 fwd/lesion) | none | clean; costs the clustered efficiency |
| **coloc + all prompts (+ ε crop jitter)** ← contender | real contiguous crop | anchor exact, neighbours approx | ✓ | none | see §7 |
| anchor-coloc, anchor prompt only (current) | real contiguous crop | anchor only | ✓ | none | only the anchor benefits; others ~cross-sectional |
| ~~stamping~~ | composite collage | all | ✓ | **yes** | rejected (§5.4) |

The only way to keep clustered **and** align everything **and** stay in-distribution is real
**registration (warp)** — a warped CT is still a real CT. Stamping was the hack to avoid needing φ;
its OOD cost is too high.

---

## 7. Main contender: `coloc` + all prompts + calibrated crop jitter

The pragmatic, artifact-free design, generalizing LongiSeg's single-lesion VOI to the multi-lesion
patch:

1. **Real contiguous `coloc` crop** (no stamping). Rigid-shift the BL window to align the anchor.
2. **Encode every lesion's prompt in both streams** (BL prompts at their true, known baseline
   centroids, no jitter; FU prompts jittered as now). This makes the BL encoder *aware of all*
   baseline lesions — symmetric with the FU stream — instead of just the anchor.
3. **Jitter the BL-vs-FU crop alignment by the calibrated propagation offset** (the σ we already
   have), so training reproduces the inference anchor misalignment ε (fixes §5.5). Today σ only
   jitters the prompt; it must *also* perturb the image co-location.

### The crucial correction (why this is honest, not magic)
Encoding all prompts does **not** register the neighbours. A prompt rides *on* its lesion: when the
BL window shifts to center the anchor, lesion N's **anatomy and its prompt move together** by the
same relative-deformation vector. Formally `BL_prompt_N − FU_prompt_N = (bl_N−bl_anchor) −
(fu_N−fu_anchor)` = the same drift as the image. So:

- **What it buys:** multi-lesion awareness on real CT, zero artifacts, and ε-robust training.
  Nearby/aligned lesions get the longitudinal benefit; the anchor is handled correctly under ε.
- **What it does *not* buy:** alignment for far/deformed lesions. Their prompt and image both drift
  together; DWB still subtracts misaligned features there (the residual absorbs it → ~cross-
  sectional for those lesions). Only **warp** (or one-lesion-per-patch **centered**) lifts that
  ceiling, and that ceiling is set by *not registering* — the prompt cannot fix it.

This is the same trade LongiSeg accepts (their VOIs hold one lesion, so there are no neighbours to
drift). We accept it across a multi-lesion patch.

---

## 8. How to actually find out if DWB helps

The aggregate val curve is insensitive (§4). Decisive measurements, in order:

1. **Null-baseline ablation control.** Same two-stream model, but force the BL stream to duplicate
   FU for *all* patches (no real prior). If real-baseline ≈ null-baseline → the architecture is
   ignoring the prior (collapse). The single cleanest experiment.
2. **Gate-magnitude probe.** Log `‖InstNorm(x_FU − x_BL)‖` on real-baseline vs null patches. Tiny
   or identical → ignored.
3. **Centered, per-lesion test eval on the FU-with-baseline subset, by lesion type.** The real
   arbiter. Compare detected_dsc / LDR against the single-stream finetune. Expect any lift in
   ambiguous boundaries and small-lesion detection (lymph node, soft tissue/skin, skeleton), not in
   aggregate `val_dice_macro`.

If collapse is confirmed, synth-free levers (increasing intervention): prompt-localized DWB
(compute the gate only near prompts), DWB on deep skips only, brief early-train asymmetry (low-LR
the FU-decoder path so the only way to cut loss is through the BL stream), or revisit the
(declined) synthetic pretraining that LongiSeg credits as the collapse-preventer.

---

## 9. Inference note (two-stream)

`predict_case.py` already has baseline support scaffolding (`pad_bl`, baseline points). For the
two-stream model, **run `centered` mode** (one patch per prompt, BL partner at centre) so every
patch is a single co-located lesion pair — clustered breaks co-location (§5.3) for all but one
prompt. If/when warp is added, clustered becomes valid again because the whole patch is aligned.

---

## 10. Concrete next steps

- [ ] Change `build_patch_longi`: (a) encode **all** paired in-patch BL prompts (not just the
      anchor); (b) draw a propagation offset (existing σ/τ) and apply it to the **BL crop window**
      so the anchor is ε-misaligned in training (fix §5.5). Keep null-baseline = duplicate FU.
- [ ] Add the **null-baseline ablation** switch (force duplicate-FU everywhere) for §8.1.
- [ ] Log the **DWB gate magnitude** during validation for §8.2.
- [ ] Force `mode="centered"` in the longi predict path; run the per-lesion eval on the paired
      subset, by lesion type (§8.3).
- [ ] Only if drift is shown to matter: scope **dense φ warp** of the baseline into FU space
      (artifact-free path to clustered + full alignment).

---

## 11. One-paragraph summary

DWB looks at *now*, subtracts *then* in feature space, and multiplies its attention by what
changed; with no prior, the multiplier vanishes and it degrades to an ordinary segmenter. It needs
the lesion **locally co-located** between the two crops — which our pipeline gets from a *point*,
not image registration. The anchor co-locates up to the propagation error ε; neighbours co-locate
only up to the local deformation (small for nearby lesions, tens of voxels for distant ones in
deformed patients). Stamping all lesions into one crop fixes the geometry but creates OOD seams and
is rejected. The current code aligns the anchor *too perfectly* (it doesn't simulate ε), and prompts
only the anchor on the baseline. The **main contender** keeps a real contiguous co-located crop,
encodes every lesion's prompt in both streams, and jitters the crop alignment by the calibrated ε —
artifact-free, multi-lesion-aware, and honest about its one inherent limit (far-lesion drift, which
only registration can remove). Whether any of this beats the single-timepoint model is to be decided
by a null-baseline ablation and a centered per-lesion test eval, not the aggregate val curve.
