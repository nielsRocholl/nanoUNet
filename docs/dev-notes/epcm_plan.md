# EPCM: a self-grading head for nanoUNet

Date: 2026-09-23
Status: proposal — implementation scope drafted, evaluation sketched but not planned.

Previous version: [epcm_plan_v0.md](epcm_plan_v0.md). Scope: implementation. Evaluation is sketched, not planned.

## 1. The idea in plain words

- **Problem.** A segmentation model draws a line around a lesion and says nothing about how good that line is.
- **AlphaFold's trick.** Next to each prediction it outputs a grade: "I'm 90% sure about this part, 40% about that part".
  A small extra network learned to produce that grade by looking at what the main network "was thinking".
- **Our version.** A small add-on to nanoUNet. For every point on the drawn contour it says
  **how many mm the line is probably off, and in which direction** (too big or too small).
- **How it learns.** Show it lesions the segmenter never trained on, where the true border is known.
  At every contour point tell it "you were 3 mm too far out here". After training it guesses this without the answer.
- **What the clinician gets.**

| Output | Looks like | Means |
|---|---|---|
| Coloured contour | green / yellow / red, with in/out direction | "edit here, the border is further out" |
| Lesion score 0–100 | `82` | "82% of this contour is close to where an expert would draw it" |
| Volume range | `12.3 mL (10.9–13.1)` | "an expert's volume is likely in this range" |

Honest meaning: **expected disagreement with an expert annotator.** Single-reader data mixes model error and reader
noise; we never claim to separate them.

```
CT patch + click ──► nanoUNet (frozen) ──► mask
                          │ decoder features (what it "was thinking")
                          ▼
                     self-grading head ──► per contour point: 9 error bins (mm, signed)
                                       ──► per lesion: volume-error bins
                                       ──► readouts: coloured contour, score 0–100, volume range
```

## 2. Inputs

| Input | Shape | Source |
|---|---|---|
| Full-res decoder features `f0` | `B×C0×D×H×W` (C0=32 for ResEnc) | forward hook on `net.decoder.stages[-1]` |
| Coarse decoder features `f2` | `B×C2×D/4×H/4×W/4` | forward hook on `net.decoder.stages[-3]` |
| Bottleneck `fb` | `B×Cb×…` | forward hook on `net.encoder.stages[-1]` (lesion head only) |
| Foreground probability `p` | `B×1×D×H×W` | softmax of the segmenter's logits |

All inputs detached. No edit to `dynamic_network_architectures` or `model/network.py`: hooks only.
No prompt channel: the features already encode the click. Adding one invites a leakage argument.

## 3. Outputs

| Output | Where | Format |
|---|---|---|
| Local error bins | every voxel of the predicted-contour shell | 9 logits → probabilities |
| Lesion volume-error bins | one per clicked lesion | 8 logits over `log2(V_pred / V_true)` |

Local bins, signed mm (`+` = contour too far out = over-segmentation):

```
(-inf,-8] (-8,-4] (-4,-2] (-2,-r] (-r,+r) [r,2) [2,4) [4,8) [8,inf)      r = max(1.0, 1.5 * min(spacing))
```

`r` is read from `plans.json` at startup. A perfect contour's shell voxels sit about one voxel from the GT border by
construction, so the centre bin must be at least one voxel wide. Sanity test T1: feeding GT as the prediction must put
>95% of shell voxels in the centre bin.

Volume bins: edges `log2` ratio `[-1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1]` (×0.5 … ×2, tails open).
Separate head because contour errors move together: 10 points each 1 mm out in the same direction wreck the volume,
random jitter cancels. The per-point map cannot tell these apart.

Readouts (no parameters):

| Readout | Formula |
|---|---|
| Expected signed error `E[δ](v)` | `Σ_k p_k · centre_k` (tails: ±12 mm) |
| Lesion score (pLDDT analogue) | `100 · mean_shell ⅓ Σ_{t∈{1,2,4} mm} P(abs(δ) < t)`. Exact bin sums, since the edges sit at 1/2/4 |
| Volume range | `V_pred / 2^q` for the 10/50/90% quantiles `q` of the volume-bin distribution |

## 4. Targets (how "the answer" is computed)

| Step | Where | Detail |
|---|---|---|
| `sdt_gt` | dataloader worker, CPU | signed EDT of the clicked GT instance on the crop, `sampling=spacing`, `+` outside, clipped ±16 mm |
| `shell` | GPU | `maxpool(pred) − minpool(pred)`, 3×3×3 → the 2-voxel ring across the predicted border |
| `δ` | GPU | `sdt_gt[shell]`: a gather, no EDT on GPU |
| volume target | GPU | `log2((V_pred + ε) / V_true)`, `V` summed in the patch |

- **Edge cases:**
  - Missed lesion: empty shell, so no local loss; volume target falls in the lowest bin.
  - Spurious blob: `sdt_gt` is large and positive, so it lands in the top bin.
  - Lesion cut by the patch face: no volume loss; the flag `lesion_complete` gates it.
- **Known simplification:** `sdt_gt(v)` is the distance to the *nearest* GT border, not the distance along the contour normal.

## 5. Training

| Choice | Value | Why |
|---|---|---|
| Segmenter | existing fold-0 checkpoint, `eval()`, `no_grad` | AlphaFold PAE recipe: grade a finished model |
| Head data | fold-0 **val** cases only (15%, ≈750–1500 scans), split 70/10/20 → head-train/val/test, stratified by source | the segmenter memorised its train set: errors there ≈ 0, so the head would learn to be overconfident |
| Split file | `confid_split.json`, disjointness from segmenter train asserted at startup (R15) | leakage is silent otherwise |
| Patches | existing `patch_iterable` pipeline, **one click per patch**, `instance_targets=true`, same click jitter as segmenter training | one patch = one lesion, so the per-patch mean gives equal weight per lesion |
| Augmentation | on (same chain) | more error diversity from a small held-out set; ablate later |
| Loss | `CE_local` (mean over shell, per patch) `+ 0.5 · CE_volume` | AlphaFold: cross-entropy on bins of true error |
| Optimiser | AdamW 1e-3, cosine, 100 epochs × 250 iters, bf16 | head is ~100k params |
| Cost | backbone forward only + head | faster than segmenter training. G4: log samples/s before/after |

**Head architecture:**

```
local:  z = cat(f0, up(conv1x1(f2 → 16)), p)                      # 32+16+1 channels, full res
        z → Conv3d 3³ (49→32) → GroupNorm → GELU → Conv3d 3³ (32→32) → GELU → Conv3d 1³ (32→9)
lesion: h = cat(masked_mean(z_hidden, pred), masked_mean(z_hidden, shell), mean(fb), log V_pred)
        h → Linear(…→64) → GELU → Linear(64→8)
```

Masked means are sums ÷ counts, so they add across tiles: large lesions can be pooled over several inference tiles later.

## 6. Inference

`ConfNet(nn.Module)` wraps backbone + head. Its `forward` returns `cat(seg_logits, local_bin_logits)`, so the existing tile
loop and TTA (`infer/tta.py`) carry the extra channels. Signed distance is flip-invariant, so mirroring is safe.
**To check:** `predict_case` max-merges logits across tiles (`grow_canvas`). Bin logits need their own merge (centre-weighted
mean), not max.

| Output file | Content |
|---|---|
| `<case>.nii.gz` | mask (unchanged) |
| `<case>_err.nii.gz` | `E[δ]` in mm on the shell, 0 elsewhere (signed; colourmap in viewer) |
| `<case>_bins.npz` | shell coords + 9 probabilities (float16): full distribution, kept for later analysis |
| `<case>_lesions.csv` | `lesion_id, click_xyz, volume_ml, score, vol_p10, vol_p50, vol_p90, lesion_complete` |

## 7. Files

| Path | LOC | Contents |
|---|---|---|
| `nanounet/confid/target.py` | ~110 | bin edges from plans, `sdt_mm`, `shell`, `bin_delta`, `bin_volume` |
| `nanounet/confid/head.py` | ~140 | `tap_decoder(net)` hooks, `ConfHead` (local + lesion) |
| `nanounet/confid/module.py` | ~180 | `ConfidLM`: frozen backbone, losses, val logging (δ histograms per size bin) |
| `nanounet/confid/readout.py` | ~120 | bins → `E[δ]`, score, volume quantiles, lesion table |
| `nanounet/cli/confid_train.py` | ~150 | train; `--budget-only` = one frozen pass → error-budget table (G0), no training |
| `nanounet/cli/confid_predict.py` | ~170 | thin CLI over `infer/*` with `ConfNet` (`predict.py` is at 198 LOC, no room for a flag) |

Worker-side `sdt_gt` goes in `train/patch_render.py`, behind a flag. Throughput is measured first. If EDT starves the GPU,
precompute per case offline instead.

```bash
nanounet_confid_train -d 501 -f 0 --plans nnUNetResEncUNetLPlans --seg-run runs/<seg_run> --budget-only
nanounet_confid_train -d 501 -f 0 --plans nnUNetResEncUNetLPlans --seg-run runs/<seg_run> --epochs 100
nanounet_confid_predict -i /path/to/scans -o /path/to/out -m runs/<seg_run> --head runs/<confid_run>/last.ckpt
```

## 8. Build order

| Step | Deliverable | Done when |
|---|---|---|
| 1 | `target.py` + T1 sanity test | GT-as-pred puts >95% of shell in the centre bin |
| 2 | `--budget-only` | error-budget table per lesion size (<10 / 10–30 / >30 mm). **G0:** if almost everything is in the centre bin, there is nothing to learn, so redesign |
| 3 | `head.py` + `module.py` + train CLI | head-val CE falls below the "predict the marginal histogram" baseline |
| 4 | `confid_predict` + readouts | coloured contour opens in viewer; lesion CSV written |

## 9. Novelty (backed by the Edison Scientific literature map, Sep 2026)

**Claim:** a single-pass, feature-conditioned head that predicts a **binned distribution over signed boundary error in mm**
for prompted 3D lesion segmentation, trained on held-out realised errors. From it come per-point edit guidance, a lesion
score and a volume range. No verified prior work combines these.

| Closest work | What it does | Difference |
|---|---|---|
| AlphaFold2 pLDDT / PAE (Nature 2021) | binned self-error from internal features | the template; proteins, not images |
| Zhang & Chung 2019 ([1907.12244](https://arxiv.org/abs/1907.12244)) | binary voxel error map, cardiac | separate image+mask QC network; unsigned, no mm |
| Geometric transformer error estimation 2023 ([2308.05068](https://arxiv.org/abs/2308.05068)) | node-wise mesh error | simulated errors, not the segmenter's own features |
| Mask Scoring R-CNN 2019; SAM IoU head 2023 | scalar mask quality from internal features | one global number, no location or direction |
| COMPASS 2025 ([2509.22240](https://arxiv.org/abs/2509.22240)); conformal volume, MICCAI 2024 | calibrated metric intervals | post-hoc intervals, no learned local error |
| Uncertainty-guided dual-views 2023 (Nat. Mach. Intell.) | predicts an object SDF | SDF of the object ≠ signed *error* of the contour |

- **Report's direct answers:**
  - Nobody has published a pLDDT-style binned self-error head for image segmentation, nor for 3D lesions.
  - Signed mm boundary error has not been predicted from features.
  - There is no PAE analogue for segmentation.
  - A controlled comparison of feature heads vs ensembles/TTA on lesions remains open.
- **Must also cite (missed by the report, verify):** Eppenhof & Pluim 2018, a CNN predicting local *registration* error in mm.
- **Residual risk:** the search has limits. Re-run a narrow arXiv check before submission.
- **Do not claim:** "confidence for segmentation", scalar quality prediction, or conformal intervals. All taken.

## 10. Evaluation and future work (short)

- **Core question:** does the network know its own mistakes better than an outside checker? Compare against:
  - a ruler (geometry only);
  - an outside checker (same head, input = CT + mask);
  - entropy, TTA, MC dropout, a 5-fold ensemble.
- **Metrics:**
  - bin calibration;
  - AUROC for |δ| > 2 mm and sign accuracy;
  - lesion rank correlation;
  - a review-budget curve;
  - everything split by size and source;
  - multi-reader check on LIDC-IDRI and KiTS21.
- **Future work:**
  - next-click suggestion (the head knows where the error is largest);
  - PAE analogue (global shift/scale vs local shape);
  - joint training with stop-gradient (AlphaFold pLDDT recipe) vs frozen;
  - out-of-fold K-fold for more head data;
  - an optional conformal wrapper.
