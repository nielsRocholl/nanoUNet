# Losses

nanoUNet defaults to standard nnU-Net **Dice + cross-entropy (DC+CE)** with optional deep supervision, matching the standard compound objective for voxel overlap plus calibration.

## CC-DiceCE (opt-in)

**When:** Multi-instance lesion problems with a heavy-tailed size distribution (many small lesions mixed with occasional large blobs). Rarely worthwhile for single-object tasks (whole organs or one tumour per patient).

**Idea:** Bouteille et al., ISBI 2026 — *Learning to Look Closer* ([arXiv:2511.17146](https://arxiv.org/abs/2511.17146)).

Let \(K\) be the set of 26-connected foreground components on the GT in the crop. Each component \(C \in K\) gets a Voronoi cell \(R_C\) (voxels closer to \(C\) than to any other component). The extra term averages component-wise DiceCE on that cell:

```
L_cc = λ · (1/|K|) · Σ_{C ∈ K}  DiceCE( P constrained to R_C ,  GT mask C )
```

Total loss: **L_global (full-volume DC+CE) + \(L_{\text{cc}}\)**. nanoUNet uses **λ = 1** by default.

The paper uses **ε = 0** (no additive smoothing on the **CC** Dice) because smoothed Dice was unstable on cerebral microbleeds; the global branch still keeps the usual nnU-Net smooth Dice (≈\(10^{-5}\)).

**CLI:**

```bash
nanounet_train -d <id> -f <fold> --plans <PlansBaseName> --loss cc_dc_ce
# same flag: -loss cc_dc_ce
```

Default remains `--loss dc_ce` (bit-for-bit compatible with prior runs that did not pass `--loss`).

## `batch_dice` and CC-DiceCE

`batch_dice` comes from your `plans.json` (`3d_fullres` block) and is forwarded into the underlying soft-Dice helpers.

For the **standard** Dice term it means aggregating Dice either across the spatial batch stacked as one pooled statistic (`True`) or per-sample mean (`False`). For CC-DiceCE we extend this to **pairs** `(slice, lesion_cc)` produced by the Voronoi split:

- `batch_dice: false` → per-pair Dice/CE scalars then **mean** over all lesion instances in the training batch.
- `batch_dice: true` → sum **TP / FP / FN** over all lesion instances across the GPU batch, divide once (**one pooled Dice**) and pool CE numerators/denominators the same way.

nanoUNet’s prompt-aware preset uses `batch_dice: false`; your runs inherit that behaviour unless you change the plan.

## Why not Blob Loss?

Blob loss masks other instances and averages per blob; Voronoi CC-DiceCE assigns every FP voxel to exactly one lesion cell, sharpening gradients toward small lesions. Bouteille et al. benchmark CC-DiceCE against blob-style objectives across heterogeneous brain lesion datasets inside nnU-Net and report favourable per-instance recall at equal global Dice.

## Cost / deps

Connected components + Euclidean distance Voronoi run on CPU with `connected-components-3d` and `scipy`; forward/backprop still execute on GPU. Expect roughly tens of milliseconds per crop for typical prompt-centred patches (`cc3d` + EDT dominates). No CuPy dependency.

## Caveats

- Opt-in objective: checkpoints are **not** comparable mix-and-match with pure DC+CE seeds.
- Prompt-centred crops often contain **few** lesions per crop; the marginal gain versus full-volume training may be modest.
- The published ablations are predominantly **MRI** brain lesion data — validate on CT before trusting transfer.
