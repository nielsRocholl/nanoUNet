# Patch size and universal lesion segmentation

This note ties together what the **patch size** knobs in nanoUNet do, what the literature implies for **tiny vs mega** lesions, and a practical playbook when you train **one** model on a lesion population that spans many spatial scales.

## What patch size actually controls

- Training patch spatial extent gates **effective field-of-view** seen per gradient step, together with anatomy context for large structures.
- It trades off against **batch size**, **VRAM**, and residual **spatial resolution** (how many pooling stages you tolerate before bottleneck).
- In nanoUNet you influence this through preprocessor flags documented in the main [README](../README.md):
  - `--patch-vol small|medium|large|xlarge` — starting isotropic edge before the VRAM shrink loop.
  - `--gpu-memory-gb` — VRAM budget the planner targets (use the **GPU you train on**, not a random CPU node).
- The ResEnc planner implementation is in [`nanounet/plan/planner_resenc.py`](../nanounet/plan/planner_resenc.py): it may **shrink** the patch further if footprint × network width exceeds VRAM **or** enlarge if memory allows starting from `--patch-vol`.
- Inference **sliding-window** spacing (`tile_step_size` in `configs/default.json`) interacts with patch size — smaller footprints mean more sliding tiles and higher compute cost for the same overlap policy.

Empirically patch size couples **everything**: "larger everywhere" rarely holds; you optimise the quadruple (FOV × memory × batch × pyramid depth).

## Evidence: huge FOV helps big anatomy segmentation

Primus benchmarks patch-size sensitivity on heterogeneous tasks ([arXiv:2503.01835](https://arxiv.org/abs/2503.01835)). On **Kidney Tumour Challenge 2023** (KiTS23) they report nnU-Net **Dice dropping ~34.9** points when shrinking the inference patch (~86.22 → ~51.33 in their Table 13 narrative). Interpretation: when the tumour + required kidney context routinely exceeds what fits in RAM at once, cropping starves contextual cues global Dice exploits.

Lesson: lesions that occupy a **significant slab of the axial field** need patches big enough that the tumour (plus minimum context) survives training without perpetual cropping artefacts.

## Evidence: finer spatial tiles help small lesion targets

Same Primus work shows regimes (e.g. Stanford Brain metastases / Brain MetShare table rows) where **smaller inference patches / finer tokens restore detail** versus aggressive downsampling footprints (see Tables ~7 / 13 in that paper).

CC-DiceCE adds instance-balanced gradients for small lesion instances ([arXiv:2511.17146](https://arxiv.org/abs/2511.17146)); it does **not** replace the need for enough native resolution/voxels covering the lesion in the cropped input.

Lesson: micronodular pathology benefits from freeing VRAM towards **spatial resolution increases** / **moderate footprints** tuned to lesion diameter — not blindly maximising naive isotropic superslabs if that forces batch size or pooling depth regressions.

## Why prompt-centric sampling shifts the calculus

nanoUNet’s prompt-aware cropping **anchors** crops on positive centroids whenever possible. Practical consequences:

- A **tiny** lesion is often centred and **fully observable** despite a moderately small patch footprint (the constraint becomes local texture + boundary context, not whether the tumour intersects the slab at all).
- A **mega** lesion violates the lesion-fits-patch assumption per training step — you repeatedly see slabs of the tumour. Global Dice then depends on tiling behaviour and whether border expansion catches exterior context at inference.
- Thus universal-lesion training is inherently **dual**: small lesion statistics push toward resolution + multi-instance fairness; gigantic lesions push toward **FOV completeness + context**.

## Recommendation playbook (tiny lesions → gigantic masses)

Prioritise empirical measurement over doctrine:

1. **Profile lesion volumes**: record per-connected-component voxel counts on resampled grids (training resolution). Inspect **p5 / p50 / p95 / max**.
2. If **p95** fits inside planner patch extents with comfortably **≥ ~20 % axial margin**, a **single-stage** `--patch-vol large` (256³-ish start) backbone is reasonable when VRAM permits — it aligns with classical nnU-Net defaults and keeps large tumour context sane.
3. If **mega lesions** truncate across many cases even after raising `--gpu-memory-gb`: either **train a second specialised model** on larger footprints (`xlarge` starter + maximal VRAM) or split workflow (whole-tumour model vs nodular refinement). Ensemble or cascade by lesion bounding-box volume inferred from prompts.
4. If the cohort is dominated by **sub-centimetre** metastases/micro bleeds/etc., freeing memory via `--patch-vol medium` (192³-ish) frequently improves **effective batch size**, stabilising gradients from auxiliary objectives like CC-DiceCE without sacrificing lesion-centred coverage.
5. Never assume `--patch-vol small` nails a literal small patch blindly — presets only initialise the shrink loop; finalisation still obeys [`planner_resenc.py`](../nanounet/plan/planner_resenc.py) feasibility checks.
6. Compare runs using tracked validation metrics (`val_dice_pos`, mode splits) alongside qualitative mega-lesion scans before locking production footprints.

Universal lesion segmentation seldom has one magic patch; expect **dual-scale thinking** unless your dataset concentrates in one tumour size bracket.
