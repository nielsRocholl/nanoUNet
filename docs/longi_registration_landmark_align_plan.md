# Plan: landmark-rigid pre-alignment for disjoint-frame BL→FU registration

> **Transcribe-only.** Every changed/new function is given in full or as an exact, unambiguous diff.
> All itk API used here was validated live against `itk 5.4.6` / `itk-elastix 0.25.3` in `.venv` on the
> real failing case `Longitudinal_CT_v2/45fbc6d3e0_00`. Obey `.cursor/rules/nanochat-style.mdc`
> (R1 <200 LOC/file, R5 no defensive try/except, R6 one top docstring, R10 why-comments, R12 no data
> fallbacks). Do **not** invent APIs or add files beyond those listed.

---

## Context — what is still broken

`nanounet/register/` already handles the disjoint-frame case (BL and FU in non-overlapping DICOM
world frames). The current fix, in `warp_case.py`, is:

1. detect `frame_z_overlap_mm(fu, bl) <= 0`,
2. compute a **mean physical translation** between paired lesion clicks (`click_translation_mm`),
3. `shift_origin` BL + BL mask by that translation,
4. run elastix with `geometric_init=False`.

On `45fbc6d3e0_00` this lifts body-overlap correlation from ~0.15 (geometric-center only) to ~0.61.
**It is not enough.** A pure translation cannot correct the patient being *rotated* between the two
acquisitions, and it cannot use the full geometry of the 9 paired lesion clicks.

### Measured root cause (live, on the real case)
Paired lesion clicks → physical points (`TransformIndexToPhysicalPoint`), then residual of the
BL↔FU correspondence under each model:

| pre-alignment model | mean click residual | max click residual |
|---|---|---|
| translation-only (current `click_translation_mm`) | **20.1 mm** | **45.1 mm** |
| **rigid (rotation + translation) landmark fit** | **7.9 mm** | **13.3 mm** |

The landmark rigid fit recovers a **5.3° rotation** between the scans (real patient repositioning) and
cuts the residual **~2.5×**. The 9 clicks are well spread in 3-D (centered-cloud singular values
`[691.6, 181.3, 107.8] mm`), so a rigid fit is well-conditioned here. Feeding the residual-8 mm
pre-aligned pair into the *existing* elastix rigid→affine→bspline then lets the deformable stage lock
on properly instead of fighting a 2 cm bulk error.

**The fix:** replace the translation-only origin shift (steps 2–3 above) with a **landmark-based rigid
transform** estimated from the paired clicks, applied by resampling BL (linear) and the BL mask
(nearest-neighbour) onto the FU grid. Everything downstream (body-mask metric, elastix, transformix,
click centroids, per-lesion refinement, written file set) is **unchanged**.

### Scope guard — only the disjoint branch changes
Overlapping pairs (`frame_z_overlap_mm > 0`, e.g. `006f52e910_00` at corr ~0.78) keep using
`geometric_init=True` + elastix and are **not touched**. Do not route them through the landmark path.

---

## Validated API facts (do not deviate)

All confirmed live in `.venv`:

- **Landmark fit.** `T = itk.VersorRigid3DTransform[itk.D].New()`;
  `Init = itk.LandmarkBasedTransformInitializer[itk.Transform[itk.D, 3, 3]].New()` (the initializer is
  wrapped **only** on the transform-base type `itk.Transform[itk.D,3,3]`, *not* on image types — the
  image-typed template raises `TemplateTypeError`). Landmark containers are
  `v = itk.vector[itk.Point[itk.D, 3]]()` filled with `v.push_back(itk.Point[itk.D, 3]([x, y, z]))`.
  Call `Init.SetFixedLandmarks(...)`, `Init.SetMovingLandmarks(...)`, `Init.SetTransform(T)`,
  `Init.InitializeTransform()`.
- **Direction.** With `fixed = FU points`, `moving = BL points`, the resulting `T` maps
  **FU (fixed) → BL (moving)** — verified: `T.TransformPoint(FU_i) == BL_i`. This is exactly the
  direction `ResampleImageFilter` needs to pull BL into the FU frame (for each FU output voxel it
  samples BL at `T(voxel)`).
- **Degenerate landmark counts.** With 1 or 2 points the initializer returns a **pure translation**
  (rotation = identity) and does **not** raise. With ≥3 **collinear** points it can return a spurious
  180° flip (verified: 3 collinear points → rotation matrix diag `[1, -1, -1]`). → a conditioning
  guard is mandatory (below).
- **Resampling.** `itk.ResampleImageFilter.New(Input=img, Transform=T, ReferenceImage=fu,
  UseReferenceImage=True)` puts the output on the FU grid (spacing/origin/direction/size of FU);
  `SetDefaultPixelValue(-1000.0)` fills out-of-BL air. For the label map pass
  `Interpolator=itk.NearestNeighborInterpolateImageFunction.New(bl_seg)` and
  `SetDefaultPixelValue(0.0)`. Default (no interpolator arg) is linear — used for the CT. Verified end
  to end on the real case: aligned BL body-overlap corr improves and the warped mask stays a compact
  label blob.
- **Conditioning.** `np.linalg.svd(F - F.mean(0), compute_uv=False)` returns the 3 singular values of
  the centered fixed-point cloud (units mm). `sv[1]` (the second value) measures spread off the
  principal axis: large ⇒ points are not collinear ⇒ rotation is trustworthy. On the real case
  `sv[1] = 181 mm`.
- **elastix / transformix reuse.** `register(...)`, `resample_seg(...)`, `body_mask(...)` are unchanged
  and already validated in the prior plan. The aligned BL is a normal CT (air outside body), so
  `body_mask` behaves identically.

---

## File 1 — `nanounet/register/elastix.py`  (EXACT EDITS)

Add landmark-rigid alignment; drop the now-unused translation-only helpers. Keep
`frame_z_overlap_mm`, `default_params`, `body_mask`, `register`, `resample_seg`.

**1a. Module docstring** — replace the existing top docstring with:
```python
"""Classical BL->FU registration via itk-elastix: rigid->affine->bspline multi-res, MI.

register() warps the moving image into the fixed frame; body_mask() builds a body-only mask so the
metric ignores air/table; resample_seg() re-applies a transform to a label map with nearest-neighbour
(FinalBSplineInterpolationOrder=0). landmark_align() bulk-aligns a disjoint-frame BL onto the FU grid
from paired lesion clicks (rigid when the clicks are well spread, else translation). MIN_COMP_VOX lives
here so warp_case and refine share it.
"""
```

**1b. Constants** — after the existing `BODY_HU = -300.0` line, add:
```python
MIN_LANDMARKS = 3      # rigid rotation needs >= 3 non-collinear clicks; below this, translation only
MIN_SPREAD_MM = 50.0   # 2nd singular value of the centred click cloud; below this the clicks are
                       # ~collinear and the landmark rotation is untrustworthy (can flip 180 deg)
```

**1c. Delete** the two now-unused helpers in full — `click_translation_mm(...)` (its `def` through its
`return np.mean(deltas, axis=0)`) and `shift_origin(...)` (its `def` through its `return out`). They
are replaced by `landmark_align`. `frame_z_overlap_mm` stays.

**1d. Insert** `landmark_align` (place it where `click_translation_mm` / `shift_origin` were, i.e.
after `frame_z_overlap_mm` and before `default_params`):
```python
def landmark_align(
    fu: "itk.Image",
    bl: "itk.Image",
    bl_seg: "itk.Image",
    fu_pts: list,
    bl_pts: list,
):
    # Disjoint DICOM frames: estimate a rigid FU<-BL bulk transform from the paired lesion clicks and
    # resample BL + mask onto the FU grid. Rigid (not just translation) recovers the ~5 deg patient
    # rotation between scans; the landmark fit maps FU (fixed) -> BL (moving), the direction a
    # resampler samples in. Clicks are in each scan's native voxel frame → physical mm first.
    def phys(img: "itk.Image", pts: list) -> "np.ndarray":
        return np.array(
            [img.TransformIndexToPhysicalPoint([int(round(c)) for c in p]) for p in pts], dtype=float
        )

    f = phys(fu, fu_pts)
    b = phys(bl, bl_pts)
    tf = itk.VersorRigid3DTransform[itk.D].New()
    spread = np.linalg.svd(f - f.mean(axis=0), compute_uv=False)
    if len(f) >= MIN_LANDMARKS and spread[1] >= MIN_SPREAD_MM:
        init = itk.LandmarkBasedTransformInitializer[itk.Transform[itk.D, 3, 3]].New()
        init.SetFixedLandmarks(_landmarks(f))
        init.SetMovingLandmarks(_landmarks(b))
        init.SetTransform(tf)
        init.InitializeTransform()
    else:
        # too few / collinear clicks: rotation is ill-conditioned, use the robust median translation
        tf.SetTranslation([float(x) for x in np.median(b - f, axis=0)])

    bl_al = _resample(bl, tf, fu, default=-1000.0)  # linear; air fills outside the BL body
    nn = itk.NearestNeighborInterpolateImageFunction.New(bl_seg)
    seg_al = _resample(bl_seg, tf, fu, default=0.0, interp=nn)
    return bl_al, seg_al


def _landmarks(pts: "np.ndarray"):
    v = itk.vector[itk.Point[itk.D, 3]]()
    for p in pts:
        v.push_back(itk.Point[itk.D, 3]([float(p[0]), float(p[1]), float(p[2])]))
    return v


def _resample(img, tf, ref, *, default: float, interp=None):
    kw = dict(Input=img, Transform=tf, ReferenceImage=ref, UseReferenceImage=True)
    if interp is not None:
        kw["Interpolator"] = interp
    r = itk.ResampleImageFilter.New(**kw)
    r.SetDefaultPixelValue(default)
    r.Update()
    return r.GetOutput()
```

> `landmark_align` + `_landmarks` + `_resample` are ~40 LOC; deleting `click_translation_mm` +
> `shift_origin` removes ~25. The file stays well under 200 LOC (R1). `_landmarks` / `_resample` are
> tiny private siblings of their only caller, not new files (R2/R4).

---

## File 2 — `nanounet/register/warp_case.py`  (EXACT EDITS)

**2a. Import block** — replace:
```python
from nanounet.register.elastix import (
    MIN_COMP_VOX,
    body_mask,
    click_translation_mm,
    frame_z_overlap_mm,
    register,
    resample_seg,
    shift_origin,
)
```
with:
```python
from nanounet.register.elastix import (
    MIN_COMP_VOX,
    body_mask,
    frame_z_overlap_mm,
    landmark_align,
    register,
    resample_seg,
)
```

**2b. Disjoint branch** — replace the current block:
```python
        t = click_translation_mm(fu, bl, fu_pts, bl_pts)
        bl = shift_origin(bl, t)
        bl_seg = shift_origin(bl_seg, t)
        geometric_init = False
```
with:
```python
        bl, bl_seg = landmark_align(fu, bl, bl_seg, fu_pts, bl_pts)
        geometric_init = False
```

Everything above it in that branch is unchanged: the `frame_z_overlap_mm(fu, bl) <= 0` guard, loading
`fu_pts` / `bl_pts` from the `.json` sidecars, the
`assert len(fu_pts) == len(bl_pts) and len(fu_pts) > 0`. That assert stays — the clicks are the paired
correspondence `landmark_align` fits, and equal ordered length is the precondition (R15, crash on bad
input; do **not** soften it). After this branch `bl` and `bl_seg` are itk images on the **FU grid**;
the rest of `warp_case` (body_mask, register with `geometric_init=False`, resample_seg, cc3d centroids,
refine) consumes them exactly as before.

---

## File 3 — `nanounet/cli/register_longi.py`
**No change.** Stage list, flags, stage-count math, and the written file set are all unchanged — the
landmark fit happens inside the existing `register` stage. (Optional cosmetic only, not required: the
`register` stage description already reads "elastix rigid → affine → bspline (FU ← BL), body-masked
metric"; leave it.)

## File 4 — `pyproject.toml`
**No change.** `itk` / `itk-elastix` already provide `VersorRigid3DTransform`,
`LandmarkBasedTransformInitializer`, `ResampleImageFilter`,
`NearestNeighborInterpolateImageFunction` (all confirmed present in `.venv`).

---

## Non-obvious gotchas (already handled above — do not re-introduce)
- **Fit direction is FU→BL.** `SetFixedLandmarks(FU)`, `SetMovingLandmarks(BL)`. The resulting `T` maps
  FU→BL, which is what `ResampleImageFilter` (sampling the moving image at `T(fixed_voxel)`) needs. Do
  **not** swap fixed/moving or invert `T`.
- **Conditioning guard is mandatory.** Without the `len(f) >= MIN_LANDMARKS and spread[1] >=
  MIN_SPREAD_MM` check, a case with 3 collinear clicks (e.g. lesions strung along the spine) yields a
  spurious 180° rotation and a catastrophic warp. When ill-conditioned, fall back to the **median**
  translation (robust to a single mis-clicked lesion; do not use the mean).
- **Two default pixel values.** BL CT → `-1000.0` (air) so out-of-body regions read as air for the
  body-mask metric; BL mask → `0.0` (background). Swapping them corrupts the mask or injects a
  soft-tissue-valued border.
- **NN only for the mask.** The CT uses the default linear interpolator; the label map uses
  `NearestNeighborInterpolateImageFunction` so labels are not blurred to fractional values.
- **`geometric_init=False` on the disjoint path.** BL is already on the FU grid and bulk-aligned after
  `landmark_align`; leaving `geometric_init=True` would let elastix re-centre bounding boxes and
  partially undo the landmark alignment. Overlapping pairs keep `geometric_init=True`.
- **Clicks are re-derived after warping.** `landmark_align` moves the *image and mask*, not the output
  clicks. Output clicks are still the cc3d centroids of the elastix-warped mask (then VOI-refined),
  exactly as today. The paired input clicks are used **only** to estimate the rigid transform.
- **One extra interpolation on the disjoint path only.** The written warped BL is now BL resampled
  twice (landmark rigid, then elastix). This is limited to disjoint-frame cases and is acceptable for a
  moving image; do not attempt to compose the transforms to save an interpolation (elastix external
  initial transforms are silently ignored in the multi-stage functional API — verified — so that path
  is a trap, not a shortcut).

---

## Verification (real cases, before/after)

1. **Disjoint case — the target (`45fbc6d3e0_00`):**
   ```
   .venv/bin/python -m nanounet.cli.register_longi \
     --data-root "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/Longitudinal_CT_v2" \
     --pid 45fbc6d3e0 --idx 00 --out "<SCRATCH>/reg_landmark" --qc
   ```
   Must **complete** and the QC `warped BL` column must show anatomy at the **same axial level and
   orientation** as FU (no melted pelvis). Expect body-overlap correlation and click-on-lesion accuracy
   clearly better than the translation-only build (which topped out at ~0.61).
2. **Overlapping case — regression guard (`006f52e910_00`):**
   ```
   .venv/bin/python -m nanounet.cli.register_longi \
     --data-root "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/Longitudinal_CT_v2" \
     --pid 006f52e910 --idx 00 --out "<SCRATCH>/reg_overlap" --qc
   ```
   Must be **unchanged** vs the current build (it never enters the landmark branch; corr ~0.78).
3. **Same written file set** as today in both out dirs: `inputsTrBL/{stem}.{nii.gz,json}`,
   `targetsTrBL/{stem}.nii.gz`, `inputsTrFU/*`, `targetsTrFU/*`, `meta/{pid}.csv`, `qc/{stem}_qc.png`.
   No extra files.
4. **Conditioning-guard temporary check (delete after — R16):** print `spread` inside `landmark_align`
   for a few cases; confirm the disjoint target uses the **rigid** branch (`spread[1] ≈ 181 mm` on
   `45fbc6d3e0_00`) and that any pathological ≤2-click / collinear case would take the translation
   branch without raising.
5. **Residual metric (temporary snippet, delete after — R16):** for the disjoint target, compute the
   mean/max distance between `T(FU_click_i)` and `BL_click_i` under the rigid fit vs the old mean
   translation. Expect the mean to drop from ~20 mm to ~8 mm (the number that says the rotation
   recovery actually helped).

---

## Out of scope (note only)
Batch-warping the corpus, similarity/affine landmark fits (rejected: 7/12 DOF overfit noisy centroid
clicks — rigid is the robust choice and elastix's affine+bspline recovers scale/shear from intensities),
composing the landmark and elastix transforms into a single resample, and any change to the overlapping
-frame path. Separate follow-ups once the rigid pre-alignment is shown to fix the disjoint corpus.
