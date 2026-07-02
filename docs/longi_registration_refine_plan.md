# Plan: body-mask metric + per-lesion VOI instance-optimization for BL→FU registration

> **Transcribe-only.** Every changed/new file is given in full or as an exact, unambiguous diff. All
> itk-elastix API used here was validated live against `itk-elastix 0.25.3` / `itk 5.4.6` in `.venv`.
> Obey `.cursor/rules/nanochat-style.mdc` (R1 <200 LOC/file, R5 no defensive try/except, R6 one top
> docstring, R10 why-comments, R12 no data fallbacks). Do **not** invent APIs or add files beyond
> those listed.

---

## Context

The standalone BL→FU warp pipeline (`nanounet/register/`, `nanounet/cli/register_longi.py`) already
warps the baseline CT + mask into the follow-up frame and derives clicks as connected-component
centroids of the warped mask. Initial results are okay. Three upgrades: fix a hard failure on
non-overlapping pairs (`0`), then improve **point propagation** — how precisely the propagated click
lands on the true FU lesion (`2`, `3`) — all without changing the set of output files:

0. **Geometric-center transform initialization (`0`, correctness fix — required).** Elastix optimises
   in **physical mm** and assumes the two volumes roughly overlap. Many BL/FU pairs do **not**: each
   scan keeps its own DICOM frame, so the bodies sit in disjoint world coordinates (verified —
   `Longitudinal_CT_v2_val/45fbc6d3e0_00`: FU z∈[114,1464] mm vs BL z∈[-1422,-322] mm, a ~1.6 m gap).
   Default elastix then samples **0** voxels inside the moving image and dies with
   `No valid voxels found to estimate the AdaptiveStochasticGradientDescent parameters`. The pair is
   perfectly registrable — it just needs a **bulk pre-alignment**. Setting
   `AutomaticTransformInitialization=true` + `AutomaticTransformInitializationMethod=GeometricalCenter`
   on the rigid map centres the two volumes before optimisation. Verified: unlocks `45fbc6d3e0_00`,
   harmless for already-overlapping pairs like `006f52e910_00` (it just centres first, then optimises).

1. **Body mask on the registration metric (`2`).** Whole-body CT-CT registration wastes metric
   budget on air/table/bed. Passing a body mask as elastix `fixed_mask` / `moving_mask` focuses the
   metric on tissue → better global alignment, robustly, at ~zero extra runtime.
2. **Per-lesion VOI instance optimization (`3`).** The global bspline grid (10 mm) is too stiff to
   remove local deformation (breathing, bowel) at the lesion. For each propagated click we crop a
   small VOI around it in the FU and the globally-warped-BL images, run a **local** rigid+bspline
   registration (instance optimization on that one lesion), resample the warped-BL mask VOI with the
   local transform, and take its centroid as the **refined click**. This directly attacks the ε /
   neighbour-drift residual documented in `docs/longitudinal_dwb_design.md` §5.2.

**Hard constraint from the user:** the VOI is a *temporary in-memory buffer*. Nothing about the VOI
is written to disk. The final written file set is **byte-for-byte the same set as today** — the same
`inputsTrBL/…nii.gz`, `targetsTrBL/…nii.gz`, `inputsTrFU`/`targetsTrFU` copies, `qc/…png`. **Only the
values inside `inputsTrBL/{stem}.json` (the clicks) change** (refined coordinates). Refinement never
touches the warped image or warped mask files (they stay the seam-free global warp — avoids the OOD
stamping problem §5.4).

### Why refined clicks may differ slightly from the written mask (accepted)
Clicks are the propagated FU **prompt**; the written BL mask is the co-located BL **stream** for DWB
image subtraction. They serve different roles and are allowed to sit a few voxels apart (training
already jitters prompts by σ=(2.75,5.19,5.40) vox). The refined click is a strictly better estimate
of the FU lesion centre than the global-mask centroid.

### Validated API facts (do not deviate)
- The default rigid map has **no** `AutomaticTransformInitialization` key; setting it plus
  `AutomaticTransformInitializationMethod="GeometricalCenter"` on that map centres the volumes first.
  Verified to fix the disjoint-frame `45fbc6d3e0_00` pair (rigid stage completes, warped HU range
  valid) and to leave overlapping pairs unaffected. Values are 1-tuples of str, e.g. `("true",)`.
- `itk.elastix_registration_method(fixed, moving, parameter_object=po, fixed_mask=fm, moving_mask=mm,
  log_to_console=False)` — masks are `itk.Image[itk.UC,3]` (uint8), same grid as their image. Works
  (verified; the earlier "no valid voxels" error was a degenerate synthetic image, not the masks).
- `itk.region_of_interest_image_filter(img, region_of_interest=reg)` with `reg = itk.ImageRegion[3]()`,
  `reg.SetIndex([x,y,z])`, `reg.SetSize([sx,sy,sz])` — crop preserves spacing/origin/direction. ITK
  index order is **(x,y,z)**; numpy arrays from `itk.array_from_image` are **(z,y,x)**.
- `resample_seg(seg_itk, tp)` (existing) applies a transform with NN. Reused for the local VOI mask.
- `scipy.ndimage`, `cc3d`, `numpy` all importable in `.venv`.

---

## File 1 — `nanounet/register/elastix.py`  (REPLACE ENTIRE FILE)

Adds `MIN_COMP_VOX` (moved here so both `warp_case` and `refine` import it cycle-free), `body_mask()`,
and `fixed_mask`/`moving_mask` passthrough on `register()`.

```python
"""Classical BL->FU registration via itk-elastix: rigid->affine->bspline multi-res, MI.

register() warps the moving image into the fixed frame; body_mask() builds a body-only mask so the
metric ignores air/table; resample_seg() re-applies a transform to a label map with nearest-neighbour
(FinalBSplineInterpolationOrder=0). MIN_COMP_VOX lives here so warp_case and refine share it.
"""

from __future__ import annotations

import cc3d
import itk
import numpy as np
import scipy.ndimage as ndi

MIN_COMP_VOX = 5  # drop connected-component speckle below this many voxels
BODY_HU = -300.0  # everything above this is body/tissue; below is air


def default_params() -> "itk.ParameterObject":
    po = itk.ParameterObject.New()
    # BL/FU scans often live in disjoint world frames (own DICOM origin) with no overlap in z; without
    # a bulk pre-alignment elastix samples 0 valid voxels and dies. Centre the volumes first.
    rigid = po.GetDefaultParameterMap("rigid")
    rigid["AutomaticTransformInitialization"] = ("true",)
    rigid["AutomaticTransformInitializationMethod"] = ("GeometricalCenter",)
    po.AddParameterMap(rigid)
    po.AddParameterMap(po.GetDefaultParameterMap("affine"))
    po.AddParameterMap(po.GetDefaultParameterMap("bspline"))
    return po


def body_mask(img: "itk.Image") -> "itk.Image":
    # Fill internal air (lungs) so the mask is the whole body, then keep the largest component to
    # drop the table/bed which usually sits in its own air-separated component.
    a = itk.array_from_image(img)
    m = ndi.binary_fill_holes(a > BODY_HU)
    lbl = cc3d.connected_components(m.astype(np.uint8))
    counts = np.bincount(lbl.reshape(-1))
    counts[0] = 0
    m = (lbl == int(counts.argmax())).astype(np.uint8)
    out = itk.image_from_array(m)
    out.CopyInformation(img)
    return out


def register(
    fixed: "itk.Image",
    moving: "itk.Image",
    *,
    fixed_mask: "itk.Image | None" = None,
    moving_mask: "itk.Image | None" = None,
    threads: int | None = None,
    verbose: bool = False,
):
    # fixed=FU, moving=BL, both itk.Image[itk.F]. warped is moving resampled into the fixed frame.
    kw: dict = dict(parameter_object=default_params(), log_to_console=verbose)
    if fixed_mask is not None:
        kw["fixed_mask"] = fixed_mask
    if moving_mask is not None:
        kw["moving_mask"] = moving_mask
    if threads is not None:
        kw["number_of_threads"] = threads
    warped, tp = itk.elastix_registration_method(fixed, moving, **kw)
    return warped, tp


def resample_seg(
    moving_seg: "itk.Image",
    tp: "itk.ParameterObject",
    *,
    verbose: bool = False,
) -> "itk.Image":
    # NN keeps the binary/label mask crisp; the flag must be set on every map in the transform.
    for i in range(tp.GetNumberOfParameterMaps()):
        tp.SetParameter(i, "FinalBSplineInterpolationOrder", "0")
    filt = itk.TransformixFilter.New(moving_seg, tp)
    filt.SetLogToConsole(verbose)
    filt.UpdateLargestPossibleRegion()
    return filt.GetOutput()
```

---

## File 2 — `nanounet/register/refine.py`  (NEW FILE, complete)

Per-lesion instance optimization. VOIs are in-memory only — never written.

```python
"""Per-lesion instance optimization: local rigid+bspline register in a VOI around each click,
resample the warped-BL mask VOI, and take its centroid as the refined FU click.

VOIs are in-memory buffers only; nothing here writes to disk. A lesion whose local registration
does not converge or whose refined mask empties keeps its global click (convergence fallback, not a
missing-data fallback).
"""

from __future__ import annotations

import itk
import numpy as np

from nanounet.register.elastix import MIN_COMP_VOX, resample_seg

VOI_HALF_MM = 30.0  # physical half-extent of the refinement box around each click


def local_params() -> "itk.ParameterObject":
    # Rigid captures residual translation; a tight bspline grid captures local breathing/deformation
    # the stiff 10 mm global grid cannot.
    po = itk.ParameterObject.New()
    po.AddParameterMap(po.GetDefaultParameterMap("rigid"))
    b = po.GetDefaultParameterMap("bspline")
    b["FinalGridSpacingInPhysicalUnits"] = ("6.0",)
    po.AddParameterMap(b)
    return po


def refine_clicks(
    fu: "itk.Image",
    warped_bl: "itk.Image",
    warped_seg: np.ndarray,
    clicks_xyz: list,
    *,
    verbose: bool = False,
) -> list:
    sp = np.asarray(fu.GetSpacing(), dtype=float)          # (x, y, z) mm
    half = np.maximum(1, np.round(VOI_HALF_MM / sp)).astype(int)  # (hx, hy, hz) vox
    nz, ny, nx = warped_seg.shape
    dims = np.array([nx, ny, nz])
    out: list = []
    for cx, cy, cz in clicks_xyz:
        c = np.array([cx, cy, cz], dtype=float)
        lo = np.maximum(0, np.floor(c - half)).astype(int)
        hi = np.minimum(dims, np.ceil(c + half) + 1).astype(int)
        seg_voi = warped_seg[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
        if int(seg_voi.sum()) < MIN_COMP_VOX:
            out.append([cx, cy, cz])
            continue

        reg = itk.ImageRegion[3]()
        reg.SetIndex([int(lo[0]), int(lo[1]), int(lo[2])])
        reg.SetSize([int(hi[0] - lo[0]), int(hi[1] - lo[1]), int(hi[2] - lo[2])])
        fu_voi = itk.region_of_interest_image_filter(fu, region_of_interest=reg)
        bl_voi = itk.region_of_interest_image_filter(warped_bl, region_of_interest=reg)
        seg_voi_itk = itk.image_from_array(seg_voi.astype(np.float32))
        seg_voi_itk.CopyInformation(bl_voi)

        try:
            _, tpl = itk.elastix_registration_method(
                fu_voi, bl_voi, parameter_object=local_params(), log_to_console=verbose
            )
        except RuntimeError:
            out.append([cx, cy, cz])  # local reg did not converge → keep global click
            continue

        rseg = itk.array_from_image(resample_seg(seg_voi_itk, tpl, verbose=verbose)) > 0.5
        if int(rseg.sum()) < MIN_COMP_VOX:
            out.append([cx, cy, cz])
            continue
        zz, yy, xx = np.nonzero(rseg)
        out.append([float(xx.mean() + lo[0]), float(yy.mean() + lo[1]), float(zz.mean() + lo[2])])
    return out
```

> The single `except RuntimeError` is **not** defensive log-and-reraise (R5). elastix raises when the
> local ASGD cannot estimate parameters on a low-texture VOI; the correct algorithmic response is to
> keep that lesion's already-valid global click. It is a two-branch `if`, not a swallow.

---

## File 3 — `nanounet/register/warp_case.py`  (EXACT EDITS)

**3a. Import line** — replace the existing import block near the top:
```python
from nanounet.register.elastix import register, resample_seg
```
with:
```python
from nanounet.register.elastix import MIN_COMP_VOX, body_mask, register, resample_seg
from nanounet.register.refine import refine_clicks
```

**3b. Delete** the now-duplicated module constant line:
```python
MIN_COMP_VOX = 5  # drop resampling speckle below this many voxels
```
(`MIN_COMP_VOX` now comes from `elastix`.) Keep the `StepFn = Callable[[str], None]` line.

**3c. `warp_case` signature** — add two flags (default on):
```python
def warp_case(
    data_root: str,
    pid: str,
    idx: str,
    *,
    body_mask_metric: bool = True,
    refine: bool = True,
    threads: int | None = None,
    verbose: bool = False,
    on_step: StepFn | None = None,
) -> WarpResult:
```

**3d. Register step** — replace:
```python
    step("register")
    warped_img, tp = register(fu, bl, threads=threads, verbose=verbose)
```
with:
```python
    step("register")
    fu_mask = body_mask(fu) if body_mask_metric else None
    bl_mask = body_mask(bl) if body_mask_metric else None
    warped_img, tp = register(
        fu, bl, fixed_mask=fu_mask, moving_mask=bl_mask, threads=threads, verbose=verbose
    )
```

**3e. After the existing `clicks` loop** (right before `return WarpResult(...)`), insert:
```python
    if refine:
        step("refine")
        clicks_xyz = refine_clicks(fu, warped_img, seg, clicks_xyz, verbose=verbose)
```
Leave the `return WarpResult(warped_img, seg, clicks_xyz, fu, bl_seg_vox)` line unchanged — `seg`
(the written mask) is **not** modified by refinement, only `clicks_xyz`.

---

## File 4 — `nanounet/cli/register_longi.py`  (EXACT EDITS)

**4a. `_STAGES`** — insert a `refine` stage after `clicks`:
```python
_STAGES = (
    ("load", "load FU / BL / mask"),
    ("register", "elastix rigid → affine → bspline (FU ← BL), body-masked metric"),
    ("mask", "transformix BL mask (nearest-neighbour)"),
    ("clicks", "cc3d centroids on warped mask"),
    ("refine", "per-lesion VOI instance optimization"),
    ("write", "write dataset layout + copy FU"),
    ("qc", "QC montage PNG"),
)
```

**4b. argparse** — add two off-switches after the `--qc` argument:
```python
    ap.add_argument("--no-body-mask", action="store_true", help="disable body-masked metric")
    ap.add_argument("--no-refine", action="store_true", help="disable per-lesion VOI refinement")
```

**4c. Stage-count math** — the `refine` stage is optional too. Replace:
```python
    n_stages = len(_STAGES) if args.qc else len(_STAGES) - 1
```
with:
```python
    n_stages = len(_STAGES)
    if not args.qc:
        n_stages -= 1
    if args.no_refine:
        n_stages -= 1
```
`on_step` uses `stage_idx[name]` for `prog.update(completed=i)`; with `refine` skipped the bar simply
never reports that index — harmless (the final `completed=n_stages` still lands on "done"). No other
change needed there.

**4d. `warp_case` call** — pass the flags:
```python
            res = warp_case(
                args.data_root,
                args.pid,
                args.idx,
                body_mask_metric=not args.no_body_mask,
                refine=not args.no_refine,
                threads=threads,
                verbose=args.verbose,
                on_step=on_step,
            )
```

No change to `_write_dataset`, `_write_points`, `_qc_png`, or the summary table: the written file set
is identical; only the click coordinates inside `inputsTrBL/{stem}.json` differ.

---

## File 5 — `pyproject.toml`
No change. `itk-elastix`, `scipy`, `connected-components-3d` are already declared/installed.

---

## Non-obvious gotchas (already handled above — do not re-introduce)
- **Init only on the rigid map.** Put the two `AutomaticTransformInitialization*` keys on the **rigid**
  map only (the first stage); affine/bspline inherit its result. Do not add them to `local_params()`
  in `refine.py` — VOIs are cropped from already-globally-aligned, same-grid images and overlap by
  construction, so local init is unnecessary.
- **ITK index (x,y,z) vs numpy (z,y,x).** `ImageRegion.SetIndex/SetSize` take (x,y,z); `warped_seg`
  is (z,y,x). `refine.py` crops the numpy mask with `[z, y, x]` slices and the itk VOIs with (x,y,z)
  — keep both.
- **VOI shares the FU grid.** `warped_bl`, `warped_seg`, `fu` are all on the FU grid, so a voxel
  index inside the VOI maps to global by adding `lo` (`+lo[0]` for x, etc). No physical-coordinate
  math is needed; do not introduce `TransformIndexToPhysicalPoint` here.
- **No disk temp for VOIs.** Everything in `refine_clicks` is in-memory itk/numpy; nothing is written.
  If you feel tempted to `itk.imwrite` a VOI for debugging, don't — it violates the "same files"
  constraint. Use `--verbose` instead.
- **Refinement edits clicks only.** Never write the refined VOI mask back into `warped_seg`/the mask
  file (that would create stamping seams §5.4 and change an output file).
- **Body mask keeps largest component** — assumes the patient body is the largest air-separated
  component. True for these whole-body CTs; if a future dataset has the patient touching the table
  through a dense pad, revisit. Not a concern for Longitudinal_CT_v2.

---

## Verification (one sample, before/after)

1. **Baseline (refine off), for comparison:**
   ```
   .venv/bin/python -m nanounet.cli.register_longi \
     --data-root "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/Longitudinal_CT_v2" \
     --pid 006f52e910 --idx 00 \
     --out "<SCRATCH>/reg_noref" --qc --no-refine --no-body-mask
   ```
2. **Full pipeline (body mask + refine on, the new default):**
   ```
   .venv/bin/python -m nanounet.cli.register_longi \
     --data-root "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/Longitudinal_CT_v2" \
     --pid 006f52e910 --idx 00 \
     --out "<SCRATCH>/reg_full" --qc
   ```
   (`<SCRATCH>` = the session scratchpad dir.)
3. **Same file set** in both out dirs: `inputsTrBL/006f52e910_00.{nii.gz,json}`,
   `targetsTrBL/006f52e910_00.nii.gz`, `inputsTrFU/*`, `targetsTrFU/*`, `meta/006f52e910.csv`,
   `qc/006f52e910_00_qc.png`. Confirm the new run produced **no extra VOI files**.
4. **Clicks moved, mask identical:** diff the two `inputsTrBL/006f52e910_00.json` — click coordinates
   should differ by a few voxels (refinement worked). The two `targetsTrBL/006f52e910_00.nii.gz`
   should be **identical** when only `--no-refine` differs between runs (refinement never touches the
   mask); they differ only when `--no-body-mask` also flips (better global warp).
5. **QC:** open `reg_full/qc/006f52e910_00_qc.png`. Yellow click ×-marks should sit on the FU lesions;
   red warped-BL-mask contour aligns with FU anatomy at least as well as `reg_noref`.
6. **Disjoint-frame regression case (the `0` fix):** run the pair that failed before —
   ```
   .venv/bin/python -m nanounet.cli.register_longi \
     --data-root "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/Longitudinal_CT_v2_val" \
     --pid 45fbc6d3e0 --idx 00 --out "<SCRATCH>/reg_disjoint" --qc
   ```
   It must now **complete** (no "No valid voxels" error) and the QC must show the warped BL body
   overlapping FU. Without the `default_params` init change this run dies — that is the acceptance
   test for feature `0`.
7. **Quantify ε (temporary snippet, delete after — R16):** for the paired subset, compute the mean
   distance between each refined click and the nearest **FU** lesion centroid (`targetsTrFU` cc3d
   centroids), for `reg_noref` vs `reg_full`. Expect the mean/tail distance to drop with refinement.
   This is the number that says whether the instance-optimization step actually helped.

---

## Out of scope (note only)
Batch-warping the corpus, pairing refined clicks to CSV `cog_bl` identity, splicing refined VOIs back
into the image/mask (rejected here to avoid seams), and wiring the warped corpus into the two-stream
DataModule. Separate follow-ups once refinement is shown to reduce ε on this sample.
