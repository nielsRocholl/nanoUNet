# segtrack correctness + live registration plan

Status: **plan only, nothing in this doc is implemented yet.** Written for a fresh coding
agent with no memory of the conversation that produced it. Read this whole document before
touching any code — later sections depend on context established earlier.

## 0. Why this plan exists (read first)

`nanounet_segtrack` (nanoUNet) predicts a binary lesion mask per scan, turns it into
per-lesion instance IDs, and calls a GNN matcher (lesion-tracking) to link baseline (BL)
lesions to follow-up (FU) lesions. A prior investigation this session found the tool was
silently producing near-zero ID-matched Dice on a full held-out test set (median
per-lesion Dice **0.000** over 549 GT lesions) despite the underlying segmentation being
fine (detected lesions scored 0.73–0.79 Dice). Root cause: **the step that turns a binary
mask into per-lesion instance IDs was throwing away correctly-segmented lesions whenever a
click coordinate didn't land exactly on the predicted blob.** That step, and why it's wrong,
is explained fully in Phase 1 below.

Separately, the user wants `nanounet_segtrack` to be a **one-command tool that works for
anyone**, not just on this project's own `Longitudinal-CT` dataset. Today the matcher
requires a CSV of pre-computed "BL lesion propagated into FU space" coordinates
(`meta/*.csv`, column `cog_propagated`) that was built once, offline, via a registration
pipeline this project ran previously. A user who only has two CT scans and click JSONs —
no such CSV — cannot run the tool at all (non-`drop_dp` matcher checkpoints hard-crash
without it). Phase 3–4 replace that hard requirement with an on-the-fly registration
fallback using [uniGradICON](https://github.com/uncbiag/uniGradICON), a pretrained
deep-learning registration network, so the tool degrades gracefully to "register on demand"
instead of crashing.

Three independent fixes, bundled into one plan because they touch overlapping code:

| # | Phase | Fixes |
|---|---|---|
| 1 | Instance labeling | Stop dropping correctly-segmented lesions; same fix serves both FU and BL |
| 2 | EMA default | `nanounet_segtrack` silently used non-EMA seg weights by default |
| 3+4 | Live registration | Let segtrack work with no CSV at all, and handle segmentation splitting one historical lesion into multiple pieces |

**Repos involved** — two separate git repos on this machine, both required:
- `/nanoUNet` — the segmentation CLI (`nanounet_segtrack`, `nanounet_predict`).
- `/lesion-tracking` — the GNN tracker package, imported by nanoUNet as `tracking.*` (installed editable: `pip install -e /lesion-tracking`).

Every file path below is prefixed `[nanoUNet]` or `[lesion-tracking]` to say which repo it's in.

## 1. Conventions you must not get backwards

These conventions are used consistently across both repos. Getting any of them backwards
will silently corrupt coordinates rather than crash — verify against the numeric checks in
each phase rather than trusting derivation alone.

- **Voxel-index coordinates are always `(z, y, x)`.** Every function in this codebase that
  takes/returns a bare coordinate tuple (`load_clicks`, `centroids`, `cog_bl`/`cog_fu`/
  `cog_propagated` in `meta/*.csv`, `parse_zyx`) uses array-index order `(z, y, x)`, i.e.
  numpy's own indexing order for a volume loaded as `arr[z, y, x]`.
- **ITK physical-space `Point`/`Index`/`ContinuousIndex` objects are always `(x, y, z)`.**
  This is the opposite order from the above, and is a real ITK convention (not a bug to
  "fix") — `itk.Image.GetSpacing()`, `.GetOrigin()`, `.TransformContinuousIndexToPhysicalPoint`,
  etc. all use `(x, y, z)`. Any code converting between voxel-index and ITK point/index
  types must explicitly reverse the triple. **Get this backwards and every produced
  coordinate silently lands in the wrong place** (not a crash) — this is why Phase 3
  mandates a numeric validation gate before wiring anything into the main pipeline.
- **`click JSON` `"point"` field is `[x, y, z]`** (see `[lesion-tracking] tracking/data/instances.py:load_clicks`, which does `p[2], p[1], p[0]` to build the `(z,y,x)` tuple it returns) — already consistent with the physical-space-order convention above, since these values ultimately came from a physical-space pipeline even though downstream code treats them as voxel indices in the resampled native grid (that native grid was built with 1 array unit = 1 voxel, so this works out, but don't assume `"point"` fields elsewhere follow the same order without checking).
- **BL ids are canonical, FU ids are provisional until tracked.** `docs/reference/track_ids.md`: "BL click names are canonical... the FU mask is remapped after matching." The final published `fu.mha` never uses whatever id `label_instances` (Phase 1) assigned FU pre-tracking — `paint_fu`/`fu_track_map` (`[lesion-tracking] tracking/data/paint.py`, unchanged by this plan) always overwrite it. This is *why* Phase 1 can safely stop trying to make FU's provisional id "correct" — it is never the final answer.

## 2. Phase 1 — stop dropping correctly-segmented lesions

### 2.1 What's wrong today

`[lesion-tracking] tracking/data/instances.py::binary_to_instances(pred, clicks_zyx)`:
runs connected-components on the binary mask, then for each click looks up the label at
the click's *exact* voxel. If that voxel is background, **the entire connected component
is dropped** (never written to the output — `lut` stays `0` for that component, even though
it may be a large, correctly-segmented lesion).

This is called for **FU always**, and for **BL only when `--bl-mask-dir` is not given**
(`[nanoUNet] nanounet/infer/segtrack.py::run_case`, lines ~97 and ~115-116). When
`--bl-mask-dir` *is* given, BL instead uses `load_instance_zyx` and reads GT mask labels
directly — Phase 1 does not touch that branch at all.

Two things make "click lands exactly on FG" a bad test, confirmed against real data this
session:
1. **FU's click coordinate is a registration estimate, not ground truth.** Verified
   directly: `inputsTrFU/{stem}.json`'s point for a given lesion id is numerically identical
   to `cog_propagated` in `meta/{pid}.csv` (BL centroid warped into FU space by
   registration), not `cog_fu` (the true annotated FU location). This is intentional — using
   the true location would leak ground truth into what's meant to be a realistic
   "where might this be now" prompt — but it means the click can be several mm to over a
   centimeter off the segmented lesion's actual location, and no fixed tolerance fully
   fixes that (small lesions have less margin than the registration error).
2. **The matcher (the GNN) doesn't need the click's identity for FU at all.** Verified by
   reading `[lesion-tracking] tracking/data/masks.py::build_mask_graph`: FU node positions
   come from `centroids(mk_fu, fu_ids)` — the **predicted mask's own connected-component
   centroid** — never from the click file. `fu_ids = _labels(mk_fu)` just takes whatever
   integer labels exist in the mask; their numeric value is irrelevant to the matcher. So
   filtering FU CCs by click identity before the matcher even runs discards real detections
   for no benefit — the matcher was never going to use that identity anyway.

For BL (non-`--bl-mask-dir` path), identity *does* matter — BL ids are canonical/final — but
the same "one CC per lesion, or it's dropped" limitation applies if the model over-segments
a historical lesion into multiple disconnected pieces: only the piece the click happens to
land on survives today; the rest are silently lost.

### 2.2 The fix

Replace `binary_to_instances` with a function that **never drops a connected component**.
A CC a click lands on exactly keeps that click's id (unchanged behavior — still correct,
still needed for BL identity). Any CC with no click on it keeps its own fresh id instead of
being deleted.

**`[lesion-tracking] tracking/data/instances.py`** — replace the `binary_to_instances`
function (lines 46-66) with:

```python
def label_instances(pred: np.ndarray, clicks_zyx: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """pred is bool/0-1, same grid as clicks. Return int32 mask, voxel = lesion_id.

    Every predicted connected component becomes an instance. A CC a click lands on exactly
    keeps that click's id. A CC with no click on it (click missed by a few voxels of
    registration error, or the model split one historical lesion into pieces) keeps a fresh
    id instead of being silently dropped -- unlike the old binary_to_instances, which threw
    away any predicted lesion a click didn't land on exactly. See
    docs/dev-notes/segtrack_correctness_and_registration_plan.md Phase 1.
    """
    lab = cc3d.connected_components((pred > 0).astype(np.uint8), connectivity=18)
    n = int(lab.max())
    if n == 0:
        return lab.astype(np.int32)
    lut = np.zeros(n + 1, dtype=np.int32)
    claimed: dict[int, int] = {}
    conflicts: list[tuple[int, int, int]] = []
    for lid, (z, y, x) in clicks_zyx.items():
        z = min(max(int(z), 0), pred.shape[0] - 1)
        y = min(max(int(y), 0), pred.shape[1] - 1)
        x = min(max(int(x), 0), pred.shape[2] - 1)
        cc = int(lab[z, y, x])
        if cc == 0:
            continue
        if cc in claimed and claimed[cc] != lid:
            conflicts.append((lid, claimed[cc], cc))
            continue
        claimed[cc] = lid
        lut[cc] = lid
    if conflicts:
        cprint(f"[yellow]click CC conflicts (later click skipped): {conflicts}[/yellow]")
    used = set(clicks_zyx.keys()) | set(claimed.values())
    next_id = (max(used) + 1) if used else 1
    for cc in range(1, n + 1):
        if lut[cc] == 0:
            lut[cc] = next_id
            next_id += 1
    return lut[lab]
```

Keep `load_clicks` and `instances_from_nifti` unchanged, except `instances_from_nifti`
(line 73) must call `label_instances` instead of `binary_to_instances`. Do not add a
`SNAP_RADIUS`/nearest-voxel-search mechanism — that was an earlier, inferior approach
(a fixed voxel tolerance) superseded by this "never drop" design; do not reintroduce it.

**`[nanoUNet] nanounet/infer/segtrack.py::run_case`** — both call sites of
`binary_to_instances` (lines 97 and 115-116) become `label_instances` (update the import at
line 66: `from tracking.data.instances import label_instances, load_clicks`). No other
change in this phase — do not touch the `--bl-mask-dir` branch (`bl_zyx` from
`load_instance_zyx`) at all.

### 2.3 Verification (do this before moving on)

Reuse the existing on-disk EMA predictions and click JSONs — no model inference needed:

```python
import json
import numpy as np
import SimpleITK as sitk
from tracking.data.instances import label_instances, load_clicks

for stem in ["307fd7f231_00", "38b18881fc_00", "bf97f24695_00", "0f49c89d1e_00"]:
    ema = sitk.GetArrayFromImage(sitk.ReadImage(f"/nnunet_data/Longitudinal-CT/results/preds_ema_finetune/{stem}.nii.gz"))
    clicks = load_clicks(f"/nnunet_data/Longitudinal-CT/inputsTrFU/{stem}.json")
    inst = label_instances(ema > 0, clicks)
    n_cc_total = int(np.unique(inst).size) - 1  # exclude background
    print(stem, "clicks:", len(clicks), "instances kept:", n_cc_total)
```

Expected: **every one of these 4 cases keeps at least as many instances as there are
foreground connected components in the EMA prediction** (i.e. `n_cc_total` should equal the
number of CCs `cc3d.connected_components(ema>0, connectivity=18)` finds, not the number of
clicks that happen to land exactly on one). Confirm this — under the old
`binary_to_instances`, `307fd7f231_00` produced **0** surviving instances from 2 clicks;
under `label_instances` it must produce instances for every real CC in the prediction
(2 CCs in this case, both kept, regardless of click accuracy).

Then run the full pipeline end to end on these 4 cases and confirm `fu.mha` is no longer
empty where `pred_fu.mha` (pass `--keep-pred`) has real foreground.

## 3. Phase 2 — EMA default

`[nanoUNet] nanounet/cli/segtrack.py` line 53: `ap.add_argument("--ema", action="store_true")`
— defaults to `False`. This is documented (`docs/steps/track.md` line 68: "`--ema` | flag |
off"), not a bug in isolation, but the project's own reference predictions
(`/nnunet_data/Longitudinal-CT/results/preds_ema_finetune`) were generated *with* `--ema`,
and EMA weights are the intended production checkpoint throughout this project (see
`DEPLOYED_CKPT` naming convention for the tracker). Default it on for consistency:

**`[nanoUNet] nanounet/cli/segtrack.py`** line 53, change:
```python
ap.add_argument("--ema", action="store_true")
```
to:
```python
ap.add_argument("--no-ema", dest="ema", action="store_false", default=True)
```
This flips the default to `True` while keeping an escape hatch (`--no-ema`) for anyone who
explicitly wants the raw checkpoint. Update the config-table row that echoes this value —
`[nanoUNet] nanounet/cli/segtrack.py` line ~100, the `("track-ema", "on", "default")` row is
for the *matcher's* EMA (always on, unrelated) — check there is a separate row surfacing the
**segmentation** UNet's `--ema`/`args.ema` value; if there isn't one today, add one so the
resolved config table doesn't silently hide this default.

**`[nanoUNet] docs/steps/track.md`** line 68 argument table row: change
`| \`--ema\` | flag | off | Seg UNet EMA. Matcher EMA is always on |` to
`| \`--ema\` | flag | **on** | Seg UNet EMA. \`--no-ema\` for the raw checkpoint. Matcher EMA is always on |`.

Do **not** change `nanounet_predict`'s own `--ema` default (`[nanoUNet] nanounet/cli/predict.py`
line 34) — out of scope for this plan; only `nanounet_segtrack`'s default was established
this session as causing a real discrepancy against the project's reference predictions.

### Verification
`nanounet_segtrack --help` (or the printed config table on a real run) must show `--ema`
defaulting to on; `--no-ema` must still work and load the raw checkpoint (confirm via the
existing `load_net_from_ckpt(..., ema=args.ema)` call unchanged at
`[nanoUNet] nanounet/cli/segtrack.py` line 128).

## 4. Phase 3 — uniGradICON live registration module

### 4.1 What this replaces

Today, `[nanoUNet] nanounet/infer/segtrack.py::run_case` (lines 142-151) does this:

```python
prop = case.meta_csv if case.meta_csv is not None else case.fu_clicks
...
if not drop_dp:
    mx = int(bl_zyx.max())
    bl_ids = np.flatnonzero(np.bincount(bl_zyx.ravel(), minlength=mx + 1))[1:].tolist() if mx > 0 else []
    got, _ = load_propagated(prop, bl_ids, img_id=img_id)
    drop = sorted(set(bl_ids) - set(got))
    if drop:
        cprint(f"[dim]drop {case.stem}  BL ids {drop} (not in this FU volume)[/dim]")
```

Two problems: (1) when `case.meta_csv is None`, it silently reuses `case.fu_clicks` — the
plain nanoUNet interactive-segmentation click file — *as if* it were a real "BL centroid
warped into FU space" file. This only happens to work on this project's own dataset because
(per Phase 1's investigation) `inputsTrFU/*.json` *was itself* built from `cog_propagated`.
For any other dataset, a genuine FU click file has no such relationship to BL and this
silently produces nonsense BL positions. (2) any BL id not covered (`drop`) is just logged
and discarded — that lesion is never tracked, period, even though real registration could
recover a usable position for it.

This phase adds a live-registration fallback so a BL id with no CSV coverage — whether
because there's no CSV at all, or because it's a fresh id Phase 1 invented for an extra
connected component with no historical record — gets a real, computed FU-space position
instead of being dropped or guessed at.

### 4.2 New module: `[lesion-tracking] tracking/data/unigradicon.py` (new file)

This must be a **self-contained, lazily-imported module** — importing `tracking.infer` or
`tracking.data.masks` must not require `unigradicon`/`itk` to be installed. Only import
`itk`, `icon_registration`, `unigradicon` inside function bodies (never at module top),
so a segtrack run that's fully covered by a CSV never touches this dependency at all.

**Dependency**: add to `[lesion-tracking] pyproject.toml`:
```toml
[project.optional-dependencies]
unigradicon = ["unigradicon"]
```
(`unigradicon`'s own `pyproject.toml`/`requirements.txt` pulls in `itk`, `icon_registration`,
`torch`, `footsteps` transitively — do not list those separately.) Install with
`pip install -e /lesion-tracking[unigradicon]`.

**uniGradICON API, as actually shipped** (verified by reading the real source at
`github.com/uncbiag/uniGradICON` `src/unigradicon/__init__.py` and
`github.com/uncbiag/ICON` `src/icon_registration/itk_wrapper.py` — do not guess at this API
from memory, it does not match the CLI's `--fixed`/`--moving` naming directly):

- `unigradicon.get_unigradicon(weights_location=None)` loads the pretrained network. If
  `weights_location` doesn't exist, it **auto-downloads** the weights — but its own
  `os.makedirs(...)` call targets a hardcoded relative path (`"network_weights/unigradicon1.0/"`),
  **not** whatever `weights_location` you pass. If you pass a custom absolute path whose
  parent directory doesn't exist, the download will fail. **You must `mkdir -p` the parent
  of your chosen `weights_location` yourself before calling this.**
- `unigradicon.preprocess(itk_image, modality="ct")` casts to float, clamps to `[-1000, 1000]`
  HU, then shift/scales to `[0, 1]`. Use this exactly, don't reimplement it.
- `icon_registration.itk_wrapper.register_pair(model, image_A, image_B, finetune_steps=None)`
  returns `(phi_AB, phi_BA)`, both `itk.CompositeTransform[itk.D, 3]`. Per
  `create_itk_transform`'s own comment (`# warp(image_A, phi_AB_itk) is close to image_B`)
  and confirmed against the CLI's own `--warped_moving_out` resample call:
  **`phi_AB.TransformPoint(point in image_B's physical space)` returns the corresponding
  point in `image_A`'s physical space.** (`phi_BA` is the reverse: image_A space →
  image_B space.)
  **To get BL→FU: call `register_pair(model, image_A=fu_img, image_B=bl_img, ...)`, then
  use `phi_AB.TransformPoint(bl_physical_point)` → FU physical point.** Do not swap this —
  it is the single most likely place to introduce a silent, hard-to-detect bug (wrong
  points, not a crash). The Phase 3.4 validation step exists specifically to catch this if
  the derivation above is somehow wrong.
- `finetune_steps=None` disables "instance optimization" (IO), an iterative per-pair
  refinement (the CLI's `--io_iterations`, default `50` there). IO materially slows
  registration (tens of gradient steps per pair). **Default to `finetune_steps=None`** (base
  network only, one forward pass, a few seconds on GPU) to keep this fast — this is a
  correctness/speed tradeoff, not free; document it, don't hide it.

**Coarse pre-alignment — required, not optional.** uniGradICON resamples each image to a
fixed 175×175×175 grid **independently**, based only on that image's own header (spacing/
origin/direction) — verified in `register_pair`'s source (`F.interpolate(A_trch, size=shape[2:], ...)`
on the raw pixel array; the physical geometry only re-enters via
`resampling_transform(image_A, shape)`/`resampling_transform(image_B, shape)`, each built
independently per image). If BL and FU cover very different physical extents (different
scan range, different couch position — normal for real longitudinal CT), the network sees
unrelated anatomy in the two 175³ grids and registration degrades badly. Confirmed as a
correctness risk by the user before this plan was written. Fix: a **cheap, metadata-only
translation** that re-origins FU so its body centroid coincides with BL's, before handing
both to uniGradICON. This does not correct rotation, only gross translation offset — the
dominant failure mode for two full-body/torso CT scans acquired at different times. If this
proves insufficient in practice (see 4.4), the next escalation is a real affine
pre-registration; do not build that preemptively, it is out of scope for this plan.

Write the full file exactly as follows:

```python
"""Live BL->FU point propagation via uniGradICON, for BL lesions with no CSV coverage.

Only imported when a BL lesion has no --meta/--meta-dir coverage (no CSV at all, or an id
Phase 1's label_instances invented for an extra connected component). A case fully covered
by a propagated CSV never touches this module or its heavy dependencies (itk, torch,
icon_registration). See docs/dev-notes/segtrack_correctness_and_registration_plan.md Phase 3
in nanoUNet for the full derivation and required validation before this is trusted.
"""

from __future__ import annotations

from pathlib import Path

import cc3d
import numpy as np

BODY_HU_THRESHOLD = -500.0  # separates patient body from air/couch; matches unigradicon's own [-1000,1000] CT clamp range

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_model(weights_path: Path, device: str):
    key = (str(weights_path), device)
    if key not in _MODEL_CACHE:
        import torch
        from unigradicon import get_unigradicon

        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model = get_unigradicon(weights_location=str(weights_path))
        model.to(torch.device(device)).eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _body_centroid_phys(img) -> np.ndarray:
    """Physical-space (x, y, z) centroid of the image's largest body/torso component."""
    import itk

    arr = itk.array_from_image(img)  # (z, y, x) numpy view -- standard ITK<->numpy convention
    body = (arr > BODY_HU_THRESHOLD).astype(np.uint8)
    lab = cc3d.connected_components(body, connectivity=18)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    assert largest > 0, "no foreground above BODY_HU_THRESHOLD -- unexpected for a real CT"
    idx_zyx = np.argwhere(lab == largest).mean(axis=0)
    idx_xyz = itk.ContinuousIndex[itk.D, 3]([float(idx_zyx[2]), float(idx_zyx[1]), float(idx_zyx[0])])
    return np.array(img.TransformContinuousIndexToPhysicalPoint(idx_xyz))


def _coarse_align(bl_img, fu_img):
    """Shift fu_img's origin so its body centroid matches bl_img's, in physical space.
    Metadata-only: no resampling, no pixel data touched. Mutates and returns fu_img."""
    bl_dir = np.array(bl_img.GetDirection())
    fu_dir = np.array(fu_img.GetDirection())
    assert np.allclose(bl_dir, fu_dir, atol=1e-3), (
        f"BL/FU direction cosines differ (BL={bl_dir.tolist()}, FU={fu_dir.tolist()}); "
        f"coarse pre-alignment assumes matched patient orientation between the two scans."
    )
    delta = _body_centroid_phys(bl_img) - _body_centroid_phys(fu_img)
    fu_img.SetOrigin(tuple(np.array(fu_img.GetOrigin()) + delta))
    return fu_img


def propagate_points(
    bl_ct_path: Path,
    fu_ct_path: Path,
    points_bl_zyx: dict[int, np.ndarray],
    *,
    weights_path: Path,
    device: str,
    io_iterations: int | None = None,
) -> dict[int, np.ndarray]:
    """BL voxel-index (z, y, x) points -> FU voxel-index (z, y, x) points via uniGradICON.

    io_iterations=None runs the base network only (a few seconds on GPU, no per-pair
    refinement). Pass an int (uniGradICON's own CLI default is 50) for slower, more accurate
    per-case instance optimization if base-network accuracy proves insufficient.
    """
    import itk
    from icon_registration.itk_wrapper import register_pair
    from unigradicon import preprocess

    if not points_bl_zyx:
        return {}

    model = _get_model(Path(weights_path), device)
    bl_img = itk.imread(str(bl_ct_path))
    fu_img = _coarse_align(bl_img, itk.imread(str(fu_ct_path)))

    # image_A=FU, image_B=BL => phi_AB.TransformPoint(point in BL space) = point in FU space.
    # See plan Phase 3.2 for the full derivation. Do not swap image_A/image_B.
    phi_AB, _phi_BA = register_pair(
        model,
        preprocess(fu_img, modality="ct"),
        preprocess(bl_img, modality="ct"),
        finetune_steps=io_iterations,
    )

    out: dict[int, np.ndarray] = {}
    for lid, zyx in points_bl_zyx.items():
        idx_xyz = itk.ContinuousIndex[itk.D, 3]([float(zyx[2]), float(zyx[1]), float(zyx[0])])
        p_bl_phys = bl_img.TransformContinuousIndexToPhysicalPoint(idx_xyz)
        p_fu_phys = phi_AB.TransformPoint(p_bl_phys)
        idx_fu = fu_img.TransformPhysicalPointToContinuousIndex(p_fu_phys)
        out[lid] = np.array([idx_fu[2], idx_fu[1], idx_fu[0]], dtype=np.float64)
    return out
```

Known API-risk to check while implementing (not certainties — verify against the installed
`itk` package version, this is exactly what Phase 3.4's validation step is for):
- `itk.array_from_image`, `itk.imread`, `.GetDirection()`, `.GetOrigin()`, `.SetOrigin()`,
  `.TransformContinuousIndexToPhysicalPoint()`, `.TransformPhysicalPointToContinuousIndex()`,
  `itk.ContinuousIndex[itk.D, 3]([x, y, z])` are all standard, long-stable ITK Python API —
  high confidence these work as written.
- `icon_registration.config.device` may auto-select CUDA/CPU independently of the `device`
  argument passed here (`register_pair` internally calls `model.to(config.device)`, which
  could override the placement done in `_get_model`). If Phase 3.4 validation shows
  computation running on the wrong device, set `icon_registration.config.device` explicitly
  before calling `register_pair` — check `icon_registration/config.py`'s actual contents
  first, don't guess at the fix.

### 4.3 Weight location default

**`[lesion-tracking] tracking/common.py`** — add near `DEPLOYED_CKPT` (line 25):
```python
DEFAULT_UNIGRADICON_WEIGHTS_NAME = "unigradicon1.0/Step_2_final.trch"
```
**`[nanoUNet] nanounet/infer/segtrack.py`** — add near `DEFAULT_MODEL` (line 22-25), reusing
the exact existing `resolve_ckpt_path` pattern from
`[nanoUNet] nanounet/infer/segtrack_case.py`:
```python
from nanounet.common import results_dir
DEFAULT_UNIGRADICON_WEIGHTS = Path(results_dir()) / "unigradicon" / "unigradicon1.0" / "Step_2_final.trch"
```
Resolve it in `[nanoUNet] nanounet/cli/segtrack.py::main()` the same way `model_dir`/
`track_ckpt` already are (`resolve_ckpt_path(args.unigradicon_weights, "NANOUNET_UNIGRADICON_WEIGHTS", DEFAULT_UNIGRADICON_WEIGHTS)`),
and add a config-table row for it, matching the existing `model-dir`/`track-ckpt` rows
exactly in style.

### 4.4 Validation gate — required before Phase 4 wiring

Nobody has run this exact code path yet (`unigradicon` is not installed in this
environment). Do not wire this into `run_case` until this validation passes. Write a
throwaway script (delete it after, per this project's testing convention — no permanent
`tests/` folder), run it, and confirm the numbers before proceeding:

```python
# scratch validation -- not part of either package, delete after running
from pathlib import Path
import numpy as np
from tracking.data.unigradicon import propagate_points

# use a real case with a known, already-computed answer: 307fd7f231_00, lesion_id=1
# meta/307fd7f231.csv, img_id_fu=0: cog_bl="199.601855155191 235.505440599358 122.067160185516"
#                                    cog_fu="210.814898057473 304.457296516295 126.892438593675"
#                                    cog_propagated="207.95331503893644 286.0173397756782 119.03847670485004"
bl_zyx = np.array([199.601855155191, 235.505440599358, 122.067160185516])

out = propagate_points(
    Path("/nnunet_data/Longitudinal-CT/inputsTrBL/307fd7f231_00.nii.gz"),
    Path("/nnunet_data/Longitudinal-CT/inputsTrFU/307fd7f231_00.nii.gz"),
    {1: bl_zyx},
    weights_path=Path("/nnunet_data/NanoUNet_results/unigradicon/unigradicon1.0/Step_2_final.trch"),
    device="cuda",
)
print("computed:", out[1])
print("existing cog_propagated (z,y,x):", [119.03847670485004, 286.0173397756782, 207.95331503893644])
print("true cog_fu             (z,y,x):", [126.892438593675, 304.457296516295, 210.814898057473])
```

**Pass criterion**: `out[1]` should land closer to the true `cog_fu` than a wrong-direction
bug would (a swapped `image_A`/`image_B` typically produces a point wildly outside the body,
or very close to `cog_bl` unchanged — either is an obvious, large-magnitude failure, not a
subtle few-mm difference). It does not need to exactly match the existing `cog_propagated`
(that came from a different, offline registration run, possibly a different backend) or
`cog_fu` (that's ground truth, registration is never exact) — but it must be in the same
general neighborhood (tens of mm, not hundreds), and the coarse pre-alignment assert must
not fire. If the result is nonsensical, the most likely bugs, in order of likelihood: (1)
`image_A`/`image_B` swapped in `register_pair` (try swapping and re-check against `cog_fu`),
(2) the `(z,y,x)`↔`(x,y,z)` reversal missing or doubled somewhere, (3) the coarse-alignment
delta sign flipped (`c_bl - c_fu` vs `c_fu - c_bl`). Re-derive from Section 1's conventions,
don't guess-and-check blindly.

Also time this call — confirm it completes in single-digit seconds on this GPU with
`io_iterations=None`. If it's much slower, that's a signal `icon_registration.config.device`
is not actually using the GPU (see the note in 4.2).

## 5. Phase 4 — wire the fallback into `run_case` / `track()` / `build_mask_graph`

Only start this after Phase 3.4 passes.

### 5.1 `[lesion-tracking] tracking/data/masks.py::build_mask_graph`

Add a parameter and relax the hard-fail-on-missing-CSV behavior. Current (lines 56-70):
```python
def build_mask_graph(
    ct_bl: np.ndarray, aff_bl: np.ndarray, sp_bl: np.ndarray, mk_bl: np.ndarray,
    ct_fu: np.ndarray, aff_fu: np.ndarray, sp_fu: np.ndarray, mk_fu: np.ndarray,
    propagated_csv: Path | None, cfg: GraphConfig, default_lesion_type: str | None = None,
    types_csv: Path | None = None, img_id: int | None = None,
) -> HeteroData | None:
    all_bl, fu_ids = _labels(mk_bl), _labels(mk_fu)
    types: dict[int, str] = {}
    if cfg.drop_dp:
        bl_ids = all_bl
        prop: dict = {}
    else:
        if propagated_csv is None:
            raise FileNotFoundError(
                "No propagated file for a geo matcher.\n"
                "Expected meta CSV, slim CSV, or FU-frame JSON.\n"
                "Fix: pass --propagated /nnunet_data/Longitudinal-CT/meta/<pid>.csv"
            )
        prop, types = load_propagated(propagated_csv, all_bl, img_id=img_id)
        bl_ids = [i for i in all_bl if i in prop]
```

Change to:
```python
def build_mask_graph(
    ct_bl: np.ndarray, aff_bl: np.ndarray, sp_bl: np.ndarray, mk_bl: np.ndarray,
    ct_fu: np.ndarray, aff_fu: np.ndarray, sp_fu: np.ndarray, mk_fu: np.ndarray,
    propagated_csv: Path | None, cfg: GraphConfig, default_lesion_type: str | None = None,
    types_csv: Path | None = None, img_id: int | None = None,
    extra_propagated: dict[int, np.ndarray] | None = None,
) -> HeteroData | None:
    all_bl, fu_ids = _labels(mk_bl), _labels(mk_fu)
    types: dict[int, str] = {}
    if cfg.drop_dp:
        bl_ids = all_bl
        prop: dict = {}
    else:
        prop = {}
        if propagated_csv is not None:
            prop, types = load_propagated(propagated_csv, all_bl, img_id=img_id)
        if extra_propagated:
            prop.update(extra_propagated)
        bl_ids = [i for i in all_bl if i in prop]
```
The existing `if not bl_ids or not fu_ids: return None` two lines below (unchanged) already
handles "still nothing trackable" gracefully — do not add a new error path there.

### 5.2 `[lesion-tracking] tracking/infer.py::track`

Add the same parameter and thread it through (current signature at line 93-113, current
`build_mask_graph` call at line 154-158):
```python
def track(
    bl_img: Path, bl_mask: Path, fu_img: Path, fu_mask: Path,
    propagated: Path | None, ckpt: Path, *,
    decode: str, device: str = "cuda", default_lesion_type: str | None = "unclear",
    k_intra: int = 8, thresh: float = 0.5, sinkhorn_iters: int = 20,
    sinkhorn_tau: float = DEPLOYED_DUST_TAU, use_ema: bool = True,
    matcher: MatcherModule | None = None, types_csv: Path | None = None,
    volumes: tuple | None = None, img_id: int | None = None,
    extra_propagated: dict[int, np.ndarray] | None = None,
) -> TrackResult:
```
and pass `extra_propagated=extra_propagated` into the `build_mask_graph(...)` call.

Also relax the existing hard guard in the `else` branch (used when `volumes is not None`,
which is always true for the `nanounet_segtrack` in-process path) at lines 144-150:
```python
        if not gcfg.drop_dp and (propagated is None or not Path(propagated).is_file()):
            raise FileNotFoundError(...)
```
change to:
```python
        if not gcfg.drop_dp and propagated is None and extra_propagated is None:
            raise FileNotFoundError(
                "No propagated coordinates for a geo matcher.\n"
                "Expected --meta/--meta-dir, or the uniGradICON fallback enabled.\n"
                "Fix: pass --meta, or drop --drop-uncovered so live registration can run\n"
                "(see docs/dev-notes/segtrack_correctness_and_registration_plan.md)"
            )
```
Note this still requires `Path(propagated).is_file()` checking to be preserved for the case
where `propagated is not None` but points at a missing file — keep that part of the original
condition, only change the "is `None` OK now" logic. Leave the `if volumes is None:` branch
(lines 131-140, the plain-CLI `lesion_track` path with no in-memory volumes) untouched —
out of scope for this plan (see Section 7, non-goals).

### 5.3 `[nanoUNet] nanounet/infer/segtrack.py::run_case`

This is the main wiring point. Current code (lines 141-159):
```python
    drop_dp = bool(getattr(matcher.hparams, "drop_dp", False))
    prop = case.meta_csv if case.meta_csv is not None else case.fu_clicks
    _, region = stem_pid_region(case.stem)
    img_id = region if case.meta_csv is not None else None
    if not drop_dp:
        mx = int(bl_zyx.max())
        bl_ids = np.flatnonzero(np.bincount(bl_zyx.ravel(), minlength=mx + 1))[1:].tolist() if mx > 0 else []
        got, _ = load_propagated(prop, bl_ids, img_id=img_id)
        drop = sorted(set(bl_ids) - set(got))
        if drop:
            cprint(f"[dim]drop {case.stem}  BL ids {drop} (not in this FU volume)[/dim]")
    r = track(
        case.bl_img, case.bl_img, case.fu_img, case.fu_img,
        None if drop_dp else prop, track_ckpt,
        decode=decode, device=device, matcher=matcher, thresh=thresh,
        sinkhorn_tau=DEPLOYED_DUST_TAU, use_ema=True,
        types_csv=case.types_csv, img_id=img_id,
        volumes=(ct_bl, aff_bl, sp_bl, mk_bl, ct_fu, aff_fu, sp_fu, mk_fu),
    )
```
Replace with:
```python
    drop_dp = bool(getattr(matcher.hparams, "drop_dp", False))
    prop = case.meta_csv  # may be None now -- means "no CSV, rely on live registration"
    _, region = stem_pid_region(case.stem)
    img_id = region if case.meta_csv is not None else None
    extra_propagated = None
    if not drop_dp:
        mx = int(bl_zyx.max())
        bl_ids = np.flatnonzero(np.bincount(bl_zyx.ravel(), minlength=mx + 1))[1:].tolist() if mx > 0 else []
        got: dict = {}
        if prop is not None:
            got, _ = load_propagated(prop, bl_ids, img_id=img_id)
        missing = sorted(set(bl_ids) - set(got))
        if missing and not no_live_registration:
            from tracking.data.appearance import centroids
            from tracking.data.unigradicon import propagate_points
            extra_propagated = propagate_points(
                case.bl_img, case.fu_img, centroids(bl_zyx, missing),
                weights_path=unigradicon_weights, device=device,
            )
            cprint(f"[dim]{case.stem}  live-registered {len(missing)} BL id(s) via uniGradICON (no CSV coverage)[/dim]")
        elif missing:
            cprint(f"[dim]drop {case.stem}  BL ids {missing} (no propagated coverage, --drop-uncovered set)[/dim]")
    r = track(
        case.bl_img, case.bl_img, case.fu_img, case.fu_img,
        None if drop_dp else prop, track_ckpt,
        decode=decode, device=device, matcher=matcher, thresh=thresh,
        sinkhorn_tau=DEPLOYED_DUST_TAU, use_ema=True,
        types_csv=case.types_csv, img_id=img_id,
        volumes=(ct_bl, aff_bl, sp_bl, mk_bl, ct_fu, aff_fu, sp_fu, mk_fu),
        extra_propagated=extra_propagated,
    )
```
`run_case`'s own signature needs two new keyword parameters, `no_live_registration: bool`
and `unigradicon_weights: Path`, added alongside the existing `seg_kw`/`on_step` parameters
(around line 64); thread them in from the caller (5.4 below). `bl_zyx` here is the same
array already computed earlier in `run_case` (either from `label_instances` or, in the
`--bl-mask-dir` branch, from `load_instance_zyx`) — this fallback applies uniformly to both,
by design (Section 2.1 already established BL identity needs this in either case).

### 5.4 `[nanoUNet] nanounet/cli/segtrack.py` — new CLI flags

Add to `_mode()` (near the existing `--ema`/`--batch-size` flags, line ~52-57):
```python
ap.add_argument("--unigradicon-weights")
ap.add_argument("--drop-uncovered", action="store_true",
                 help="Skip BL lesions with no propagated-coordinate coverage instead of live-registering them")
```
In `main()`, resolve the weights path the same way `model_dir`/`track_ckpt` already are
(near line 69-70):
```python
unigradicon_weights, uw_src = resolve_ckpt_path(
    args.unigradicon_weights, "NANOUNET_UNIGRADICON_WEIGHTS", DEFAULT_UNIGRADICON_WEIGHTS,
)
```
Add a config-table row (near line 95-110) matching the existing style:
```python
("unigradicon-weights", unigradicon_weights, uw_src),
```
Pass `no_live_registration=args.drop_uncovered, unigradicon_weights=unigradicon_weights`
into the `run_case(...)` call inside the progress loop (around line 152-157).

### 5.5 Docs

Update `[nanoUNet] docs/steps/track.md`'s argument table (near line 68) with rows for
`--unigradicon-weights` and `--drop-uncovered`, and add a short paragraph explaining the
live-registration fallback and when it triggers (no `--meta`/`--meta-dir` at all, or a BL
lesion id the given CSV doesn't cover). Follow the existing doc format exactly (D3 in this
project's nanochat-style skill: argument table with `Argument | Type | Default | Description`
columns).

## 6. End-to-end verification protocol

Run in this order; each step gates the next.

1. **Phase 1 alone**: run the Section 2.3 snippet. Confirm no lesion with real predicted
   foreground is dropped purely for a click miss.
2. **Phase 1 full pipeline**: rerun `nanounet_segtrack` on the same 4 flagged cases
   (`307fd7f231_00`, `38b18881fc_00`, `bf97f24695_00`, `0f49c89d1e_00`) with `--overwrite
   --keep-pred --ema`, confirm `fu.mha` is non-empty wherever `pred_fu.mha` has real
   foreground.
3. **Phase 2**: confirm the printed config table shows `--ema` on by default; rerun without
   `--ema`/with `--no-ema` and confirm it still works.
4. **Phase 3 validation gate**: run Section 4.4's script standalone. Do not proceed past
   this step until it passes.
5. **Phase 4, CSV-covered case (regression check)**: rerun a case that has full `--meta`
   coverage today (any case already in `meta/*.csv` with no gaps). Confirm `uniGradICON` is
   **never imported** for this case (add a temporary print/breakpoint in `_get_model` if
   needed to confirm it's not called) and timing is unchanged from before this plan.
6. **Phase 4, no-CSV case**: run `nanounet_segtrack` on a case **without** `--meta`/
   `--meta-dir` at all. Confirm it completes (does not crash with the old
   `FileNotFoundError`), confirm `matches.csv` has real pairs, and spot-check that the
   painted `fu.mha` ids look reasonable against `bl.mha`.
7. **Full 63-case test-set rerun**: once 1-6 all pass, rerun the full `followup` scoring
   command from `docs/steps/track.md` with `--overwrite`, and recompute Dice using whatever
   scoring approach is current at that time (the original `scripts/score_segtrack_fu.py`
   used earlier this session was a standalone throwaway script not committed to the repo —
   check whether it still exists before rewriting it). Expect ID-matched median Dice to move
   substantially off `0.000` — Phase 1 alone should recover most of the previously-dropped
   lesions.

## 7. Explicitly out of scope for this plan (do not build these)

- Full affine/rigid pre-registration beyond the single translation in `_coarse_align`. Only
  build this if 4.4's validation (or later real-world use) shows the translation-only
  approach is insufficient.
- Wiring the uniGradICON fallback into the standalone `lesion_track` CLI
  (`tracking/cli/track.py`, the `volumes is None` path in `track()`). This plan only wires
  it into `nanounet_segtrack`'s in-process fast path.
- Any change to `nanounet_predict`'s `--ema` default.
- Any change to how BL identity is displayed/renumbered in the final `bl.mha`/`fu.mha` for
  a fresh synthetic id from Phase 1 (e.g. wanting "nicer" numbers) — the only requirement
  per `docs/reference/track_ids.md` is "same integer = same lesion," which fresh ids already
  satisfy.
- Multi-modality (MRI) support in the registration module — this project is CT-only
  (`preprocess(..., modality="ct")` is hardcoded intentionally).
