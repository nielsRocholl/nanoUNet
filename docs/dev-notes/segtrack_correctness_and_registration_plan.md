# segtrack correctness + live registration plan

Status: **plan only. Nothing in this doc is implemented yet.** Written for a coding agent
with no session memory. Read the whole document before touching code.

Verified against the live trees on 2026-08-29 (not against the previous draft of this
file). Line numbers below are from that snapshot. If a cited line has drifted, search
the symbol, do not trust the number.

**Repos** — two git repos, both required:

- `/nanoUNet` — `nanounet_segtrack`. Paths prefixed `[nanoUNet]`.
- `/lesion-tracking` — GNN matcher, imported as `tracking.*` (`pip install -e /lesion-tracking`).

nanoUNet already depends on `unigradicon>=1.0.4` (`[nanoUNet] pyproject.toml`). lesion-tracking
does **not** get a new optional extra.

---

## 0. What this plan is for

`nanounet_segtrack` predicts a binary lesion mask per scan, turns it into per-lesion
instance ids, and calls the GNN matcher to link baseline (BL) lesions to follow-up (FU).

A held-out run is on disk at `$NANOUNET_RESULTS/segtrack/followup/`
(`NANOUNET_RESULTS=/nanounet_data/NanoUNet_results`). Scoring JSON:

`/nnunet_data/NanoUNet_results/segtrack/followup/dice_vs_targetsTrFU.json`

| agg field | value |
|-----------|-------|
| `n_cases` | 63 |
| `n_gt_lesions` | 549 |
| `n_pred_lesions` | 284 |
| `n_detected` | 266 |
| `median_comp_lesion` | **0.0** |
| `mean_detected` | 0.731 |

The UNet is fine on lesions that survive onto `fu.mha` (`mean_detected` 0.73; case-level
detected medians in the 0.79 range). The published ID-matched median is zero because
**correctly segmented FU blobs are deleted before the matcher runs**.

Separately, the matcher (deployed `v7_complete/last.ckpt`: `drop_dp=false`, `intra=complete`,
`type_mask=false`) needs a BL lesion centroid in the **FU voxel grid**. Today that comes from
`meta/{pid}.csv` column `cog_propagated`, or, when `meta_csv is None`, from `case.fu_clicks`.
Folder mode **skips** a case with no `{pid}.csv` if a meta dir is inferred. A user who only
has two CTs and click JSONs cannot run the tool.

Three fixes, one plan, overlapping code:

| Phase | Fix |
|-------|------|
| 1 | Stop dropping predicted connected components that a click missed |
| 2 | Default the **seg** UNet to EMA (`--no-ema` to opt out) |
| 3+4 | Compute missing BL→FU centroids with the **existing** uniGradICON wrapper; stop pretending FU click JSON is a registration product |

Phase 1 is the metric bug (the `--bl-mask-dir` eval path). Phases 3–4 are the product path
(no CSV, or a BL id Phase 1 invented). Do not skip Phase 1 to start registration.

---

## 1. Two coordinate frames — get this wrong and nothing crashes

The previous draft of this plan claimed every bare triple is `(z, y, x)`. That is false.
There are two frames. Mixing them is a silent, millimetre-scale bug.

### Frame A — SimpleITK / instance masks: `(z, y, x)`

`SimpleITKIO().read_images` / `read_seg` and `sitk.GetArrayFromImage` yield `(z, y, x)`.

- `[lesion-tracking] tracking/data/instances.py::load_clicks` — JSON `"point"` is `[x, y, z]`;
  returns `(int(round(p[2])), int(round(p[1])), int(round(p[0])))` for indexing a SITK array.
- `[nanoUNet] nanounet/infer/segtrack.py` `bl_zyx` / `fu_zyx` / `pred_*` are this frame.
- `binary_to_instances` / `label_instances` index `lab[z, y, x]`.

### Frame B — matcher / CSV / JSON-as-propagated / ITK Index: `(x, y, z)`

`[nanoUNet] nanounet/infer/segtrack.py` transposes before `track()`:

```137:138:[nanoUNet] nanounet/infer/segtrack.py
    mk_bl = np.ascontiguousarray(bl_zyx.transpose(2, 1, 0))
    mk_fu = np.ascontiguousarray(fu_zyx.transpose(2, 1, 0))
```

`[lesion-tracking] tracking/data/graph.py::vol_from_zyx` documents this as
"XYZ float32 + RAS affine + spacing. Same layout as nibabel get_fdata."

ITK `TransformIndexToPhysicalPoint` / `TransformContinuousIndexToPhysicalPoint` take `(x, y, z)`.

On-disk `meta/*.csv` `cog_*` strings are **the same numbers, same order** as click JSON
`[x, y, z]`. Empirically, case `307fd7f231_00` lesion 1:

| source | triple |
|--------|--------|
| BL JSON `"point"` | `[199.60, 235.50, 122.07]` |
| `cog_bl` | `199.60 235.50 122.07` |
| SITK mask centroid `(z, y, x)` | `[121.57, 235.00, 199.10]` |
| that centroid reversed to `(x, y, z)` | `[199.10, 235.00, 121.57]` (+0.5 = CSV) |
| `sitk_array[z=122, y=236, x=200]` (CSV as xyz) | **label 1** |
| `sitk_array[z=200, y=236, x=122]` (CSV as zyx) | **0** |
| nibabel `data[199, 235, 122]` | **1** |

`parse_zyx` (`[lesion-tracking] tracking/data/meta.py:24-33`) splits `"a b c"` and returns
`(a, b, c)` with an error message that says "expected z y x". The **values** are Frame B
`(x, y, z)`. The GNN was trained on that. **Do not swap CSV triples to "fix" the name.**
Do not swap live-registration output either.

`load_propagated` JSON branch (`_from_json`) stores `[p[0], p[1], p[2]]` with **no**
`load_clicks` swap. That is correct for Frame B. `load_clicks` swap is correct for Frame A.
Both are intentional.

### ITK physical `(x, y, z)`

`itk.Image.GetSpacing/GetOrigin/GetDirection` and point/index APIs are `(x, y, z)`.
SimpleITK on this case reports identity direction; nibabel RAS affine is the LPS↔RAS flip
of the same geometry. Registration code in this repo uses the `itk` package
(`itk.imread(..., itk.F)`), not SimpleITK, matching `[nanoUNet] nanounet/register/`.

### BL ids vs FU ids

`[nanoUNet] docs/reference/track_ids.md`: BL click names (or `--bl-mask` voxel values) are
canonical. `paint_fu` / `fu_track_map` always rewrite the FU mask after matching. Phase 1
FU provisional ids are never the published `fu.mha` values.

### What `extra_propagated` must be

`build_mask_graph` does `prop[lid] * sp_fu` with `centroids(mk_*, ids)` on **Frame B**
arrays. Live-registration output must be Frame B `(x, y, z)`, the same convention as
`parse_zyx(cog_propagated)` and `_from_json`.

---

## 2. Phase 1 — stop dropping predicted lesions

### 2.1 What is wrong

`[lesion-tracking] tracking/data/instances.py:46-66` `binary_to_instances`:

1. `cc3d.connected_components(..., connectivity=18)`
2. For each click, read `lab[z, y, x]` (Frame A, click already swapped by `load_clicks`)
3. If that voxel is 0, `continue` — the CC is never written into `lut`
4. Unclaimed CCs stay `lut[cc]==0` → output background

Call sites in `[nanoUNet] nanounet/infer/segtrack.py`:

- line 97: **FU always** (including `--bl-mask` / `--bl-mask-dir`)
- lines 115-116: BL and FU when BL is predicted (no `--bl-mask-dir`)

`--bl-mask-dir` BL path uses `load_instance_zyx` (line 90). Phase 1 does not touch that
branch. The 63-case followup run used `--bl-mask-dir targetsTrBL`, so the zero Dice is
**FU instance labeling**, not BL.

Why "click on FG" is the wrong test:

1. **FU JSON is `cog_propagated`, not `cog_fu`.** Verified: `inputsTrFU/307fd7f231_00.json`
   lesion 1 equals `cog_propagated` exactly, not `cog_fu`. The click can miss the segmented
   blob by several millimetres. On EMA pred
   `/nnunet_data/Longitudinal-CT/results/preds_ema_finetune/307fd7f231_00.nii.gz`:
   **2 CCs, 2 clicks, 0 exact hits, 13433 FG voxels, followup `n_pred=0`.**
2. **The matcher does not use FU click identity.** `build_mask_graph` sets
   `fu_ids = _labels(mk_fu)` and `centroids(mk_fu, fu_ids)`. Dropping a CC because the
   click missed it deletes a detection the GNN would have used.

For predicted BL (no `--bl-mask-dir`), identity **does** matter. Same function still drops
pieces of a split lesion the click did not land on.

### 2.2 The fix

Replace `binary_to_instances` with `label_instances`. Never drop a CC. A CC a click hits
keeps that click's id. Any other CC gets a fresh id. No `SNAP_RADIUS`.

**`[lesion-tracking] tracking/data/instances.py`** — replace `binary_to_instances`
(lines 46-66). Keep `load_clicks` unchanged. Point `instances_from_nifti` (line 73) at
`label_instances`. Delete `binary_to_instances` (one function, not two paths).

```python
def label_instances(pred: np.ndarray, clicks_zyx: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """pred is bool/0-1, Frame A (z,y,x), same grid as load_clicks. Voxel = lesion_id.

    Every predicted 18-connected component becomes an instance. A CC a click lands on
    keeps that click's id. A CC with no click keeps a fresh id instead of being deleted.
    Fresh ids start after max(click ids) so they cannot steal a canonical BL id.
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

`cprint` is already imported from `tracking.common`. Clamp is the same as today's function.

**`[nanoUNet] nanounet/infer/segtrack.py`**

- line 66: `from tracking.data.instances import label_instances, load_clicks`
- line 97 and lines 115-116: `label_instances(...)` instead of `binary_to_instances(...)`

Do not touch the `--bl-mask-dir` BL load.

### 2.3 Verification (before Phase 2)

On-disk EMA preds + FU clicks; no UNet:

```python
import json
import numpy as np
import cc3d
import SimpleITK as sitk
from tracking.data.instances import label_instances, load_clicks

# expected: (n_cc, n_clicks, n_exact_hits) measured 2026-08-29 on preds_ema_finetune
# 307fd7f231_00: (2, 2, 0)
# 38b18881fc_00: (13, 13, 10)
# bf97f24695_00: (25, 23, 19)
# 0f49c89d1e_00: (8, 9, 2)
for stem, n_cc_expect in [("307fd7f231_00", 2), ("38b18881fc_00", 13),
                          ("bf97f24695_00", 25), ("0f49c89d1e_00", 8)]:
    ema = sitk.GetArrayFromImage(sitk.ReadImage(
        f"/nnunet_data/Longitudinal-CT/results/preds_ema_finetune/{stem}.nii.gz"))
    clicks = load_clicks(f"/nnunet_data/Longitudinal-CT/inputsTrFU/{stem}.json")
    inst = label_instances(ema > 0, clicks)
    n_cc = int(cc3d.connected_components((ema > 0).astype(np.uint8), connectivity=18).max())
    n_kept = int(np.unique(inst).size) - 1
    assert n_kept == n_cc == n_cc_expect, (stem, n_kept, n_cc)
    print(stem, "ok", n_kept)
```

Pass: `n_kept == n_cc` for all four. `307fd7f231_00` must keep 2, not 0.

Then one real `nanounet_segtrack` on `307fd7f231_00` with `--bl-mask` + `--keep-pred --overwrite`
(and `--ema` until Phase 2 lands). `pred_fu.mha` has FG; `fu.mha` must also have FG.

Current followup: `307fd7f231_00` `n_pred=0`, `dices=[0.0, 0.0]`.

---

## 3. Phase 2 — EMA default

`[nanoUNet] nanounet/cli/segtrack.py:53`

```python
ap.add_argument("--ema", action="store_true")
```

Default off. Docs (`docs/steps/track.md:68`) agree. Matcher EMA is a different thing
(`use_ema=True` hardcoded in `run_case` line 156; config table `"track-ema", "on"`). Seg EMA
is **not** in the config table today.

The project's reference preds are `preds_ema_finetune`. Default the seg checkpoint to EMA.

Replace line 53 with:

```python
ap.add_argument("--no-ema", dest="ema", action="store_false", default=True)
```

Do **not** also keep `--ema` (argparse conflict). `load_net_from_ckpt(..., ema=args.ema)` at
line 128 stays.

Config table (after the `track-ema` row, ~line 100):

```python
("seg-ema", "on" if args.ema else "off", "default" if args.ema else "cli"),
```

**`[nanoUNet] docs/steps/track.md`** argument table: replace the `--ema` row with

```markdown
| `--no-ema` | flag | off | Load the raw seg checkpoint. Default is EMA. Matcher EMA is always on |
```

Do **not** change `nanounet_predict` (`[nanoUNet] nanounet/cli/predict.py:34`).

### Verification

`nanounet_segtrack --help` shows `--no-ema`. Printed config table: `seg-ema on default`.
`--no-ema` prints `seg-ema off cli` and still loads.

---

## 4. Phase 3 — live BL→FU points, reuse existing uniGradICON

### 4.1 What this replaces

`run_case` lines 141-151:

```python
prop = case.meta_csv if case.meta_csv is not None else case.fu_clicks
...
got, _ = load_propagated(prop, bl_ids, img_id=img_id)
drop = sorted(set(bl_ids) - set(got))
```

Two bugs:

1. `fu_clicks` as propagated is only valid on *this* dataset, where `inputsTrFU/*.json`
   was built from `cog_propagated`. On any other dataset a FU click JSON is "where the user
   clicked now", not "BL centroid in FU space". Axis order of `_from_json` is fine (Frame B);
   the **semantics** are wrong.
2. BL ids not in the CSV are logged and omitted from the matcher (`bl_ids = [i for i in all_bl if i in prop]`).

Folder mode (`[nanoUNet] nanounet/cli/segtrack_cases.py:61-72`) **skips** `no meta csv` when
a meta dir exists (inferred `{bl-dir}/../meta` or `--meta-dir`). Single mode infers
`{bl-img}/../../meta/{pid}.csv` (`segtrack_cases.py:117-124`). On Longitudinal-CT you
cannot "just omit `--meta`" — the CSV is auto-attached.

### 4.2 Do not create `tracking/data/unigradicon.py`

That file does not exist, and it must not. `[nanoUNet] nanounet/register/unigradicon.py`
already loads weights, calls `register_pair`, and warps BL onto FU. nanoUNet already
depends on `unigradicon`. lesion-tracking stays a GNN package: it accepts an
`extra_propagated` dict and never imports `itk` / `unigradicon`.

Production A/B convention (`warp_pair` lines 63-68), verified in source:

```python
phi_AB, _ = itk_wrapper.register_pair(get_model(), bl_pp, fu_pp, finetune_steps=steps)
warped_img = resample_to(bl, phi_AB, fu, default=-1000.0)
```

`image_A=BL`, `image_B=FU`. ICON comment: `warp(image_A, phi_AB_itk) is close to image_B`.
ITK resample pulls: `phi_AB.TransformPoint(point in FU physical) → BL physical`.

The reverse map `phi_BA` is `create_itk_transform(phi_BA, ident, image_B, image_A)` =
`(FU, BL)` → `warp(FU, phi_BA) ≈ BL` → `phi_BA.TransformPoint(BL physical) → FU physical`.

**Point propagation uses `phi_BA` from the same `register_pair(model, bl, fu)` as `warp_pair`.
Do not swap A/B relative to `warp_pair`.** If Phase 3.4 fails, the ordered retries are
(1) `phi_AB.GetInverseTransform()` with the same A/B, (2) swapped A/B using `phi_AB` as
the previous draft claimed. Re-derive from `warp_pair`, do not guess.

### 4.3 Coarse origin shift is out

The previous draft added `_coarse_align` (metadata `SetOrigin`). `register_pair` does
`F.interpolate` on the **raw pixel array** to 175³ (`icon_registration/itk_wrapper.py`).
Origin is not in that tensor. Origin-only translation does not change what the network
sees. `resampling_transform` uses origin when composing the ITK composite; that already
handles header differences.

`warp_pair` has no origin shift and is the production backend. `warp_case` already
resamples pixels (`landmark_align`) when `frame_z_overlap_mm <= 0`. Do not invent a third
pre-align. If Phase 3.4 fails with a huge miss, escalate to that existing pixel resample —
not an origin hack. Out of scope until 3.4 fails.

`BODY_HU = -300` already lives in `elastix.py`. Do not add `-500`.

### 4.4 Weights

Already in `unigradicon.py`:

```python
WEIGHTS_PATH = os.environ.get(
    "NANOUNET_UNIGRADICON_WEIGHTS",
    os.path.expanduser("~/.cache/nanounet/unigradicon/Step_2_final.trch"),
)
WEIGHTS_URL = "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch"
```

`get_model()` already `os.makedirs` the parent, then `get_unigradicon(weights_location=...)`.
(uniGradICON's own `makedirs("network_weights/unigradicon1.0/")` is a relative CWD path and
is **not** the download target when `weights_location` is set. Parent mkdir is required;
`get_model` already does it.)

As of 2026-08-29 the cache file is **missing** on this machine. `unigradicon` **is**
installed (`/usr/local/lib/python3.11/site-packages/unigradicon`). First `get_model()` will
download. If download is blocked:

```bash
mkdir -p ~/.cache/nanounet/unigradicon
curl -L -o ~/.cache/nanounet/unigradicon/Step_2_final.trch \
  https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch
```

Do **not** invent `$NANOUNET_RESULTS/unigradicon/unigradicon1.0/Step_2_final.trch`.
Do **not** add `DEFAULT_UNIGRADICON_*` to `tracking/common.py`.
Do **not** add `--unigradicon-weights` (env already exists; `cli/segtrack.py` is at 188 LOC).

### 4.5 `icon_registration.config.device`

`icon_registration/config.py`: `device = torch.device("cuda")` if CUDA else CPU.
`register_pair` does `model.to(config.device)` and can override a placement in `get_model`.
`get_unigradicon` also `net.to(config.device)`.

Before `register_pair`, set:

```python
import icon_registration.config as icon_config
import torch
icon_config.device = torch.device(device)
```

`device` comes from `nanounet_segtrack --device` (default `cuda`). `_MODEL` in `get_model`
is a single global — first call wins. Acceptable; segtrack is one process, one device.

### 4.6 New function in `[nanoUNet] nanounet/register/unigradicon.py`

Append. File is 73 LOC; this stays under 200. Lazy-import `itk` / `itk_wrapper` /
`preprocess` / `icon_registration.config` inside the function (the module already imports
`resample_to` from elastix at top — that is existing). `numpy` is a new module-level or
function-body import; inside the function is fine.

`register_pair` **always** `print(loss)` (ICON source). Quiet it: this is a user-facing
segtrack path (nanochat U5 / U7). `register_longi` already has `_quiet_stderr`; this print
is stdout.

```python
def propagate_points(
    bl_ct_path: "os.PathLike | str",
    fu_ct_path: "os.PathLike | str",
    points_xyz: dict[int, np.ndarray],
    *,
    io_iterations: int = 0,
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """BL Frame-B (x,y,z) voxel coords → FU Frame-B (x,y,z) via uniGradICON.

    Same register_pair(model, bl, fu) as warp_pair. Uses phi_BA to map BL physical → FU
    physical. io_iterations<=0: one forward pass. io_iterations>0: ICON instance optimization
    (register_longi defaults to 50; live segtrack default is 0).
    """
    import contextlib
    import numpy as np
    import torch
    import icon_registration.config as icon_config
    import icon_registration.itk_wrapper as itk_wrapper
    import itk
    from unigradicon import preprocess

    if not points_xyz:
        return {}

    icon_config.device = torch.device(device)
    model = get_model()
    bl = itk.imread(str(bl_ct_path), itk.F)
    fu = itk.imread(str(fu_ct_path), itk.F)
    steps = io_iterations if io_iterations and io_iterations > 0 else None
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _phi_AB, phi_BA = itk_wrapper.register_pair(
            model, preprocess(bl, "ct"), preprocess(fu, "ct"), finetune_steps=steps,
        )
    out: dict[int, np.ndarray] = {}
    for lid, xyz in points_xyz.items():
        xyz = np.asarray(xyz, dtype=np.float64)
        p_bl = bl.TransformContinuousIndexToPhysicalPoint(
            [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        )
        p_fu = phi_BA.TransformPoint(p_bl)
        idx = fu.TransformPhysicalPointToContinuousIndex(p_fu)
        out[int(lid)] = np.array([idx[0], idx[1], idx[2]], dtype=np.float64)
    return out
```

If `TransformContinuousIndexToPhysicalPoint` rejects a Python list on this ITK 5.4.7
build, use the same pattern as `landmarks.py:40-41` (`TransformIndexToPhysicalPoint` with
rounded ints) only as a last resort — centroids are subvoxel (`+0.5` in `centroids()`).
Prefer:

```python
cidx = itk.ContinuousIndex[itk.D, 3]()
cidx[0], cidx[1], cidx[2] = float(xyz[0]), float(xyz[1]), float(xyz[2])
p_bl = bl.TransformContinuousIndexToPhysicalPoint(cidx)
```

Add `import numpy as np` at function body (already in the snippet). Type hints that
reference `np.ndarray` need numpy imported at type-check time — use
`from __future__ import annotations` (already in the file) so hints are strings.

### 4.7 Validation gate — required before Phase 4

Throwaway script, delete after (R16). Do **not** wire `run_case` until this prints a
sensible point.

```python
from pathlib import Path
import numpy as np
from nanounet.register.unigradicon import propagate_points

# Frame B (x,y,z) — CSV / JSON order. NOT reversed.
bl_xyz = np.array([199.601855155191, 235.505440599358, 122.067160185516])
cog_prop = np.array([207.95331503893644, 286.0173397756782, 119.03847670485004])
cog_fu = np.array([210.814898057473, 304.457296516295, 126.892438593675])
fu_sp = np.array([0.841796875, 0.841796875, 3.0])  # ITK spacing (x,y,z)
shape = np.array([512, 512, 267])  # FU nibabel / Frame B size

out = propagate_points(
    "/nnunet_data/Longitudinal-CT/inputsTrBL/307fd7f231_00.nii.gz",
    "/nnunet_data/Longitudinal-CT/inputsTrFU/307fd7f231_00.nii.gz",
    {1: bl_xyz},
    io_iterations=0,
    device="cuda",
)[1]
print("computed xyz", out)
print("cog_propagated   ", cog_prop, "err_mm", np.linalg.norm((out - cog_prop) * fu_sp))
print("cog_fu           ", cog_fu, "err_mm", np.linalg.norm((out - cog_fu) * fu_sp))
print("in volume", np.all((out >= -1) & (out <= shape)))
```

**Pass (all of):**

1. `out` is inside the FU grid (allow a few voxels of margin). A swapped A/B typically
   lands hundreds of mm away or outside `[0,512)×[0,512)×[0,267)`.
2. `||(out - cog_fu) * fu_sp||` is tens of mm, not hundreds. Need not match `cog_propagated`
   (different offline backend) or `cog_fu` (GT; registration is not identity).
3. Wall time with `io_iterations=0` is single-digit seconds on GPU. Much slower →
   `icon_registration.config.device` is CPU (see 4.5).

If fail, retry in the order in 4.2. Time the call.

---

## 5. Phase 4 — wire fallback into matcher + `run_case` + case collection

Only after 4.7 passes.

### 5.1 `[lesion-tracking] tracking/data/masks.py::build_mask_graph`

Current geo branch (lines 63-70) raises if `propagated_csv is None`. Change to:

```python
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

Add `extra_propagated` to the signature after `img_id`. Keep
`if not bl_ids or not fu_ids: return None` (lines 73-74). File is 101 LOC; stays under 200.

Deployed ckpt has `type_mask=false`, so empty `types` + `default_lesion_type="unclear"` is OK.

### 5.2 `[lesion-tracking] tracking/infer.py::track`

Signature (lines 93-113): add

```python
    extra_propagated: dict[int, np.ndarray] | None = None,
) -> TrackResult:
```

Volumes branch guard today (lines 145-150):

```python
        if not gcfg.drop_dp and (propagated is None or not Path(propagated).is_file()):
```

Change to (use `is None`, not truthiness — `{}` must not trip the "missing extra" path):

```python
        if not gcfg.drop_dp and extra_propagated is None and (
            propagated is None or not Path(propagated).is_file()
        ):
            raise FileNotFoundError(
                f"No propagated at {propagated}.\n"
                f"Expected meta CSV, slim CSV, FU-frame JSON, or extra_propagated from live registration.\n"
                f"Fix: pass --meta / --meta-dir, or omit --drop-uncovered so nanounet_segtrack can register\n"
                f"(see docs/steps/track.md)"
            )
```

`build_mask_graph` call (lines 154-158): pass `extra_propagated=extra_propagated`.
When `propagated is None`, pass `None` into `build_mask_graph`, **not** `Path(propagated)`.

```python
        None if gcfg.drop_dp else (None if propagated is None else Path(propagated)),
```

and `extra_propagated=extra_propagated`.

Leave the `volumes is None` branch (lines 131-143, `lesion_track` CLI) unchanged.

File is 181 LOC; stays under 200.

### 5.3 `[nanoUNet] nanounet/infer/segtrack.py::run_case`

Signature (lines 62-64): add `no_live_registration: bool = False, io_iterations: int = 0`.

Replace lines 141-159. `mk_bl` is already Frame B (lines 137-138). Use `centroids(mk_bl, missing)`,
**not** `centroids(bl_zyx, missing)` (that would be Frame A).

```python
    drop_dp = bool(getattr(matcher.hparams, "drop_dp", False))
    prop = case.meta_csv  # None ⇒ no CSV; do not substitute fu_clicks
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
            from nanounet.register.unigradicon import propagate_points
            extra_propagated = propagate_points(
                case.bl_img, case.fu_img, centroids(mk_bl, missing),
                io_iterations=io_iterations, device=device,
            )
            cprint(f"[dim]{case.stem}  live-registered {len(missing)} BL id(s) via uniGradICON[/dim]")
        elif missing:
            cprint(f"[dim]drop {case.stem}  BL ids {missing} (no propagated coverage, --drop-uncovered)[/dim]")
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

This applies to both `--bl-mask-dir` BL and predicted BL.

File is 171 LOC. If `wc -l` ≥ 200 after the edit, extract nothing new — tighten the block
above (it replaces a similar-sized block).

### 5.4 `[nanoUNet] nanounet/cli/segtrack.py`

`_mode()` after `--no-ema` / `--batch-size`:

```python
    ap.add_argument("--drop-uncovered", action="store_true",
                    help="Omit BL ids with no CSV coverage instead of live-registering them")
    ap.add_argument("--io-iterations", type=int, default=0,
                    help="uniGradICON instance-optimization steps; 0 = one forward pass")
```

No weights flag. Config table:

```python
    ("drop-uncovered", "on" if args.drop_uncovered else "off", "cli" if args.drop_uncovered else "default"),
    ("io-iterations", args.io_iterations, "cli" if args.io_iterations else "default"),
```

`run_case(...)` call (lines 152-157): add
`no_live_registration=args.drop_uncovered, io_iterations=args.io_iterations`.

File is 188 LOC. After edit, `wc -l` must be < 200. Compact `add_argument` lines the same
way lines 38-56 already do if needed.

### 5.5 `[nanoUNet] nanounet/cli/segtrack_cases.py::_folder`

Without this, folder mode on a tree with no `meta/` never had a problem (`md is None`,
cases keep `meta_csv=None`), but a **partial** meta dir still skips. Change the skip so
missing CSV is live-reg, not skip — unless `--drop-uncovered`:

```python
    if md is not None:
        keep = []
        for c in cases:
            pid, _ = stem_pid_region(c.stem)
            p = md / f"{pid}.csv"
            if p.is_file():
                c.meta_csv = p
                c.types_csv = p
                keep.append(c)
            elif getattr(args, "drop_uncovered", False):
                skipped.append((c.stem, "no meta csv"))
            else:
                keep.append(c)
        cases = keep
```

Single-mode inferred meta (`_single` lines 117-124) stays: if the file exists, use it
(Longitudinal-CT eval). To test no-CSV, copy two cases to a folder that has **no** sibling
`meta/` (see §6.6). Do not add `--no-meta`.

### 5.6 Docs — `[nanoUNet] docs/steps/track.md`

Same change as the CLI. Argument table (D3 columns). Add:

```markdown
| `--drop-uncovered` | flag | off | Skip BL ids with no `cog_propagated` instead of live uniGradICON |
| `--io-iterations` | int | `0` | uniGradICON IO steps. `0` = one forward pass (seconds). `50` matches `nanounet_register_longi` |
```

Short paragraph under Inputs (replace the sentence that says matcher BL positions only
come from meta CSV):

Matcher BL positions: `{dataset}/meta/{pid}.csv` `cog_propagated` when that file exists
(inferred from `--bl-dir` parent or `--meta-dir`). Any BL id the CSV does not cover is
live-registered with uniGradICON (`nanounet/register/unigradicon.py`, same
`register_pair(bl, fu)` as `nanounet_register_longi --backend unigradicon`). Weights:
`$NANOUNET_UNIGRADICON_WEIGHTS` or `~/.cache/nanounet/unigradicon/Step_2_final.trch`. A
`drop_dp` matcher does not need this. `--drop-uncovered` restores the old omit behaviour.

Update the errors table:

| old | new |
|-----|-----|
| `skip {stem} (no meta csv)` | only with `--drop-uncovered` |
| `Empty instance mask \| No click hit predicted FG` | Empty only if the binary pred has no FG CC |

`docs/steps/track.md` is 102 LOC; D4 < 200 still holds. Do not mention this plan file in
user-facing docs.

---

## 6. End-to-end protocol

Each step gates the next. Throwaway scripts are deleted after they pass (R16).
`scripts/score_segtrack_fu.py` is **not in the repo**. Recreate it when you need it (§6.7);
do not look for it.

### 6.1 Phase 1 snippet

§2.3. Four stems, `n_kept == n_cc`.

### 6.2 Phase 1 one case through the CLI

Until Phase 2, pass `--ema`.

```bash
nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/307fd7f231_00.nii.gz \
  --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/307fd7f231_00.nii.gz \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/307fd7f231_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/307fd7f231_00.json \
  -o "$NANOUNET_RESULTS/segtrack/phase1_check/307fd7f231_00" \
  --overwrite --keep-pred --ema
```

`fu.mha` FG > 0 wherever `pred_fu.mha` FG > 0. Meta will be **inferred** from
`Longitudinal-CT/meta/307fd7f231.csv` — that is intended for this step (CSV-covered BL ids).

### 6.3 Phase 2

Config table `seg-ema on default`. `--no-ema` still loads.

### 6.4 Phase 3.4

§4.7. Do not start §5 until it passes.

### 6.5 Phase 4 regression (CSV-covered, no live reg)

Rerun `307fd7f231_00` as in 6.2 **without** `--ema` (default on). uniGradICON must **not**
run: every GT BL id is in the CSV. Confirm: no `live-registered` log line; no
`propagate_points` import if you temporarily print in `get_model`. Timing unchanged.

### 6.6 Phase 4 no-CSV

Longitudinal-CT always infers `meta/`. Copy:

```bash
tmp=/tmp/segtrack_nocsv
mkdir -p "$tmp/bl" "$tmp/fu" "$tmp/blm"
for s in 307fd7f231_00; do
  cp /nnunet_data/Longitudinal-CT/inputsTrBL/$s.nii.gz "$tmp/bl/"
  cp /nnunet_data/Longitudinal-CT/inputsTrBL/$s.json "$tmp/bl/" 2>/dev/null || true
  cp /nnunet_data/Longitudinal-CT/inputsTrFU/$s.nii.gz "$tmp/fu/"
  cp /nnunet_data/Longitudinal-CT/inputsTrFU/$s.json "$tmp/fu/"
  cp /nnunet_data/Longitudinal-CT/targetsTrBL/$s.nii.gz "$tmp/blm/"
done
nanounet_segtrack \
  --bl-dir "$tmp/bl" --fu-dir "$tmp/fu" --bl-mask-dir "$tmp/blm" \
  -o "$NANOUNET_RESULTS/segtrack/nocsv_check" --overwrite --keep-pred
```

`{bl-dir}/../meta` is `/tmp/segtrack_nocsv/meta` — absent ⇒ `meta_csv is None`. Must
complete (no `FileNotFoundError`), log `live-registered`, write `matches.csv` with pairs,
`fu.mha` ids comparable to `bl.mha`.

### 6.7 Full 63-case rerun

```bash
nanounet_segtrack \
  --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL \
  --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU \
  --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL \
  --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv \
  -o "$NANOUNET_RESULTS/segtrack/followup" \
  --overwrite --keep-pred
```

Throwaway scorer (delete after). Matches the existing JSON schema
(`agg.median_comp_lesion` over all GT lesions, miss = 0):

```python
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

gt_dir = Path("/nnunet_data/Longitudinal-CT/targetsTrFU")
pred_dir = Path("/nanounet_data/NanoUNet_results/segtrack/followup")
cases, all_d = [], []
for fu in sorted(pred_dir.glob("*/fu.mha")):
    stem = fu.parent.name
    gtp = gt_dir / f"{stem}.nii.gz"
    if not gtp.is_file():
        continue
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(fu)))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gtp)))
    gids = [int(i) for i in np.unique(gt) if i != 0]
    pids = [int(i) for i in np.unique(pred) if i != 0]
    dices = []
    for lid in gids:
        p, g = pred == lid, gt == lid
        dices.append(float(2 * (p & g).sum() / (p.sum() + g.sum())) if (p.any() or g.any()) else 0.0)
    det = [d for d, lid in zip(dices, gids) if lid in pids]
    rec = dict(stem=stem, n_gt=len(gids), n_pred=len(pids),
                n_id_both=len(set(gids) & set(pids)),
                vol_dice=float(2 * ((pred > 0) & (gt > 0)).sum() / ((pred > 0).sum() + (gt > 0).sum()))
                         if (pred > 0).any() or (gt > 0).any() else 0.0,
                mean_comp=float(np.mean(dices)) if dices else 0.0,
                median_comp=float(np.median(dices)) if dices else 0.0,
                mean_detected=float(np.mean(det)) if det else float("nan"),
                n_detected=len(det), dices=dices)
    cases.append(rec)
    all_d.extend(dices)
agg = dict(n_cases=len(cases), n_gt_lesions=sum(c["n_gt"] for c in cases),
            n_pred_lesions=sum(c["n_pred"] for c in cases),
            n_detected=sum(c["n_detected"] for c in cases),
            median_comp_lesion=float(np.median(all_d)) if all_d else 0.0,
            mean_detected=float(np.nanmean([c["mean_detected"] for c in cases])))
print(json.dumps(agg, indent=2))
Path(pred_dir / "dice_vs_targetsTrFU.json").write_text(json.dumps({"agg": agg, "cases": cases}))
```

Expect `median_comp_lesion` **off 0.0**. Phase 1 alone should recover the dropped CCs
(`307fd7f231_00` must leave `n_pred=0`). Do not require matching `cog_propagated` quality
on the no-CSV path in this rerun (this rerun still has inferred meta).

---

## 7. LOC budget (nanochat R1: <200)

| file | now | note |
|------|-----|------|
| `[nanoUNet] nanounet/cli/segtrack.py` | 188 | tight — compact flags if needed |
| `[nanoUNet] nanounet/infer/segtrack.py` | 171 | replace a block, don't append a second |
| `[nanoUNet] nanounet/register/unigradicon.py` | 73 | add `propagate_points` |
| `[nanoUNet] nanounet/cli/segtrack_cases.py` | 124 | skip-logic change |
| `[nanoUNet] nanounet/cli/predict.py` | 198 | **do not touch** |
| `[lesion-tracking] tracking/infer.py` | 181 | extra param + guard |
| `[lesion-tracking] tracking/data/masks.py` | 101 | extra param |
| `[lesion-tracking] tracking/data/instances.py` | 78 | replace function |
| `[nanoUNet] docs/steps/track.md` | 102 | D4 |

After every edit: `wc -l` the file. Split only on a concept boundary, not to dodge the cap.

---

## 8. Out of scope (do not build)

- `tracking/data/unigradicon.py` or `[project.optional-dependencies]` on lesion-tracking.
- Origin-only `_coarse_align`. Pixel-space pre-align (`warp_case.landmark_align` /
  elastix rigid) only if 4.7 fails.
- Wiring live reg into `lesion_track` (`volumes is None`).
- `nanounet_predict --ema` default.
- Renaming / swapping `parse_zyx` (would desync the trained matcher).
- "Nicer" numbers for Phase 1 synthetic ids. Constraint is `track_ids.md`: same integer =
  same lesion.
- MRI (`preprocess(..., modality="ct")` stays).
- Changing `warp_pair` image-warping behaviour.

---

## 9. What the previous draft of this file got wrong

| claim | reality |
|-------|---------|
| All bare triples are `(z,y,x)` | Two frames; CSV/JSON/matcher are `(x,y,z)` |
| §4.4 used CSV as `bl_zyx` and reversed `cog_propagated` | Mixed frames in one script |
| New `tracking/data/unigradicon.py` | Duplicate of `nanounet/register/unigradicon.py` |
| `register_pair(A=FU, B=BL)` then `phi_AB(BL)→FU` | Opposite of production `warp_pair` |
| Origin shift as "required" pre-align | No-op on the 175³ network input |
| Weights under `$NANOUNET_RESULTS/unigradicon/...` | Env + `~/.cache/nanounet/unigradicon/` |
| `unigradicon` not installed | Installed; weights not cached |
| Omit `--meta` to test no-CSV on this dataset | Meta is inferred; must copy to a tree without `meta/` |
| Folder skip unchanged | Must stop skipping missing CSV unless `--drop-uncovered` |
| `centroids(bl_zyx, missing)` into propagate | Frame A into a Frame-B API |
| Scoring command in `docs/steps/track.md` | Not there; script was never committed |
| `cli/segtrack.py` has room for weights + two flags | 188 LOC; env for weights, two flags max |
