# segtrack speedup: sparse canvas + click-AABB preprocess + prefetch

Implementation plan for a lesser coding agent with **no prior context**. Follow it literally.
Read `.claude/skills/nanochat-style/SKILL.md` first — every edit below must obey it (files
`<200` LOC, no bare `print`, rich `cprint`/tables for any new CLI output, comments explain
*why* not *what*, boundary errors use the 3-question template).

**Ground rules, non-negotiable:**

1. **TTA stays exactly as-is.** `disable_tta_default`, 8-fold mirroring, `predict_batch_with_tta` —
   do not touch. Not in scope.
2. **Identical output is the gate, not a nice-to-have.** For every step below: run the two
   reference cases before the change, run them after, `cmp` the outputs byte-for-byte. If they
   differ, the change is wrong — fix it or revert it. Do not "ship and see." Do not weaken this
   to "looks similar" or "Dice is close."
3. Steps are ordered so each is independently implementable, testable, and revertible. Do not
   start step N+1 until step N's `cmp` gate is clean.
4. `torch.compile` is explicitly **out of scope** (not bitwise-identical to eager — separate
   quality-gated work, not part of this plan).

## Reference cases and baseline

Two cases, already used to validate the previous round of changes (nibabel→SimpleITK, GPU
argmax, load-once):

| stem | source | current time |
|---|---|---|
| `39059724f7_00` | `/nnunet_data/Longitudinal-CT/{inputsTrBL,inputsTrFU,targetsTrBL}` | ~80s |
| `3994f23bfb_00` | same | ~84s |

Both have a GT baseline mask in `targetsTrBL`, so use `--bl-mask` (single mode) — this segments
only FU, matching the live `nanounet_segtrack --bl-mask-dir` workflow this whole effort targets.

## Verification protocol (run before Step A, and after every step)

```bash
mkdir -p /scratch/nielsrocholl/segtrack_prof/ref_before /scratch/nielsrocholl/segtrack_prof/ref_after

run_case() {
  local stem=$1 outdir=$2
  nanounet_segtrack \
    --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/${stem}.nii.gz \
    --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/${stem}.nii.gz \
    --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/${stem}.nii.gz \
    --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/${stem}.json \
    -o ${outdir}/${stem} --keep-pred --overwrite
}

for s in 39059724f7_00 3994f23bfb_00; do run_case $s /scratch/nielsrocholl/segtrack_prof/ref_before; done
# ... apply one step's code change ...
for s in 39059724f7_00 3994f23bfb_00; do run_case $s /scratch/nielsrocholl/segtrack_prof/ref_after; done

for s in 39059724f7_00 3994f23bfb_00; do
  for f in bl.mha fu.mha pred_bl.mha pred_fu.mha matches.csv; do
    cmp /scratch/nielsrocholl/segtrack_prof/ref_before/$s/$f /scratch/nielsrocholl/segtrack_prof/ref_after/$s/$f \
      && echo "OK   $s/$f" || echo "DIFF $s/$f  <-- STOP, do not proceed"
  done
done
```

`ref_before` only needs to be generated once (before Step A); reuse it as the baseline for
every subsequent step's `ref_after` comparison, then move `ref_after` → `ref_before` and clear
`ref_after` before starting the next step. Record wall-clock seconds from the CLI's per-case
`cprint` line (`{stem}  {sec}s`) at every step so the final report has a before/after table.

---

## Step A — sparse logit canvas (`predict_case.py`)

**Problem:** `predict_case_logits` allocates `margin_buf` and `logits_acc` at the *full padded
volume* shape (`(1284, 570, 564)`-class sizes → ~3.3GB for `logits_acc` alone), then only ever
writes into a handful of `patch_size` tiles. The final argmax also runs over the full unpadded
crop even though everywhere outside the written tiles is provably background.

**Fix:** size the canvas to a bbox that is *provably* large enough to contain every tile the
loop can ever write, not the whole volume.

### Why the bbox is provably correct, not a heuristic

`border_expand` schedules at most `max_border_expand_extra` (=`MAX_BORDER_EXTRA`=16) extra
tiles **per seed cluster** — `extras_done[ci] >= max_border_expand_extra` is checked before
every schedule (`predict_case.py:164,167`). Each extra tile is exactly one `stride` step from an
already-visited tile of the same cluster (`cell_slices(origins[ci], nijk, stride, ...)`). So the
maximum possible displacement of any tile from its seed's bbox, on any one axis, is bounded by
`max_border_expand_extra * stride[axis]`. This is a hard bound (proof by the budget check), not
an empirical guess. When `border_expand=False`, no extra tiles are ever scheduled, so the canvas
is exactly the seed bbox.

### 1. Add two functions to `nanounet/infer/roi_slices.py`

Append (imports: add `cluster_points_for_patch_size, spatial_slices_covering_points` from
`nanounet.prompt.cluster` to the existing import block — `centered_spatial_slices_at_point`
is already local to this file):

```python
from nanounet.prompt.cluster import cluster_points_for_patch_size, spatial_slices_covering_points


def seed_slices_for_points(
    pts_pad: List[ZYX],
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
    cluster_margin_frac: float,
    mode: str,
) -> Tuple[List[List[ZYX]], List[Tuple[slice, slice, slice]]]:
    """Seed-tile placement for `mode`. Extracted so preprocessing can compute the same
    tile layout as predict_case_logits before any forward pass runs (click-AABB preprocess)."""
    assert mode in ("clustered", "centered")
    if mode == "clustered":
        seeds_pts = cluster_points_for_patch_size(pts_pad, patch_size, cluster_margin_frac)
        seed_slices = [spatial_slices_covering_points(cl, patch_size, padded_shape) for cl in seeds_pts]
        return seeds_pts, seed_slices
    seen: set = set()
    seeds_pts, seed_slices = [], []
    for p in pts_pad:
        sl = centered_spatial_slices_at_point(p[0], p[1], p[2], patch_size, padded_shape)
        key = (sl[0].start, sl[1].start, sl[2].start, p)
        if key in seen:
            continue
        seen.add(key)
        seeds_pts.append([p])
        seed_slices.append(sl)
    return seeds_pts, seed_slices


def canvas_bbox_for_seeds(
    seed_slices: List[Tuple[slice, slice, slice]],
    stride: Tuple[int, int, int],
    max_border_expand_extra: int,
    border_expand: bool,
    padded_shape: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    """Tight bbox guaranteed to contain every tile predict_case_logits can ever write.

    Border-expand walks at most max_border_expand_extra steps of `stride` per seed cluster
    (extras_done[ci] is checked before each schedule) so seed bbox +/- that reach is a hard
    bound, not a heuristic.
    """
    reach = (0, 0, 0) if not border_expand else tuple(max_border_expand_extra * s for s in stride)
    los = [max(0, min(sl[a].start for sl in seed_slices) - reach[a]) for a in range(3)]
    his = [min(padded_shape[a], max(sl[a].stop for sl in seed_slices) + reach[a]) for a in range(3)]
    return tuple(slice(los[a], his[a]) for a in range(3))
```

Both take only lists/tuples/ints — no torch, no cross-package import cycle (`prompt/cluster.py`
does not import `infer/roi_slices.py`).

### 2. Edit `nanounet/infer/predict_case.py`

Add `seed_slices_for_points, canvas_bbox_for_seeds` to the existing `from nanounet.infer.roi_slices
import (...)` block.

Replace lines 100–113 (the `if mode == "clustered": ... else: ...` seed-building block) with:

```python
    seeds_pts, seed_slices = seed_slices_for_points(pts_pad, patch_size, padded_shape, cluster_margin_frac, mode)
```

Replace lines 126–128:

```python
    neg = torch.finfo(acc_dtype).min
    margin_buf = torch.full(padded_shape, neg, dtype=acc_dtype, device=dev)
    logits_acc = bg_vec.view(-1, 1, 1, 1).to(acc_dtype).expand(nh, *padded_shape).contiguous()
```

with (note: `stride = grid_stride(patch_size, cfg.inference.tile_step_size)` is already computed
at the existing line 115, i.e. *before* this block — `canvas_bbox_for_seeds` can use that
variable directly, no reordering needed; do **not** rename or move it):

```python
    canvas_sl = canvas_bbox_for_seeds(seed_slices, stride, max_border_expand_extra, border_expand, padded_shape)
    canvas_shape = tuple(s.stop - s.start for s in canvas_sl)
    canvas_origin = tuple(s.start for s in canvas_sl)

    neg = torch.finfo(acc_dtype).min
    margin_buf = torch.full(canvas_shape, neg, dtype=acc_dtype, device=dev)
    logits_acc = bg_vec.view(-1, 1, 1, 1).to(acc_dtype).expand(nh, *canvas_shape).contiguous()
```

Note: do not call this variable `origin` — the existing loop just above (`for ci, sl in
enumerate(seed_slices): origin = (sl[0].start, ...)`, lines 120–124) already binds a
loop-scoped `origin` that Python leaves dangling with its last-iteration value after the loop
ends. It happens to never be read again after that loop (only `origins[ci]`, the list, is used
later), so reusing the name would work by accident — but do not rely on that; `canvas_origin`
keeps it unambiguous for the next reader.

In the write loop (originally lines 150–159), keep `sz, sy, sx = sl` as-is (it is still needed
below, unchanged, for the same-slot dedup check against `nsl`) but index `logits_acc`/`margin_buf`
through **canvas-local** slices instead:

```python
        for j, (ci, ijk, sl, _extra) in enumerate(batch):
            raw = out[j].float()
            sz, sy, sx = sl
            csz = slice(sz.start - canvas_origin[0], sz.stop - canvas_origin[0])
            csy = slice(sy.start - canvas_origin[1], sy.stop - canvas_origin[1])
            csx = slice(sx.start - canvas_origin[2], sx.stop - canvas_origin[2])
            m = (raw[1:].amax(0) - raw[0]).to(acc_dtype)
            sub_m = margin_buf[csz, csy, csx]
            keep = m > sub_m
            logits_acc[:, csz, csy, csx] = torch.where(
                keep.unsqueeze(0), raw.to(acc_dtype), logits_acc[:, csz, csy, csx]
            )
            margin_buf[csz, csy, csx] = torch.where(keep, m, sub_m)
            written.append(sl)
```

(everything below `written.append(sl)` — `fwd_done`, `on_forward`, the border-expand
`face_neighbours` loop — is unchanged; it uses the *global* `sz, sy, sx` and global `nsl`, not
canvas coords, exactly as before.)

Replace the final two lines:

```python
    # argmax on device (GPU ~1.5s vs ~50s CPU C-first); D2H is uint8 labels, not logits
    return logits_acc[(slice(None), *slicer_revert[1:])].float().argmax(0).to(torch.uint8).cpu(), tiles
```

with:

```python
    # argmax on device (GPU ~1.5s vs ~50s CPU C-first); D2H is uint8 labels, not logits.
    # Canvas may be smaller than the unpadded crop — everywhere outside it is background by
    # construction (bg_vec argmax == 0), so start from zeros and only fill the overlap.
    seg = torch.zeros(unpadded_shape, dtype=torch.uint8)
    ov = patch_unpadded_overlap(canvas_sl[0], canvas_sl[1], canvas_sl[2], slicer_revert)
    if ov is not None:
        (uz, uy, ux), (cz, cy, cx) = ov
        seg[uz, uy, ux] = logits_acc[:, cz, cy, cx].float().argmax(0).to(torch.uint8).cpu()
    return seg, tiles
```

`patch_unpadded_overlap` is already imported at the top of the file (`from
nanounet.infer.patch_export import patch_unpadded_overlap`).

### 3. LOC check

Net effect on `predict_case.py`: −14 lines (seed block → 1-line call) + ~14 lines (canvas +
local-slice + return block) ≈ flat, should land at or under the current 196 LOC. If it creeps
over 200, move `_accum_dtype`/`ACC_DTYPE_ENV` (lines 33–40) into `roi_slices.py` — they are
small, self-contained, and thematically close to the canvas math now living there.

### 4. Verify

Run the verification protocol. `argmax` is a piecewise-max over channels, so canvas placement
cannot change the result — this step is safe when the bbox proof above holds. If `cmp` fails,
the most likely bug is an off-by-one in the canvas-local slice conversion (`canvas_origin` sign)
or a mismatch between the `stride` used here and the one `predict_case_logits` already used
(must be `grid_stride(patch_size, cfg.inference.tile_step_size)`, unchanged).

---

## Step B — click-AABB preprocess

**Problem:** `preprocess_loaded` (via `run_case_npy` in `plan/prep/case_pp.py`) resamples the
*entire* cropped-to-body CT from native resolution to plan spacing — full torso, even though
Step A proves only a small bbox around the clicks is ever read. The resampled output is also
fully copied D2H (`resample_torch_to_shape` ends in `.cpu().numpy()`), so the GPU write and the
D2H transfer both cost O(torso volume) for O(few tiles) of actual use. `run_case_npy` also
resamples a throwaway `seg` (the crop-to-nonzero surrogate mask, since real `seg=None` for
inference) that no inference call site ever reads — free waste, fixed alongside this.

**This is the highest-risk step.** `F.interpolate(..., align_corners=False)` computes an
implicit `scale = in_size/out_size` *from the actual tensor shapes of that one call*. Cropping
the native input and asking for a smaller output does **not** reproduce the reference full-volume
resample — the local crop's implied scale differs from the global one, so per-voxel values would
silently be wrong (not even bit-off, systematically off — this is why the earlier plan draft
called "crop native, then resample from scratch" a trap; it is not a rounding-error risk, it is
a wrong-formula risk). Do not do that.

**Correct approach:** feed `F.grid_sample` the *full* (already crop-to-nonzero'd, native-res)
input and an explicit coordinate grid that reproduces `F.interpolate`'s own
`align_corners=False` source-index formula, but only evaluated at the AABB's output positions.
`grid_sample` and `interpolate` share the same `align_corners=False` pixel convention in
PyTorch by design — this lets us compute a *sub-region* of what `interpolate` would have
produced, using the exact same global scale, without ever materializing the full output. Input
stays full-size (so there is no missing-neighbor-at-crop-boundary risk); only the *output* shrinks
to the AABB. That output-side shrink is where the GPU compute and the D2H transfer savings come
from — `F.interpolate`'s trilinear kernel is O(output voxels) (each output voxel gathers ~8 input
taps), so this is not a partial win.

### B.1 — Mandatory numeric self-check, before touching any pipeline code

Write this as a throwaway script (`/tmp/check_grid_sample_equiv.py` or similar — delete when
done, per nanochat-style R16), run it, and confirm max abs diff is at machine-precision noise
(<1e-4 for float32) before writing a single line into the real pipeline:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
C, Zi, Yi, Xi = 2, 37, 53, 41
Zo, Yo, Xo = 91, 122, 88
x = torch.randn(1, C, Zi, Yi, Xi)

ref = F.interpolate(x, size=(Zo, Yo, Xo), mode="trilinear", align_corners=False)

# Pick an arbitrary output sub-box to compare against the corresponding slice of `ref`.
lo = (10, 15, 5)
hi = (40, 70, 60)

def src_coord(dst_idx: torch.Tensor, in_size: int, out_size: int) -> torch.Tensor:
    scale = in_size / out_size
    return (dst_idx.float() + 0.5) * scale - 0.5

zg = src_coord(torch.arange(lo[0], hi[0]), Zi, Zo)
yg = src_coord(torch.arange(lo[1], hi[1]), Yi, Yo)
xg = src_coord(torch.arange(lo[2], hi[2]), Xi, Xo)

# grid_sample align_corners=False: normalized = (2*src_idx + 1) / in_size - 1
def norm(idx: torch.Tensor, size: int) -> torch.Tensor:
    return (2 * idx + 1) / size - 1

zz, yy, xx = torch.meshgrid(norm(zg, Zi), norm(yg, Yi), norm(xg, Xi), indexing="ij")
# grid_sample grid channel order is (x, y, z) -- reversed from tensor dim order (D, H, W).
grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)  # (1, Do, Ho, Wo, 3)

out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

diff = (out - ref[:, :, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).abs().max().item()
print("max abs diff:", diff)
assert diff < 1e-4, "grid_sample AABB does not reproduce interpolate — do not proceed with Step B"
```

`mode="bilinear"` on a 5D input performs trilinear sampling in `grid_sample` (PyTorch names it
`bilinear` regardless of spatial dims — this is correct, not a bug). If this assertion fails,
stop — do not implement the rest of Step B; report the numeric mismatch instead of guessing at
a fix.

### B.2 — Refactor `nanounet/plan/prep/case_pp.py`: split `run_case_npy`

Split the existing `run_case_npy` (lines 95–142) into a reusable "everything up to the resample"
half and the existing full-resample tail, so both the untouched training/dataset-prep path and
the new click-AABB path share the crop/normalize logic verbatim (no duplication, no drift risk).

```python
def crop_normalize_case(
    data: np.ndarray,
    seg: np.ndarray | None,
    properties: dict,
    plans: Plans,
    cm: Config3d,
) -> tuple[np.ndarray, np.ndarray, dict, list, list, tuple[int, int, int]]:
    """Transpose to plan axes, crop to nonzero body bbox, per-channel normalize.

    Returns (data, seg, properties, o_sp, t_sp, new_sh) where `data`/`seg` are still at native
    (cropped) resolution -- the caller resamples. `seg` is the real segmentation when one was
    given, otherwise crop_to_nonzero's own surrogate mask (used only for normalization
    masking) -- never None.
    """
    data = data.astype(np.float32)
    if seg is not None:
        assert data.shape[1:] == seg.shape[1:]
        seg = np.copy(seg)
    tf = plans.transpose_forward
    data = data.transpose([0, *[i + 1 for i in tf]])
    if seg is not None:
        seg = seg.transpose([0, *[i + 1 for i in tf]])
    o_sp = [properties["spacing"][i] for i in tf]
    properties["shape_before_cropping"] = data.shape[1:]
    data, seg, bbox = crop_to_nonzero(data, seg)
    properties["bbox_used_for_cropping"] = bbox
    properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]
    t_sp = list(cm.spacing)
    if len(t_sp) < len(data.shape[1:]):
        t_sp = [o_sp[0]] + t_sp
    new_sh = tuple(int(x) for x in compute_new_shape(data.shape[1:], o_sp, t_sp))
    fi = plans.foreground_intensity_properties_per_channel
    for c in range(data.shape[0]):
        cls = normalization_class_from_plan_name(cm.normalization_schemes[c])
        ip = fi[str(c)] if str(c) in fi else fi[int(c)]
        nrm = cls(use_mask_for_norm=cm.use_mask_for_norm[c], intensityproperties=ip)
        data[c] = nrm.run(data[c], seg[0])
    return data, seg, properties, o_sp, t_sp, new_sh


def run_case_npy(
    data: np.ndarray,
    seg: np.ndarray | None,
    properties: dict,
    plans: Plans,
    cm: Config3d,
    dataset_json: dict,
    verbose: bool = False,
):
    has_seg = seg is not None
    data, seg, properties, o_sp, t_sp, new_sh = crop_normalize_case(data, seg, properties, plans, cm)
    data = np.asarray(cm.resampling_fn_data(data, new_sh, o_sp, t_sp))
    if has_seg:
        seg = np.asarray(cm.resampling_fn_seg(seg, new_sh, o_sp, t_sp))
    if verbose:
        cprint(f"[dim]pp {data.shape} {seg.shape if has_seg else None} {new_sh} {o_sp} {t_sp}[/dim]")
    if has_seg:
        lm = labels_from_dataset_json(dataset_json)
        coll = list(lm.foreground_labels)
        if lm.has_ignore_label:
            coll.append([-1] + lm.all_labels)
        properties["class_locations"] = sample_foreground_locations(seg, coll, verbose=verbose)
        seg = seg.astype(np.int16 if np.max(seg) > 127 else np.int8)
    else:
        seg = None
    return data, seg, properties
```

Note the behavior change here (intentional, in scope, zero risk): when `seg` was `None` on
input, `run_case_npy` now returns `seg=None` instead of a resampled-then-discarded surrogate —
skips one full-torso `resampling_fn_seg` call. Verify no caller dereferences the returned `seg`
when it passed `seg=None` in: `preprocess_loaded`/`preprocess_case` in `predict_io.py` both
already discard it as `_seg`. `run_case_save` (dataset prep) always passes a real `seg_file`, so
`has_seg=True` there and behavior is byte-identical to before. Run the verification protocol
after this refactor alone, before adding the AABB path, to isolate any regression.

### B.3 — New file `nanounet/infer/grid_resample.py`

The generic, standalone-testable primitive (the one verified in B.1), operating on real case
data:

```python
"""Resample a sub-region (AABB) of what F.interpolate(align_corners=False) would produce,
without materializing the full output. Feeds the FULL native input to grid_sample (so there is
no missing-neighbor-at-crop-boundary risk) and only evaluates the AABB's output positions.
grid_sample and interpolate share the align_corners=False pixel convention in PyTorch by
design; see docs/dev-notes/segtrack_speedup_plan.md Step B.1 for the numeric proof."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _norm_coord(idx: torch.Tensor, in_size: int, out_size: int) -> torch.Tensor:
    scale = in_size / out_size
    src = (idx.float() + 0.5) * scale - 0.5
    return (2 * src + 1) / in_size - 1


def interpolate_aabb(
    data: torch.Tensor,
    out_lo: tuple[int, int, int],
    out_hi: tuple[int, int, int],
    out_shape: tuple[int, int, int],
) -> torch.Tensor:
    """data: (C, Zi, Yi, Xi) on `dev`. Returns (C, dz, dy, dx) matching
    F.interpolate(data[None], size=out_shape, mode='trilinear', align_corners=False)
    [:, out_lo[0]:out_hi[0], out_lo[1]:out_hi[1], out_lo[2]:out_hi[2]] -- without computing
    the full out_shape volume.
    """
    in_shape = tuple(data.shape[1:])
    zg = _norm_coord(torch.arange(out_lo[0], out_hi[0], device=data.device), in_shape[0], out_shape[0])
    yg = _norm_coord(torch.arange(out_lo[1], out_hi[1], device=data.device), in_shape[1], out_shape[1])
    xg = _norm_coord(torch.arange(out_lo[2], out_hi[2], device=data.device), in_shape[2], out_shape[2])
    zz, yy, xx = torch.meshgrid(zg, yg, xg, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)  # grid_sample: channel order (x, y, z)
    out = F.grid_sample(data.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0]
```

~35 LOC. Keep it standalone (no `nanounet.plan`/`nanounet.infer` imports) so it stays unit-testable
in isolation the way B.1's throwaway script tested it.

### B.4 — New file `nanounet/infer/click_resample.py`

Orchestrator: native crop+normalize (via `crop_normalize_case`, reused verbatim) → click-driven
AABB in the *padded* pp frame (via `seed_slices_for_points` + `canvas_bbox_for_seeds`, the exact
same functions Step A uses, so tile coverage is provably identical by construction, not by
re-derivation) → one `interpolate_aabb` call → pack into the same `(pad, slicer_revert, props,
points, bl_points)` tuple `preprocess_loaded` already returns, so call sites need no other change.

```python
"""Click-driven AABB preprocess for segtrack: resamples only the pp-space bbox
predict_case_logits can ever read (see canvas_bbox_for_seeds), instead of the whole torso.
Scoped to single-timepoint (non-longi) segtrack inference only -- preprocess_loaded/
preprocess_case (longi, training) are untouched."""

from __future__ import annotations

import numpy as np
import torch

import nanounet.data.resampling as _rs
from nanounet.infer.grid_resample import interpolate_aabb
from nanounet.infer.roi_slices import canvas_bbox_for_seeds, map_points_zyx_unpadded_to_padded, seed_slices_for_points
from nanounet.plan.prep.case_pp import crop_normalize_case
from nanounet.prompt.cluster import grid_stride
from nanounet.prompt.coords import load_points_xyz, points_to_centers_zyx


def preprocess_loaded_click_aabb(
    data: np.ndarray, props: dict, json_path: str, pl, cm, dj,
    *, tile_step_size: float, cluster_margin_frac: float, mode: str,
    border_expand: bool, max_border_expand_extra: int,
):
    data, _seg, props, o_sp, t_sp, new_sh = crop_normalize_case(data, None, props, pl, cm)
    patch_size = tuple(cm.patch_size)
    padded_shape = tuple(max(patch_size[a], new_sh[a]) for a in range(3))
    pad_below = tuple((padded_shape[a] - new_sh[a]) // 2 for a in range(3))
    slicer_revert = (
        slice(0, data.shape[0]),
        slice(pad_below[0], pad_below[0] + new_sh[0]),
        slice(pad_below[1], pad_below[1] + new_sh[1]),
        slice(pad_below[2], pad_below[2] + new_sh[2]),
    )

    points_xyz = load_points_xyz(json_path)
    pad = torch.zeros((data.shape[0], *padded_shape), dtype=torch.float32)
    if points_xyz:
        pts_zyx = [(z, y, x) for x, y, z in points_xyz]
        pre = points_to_centers_zyx(pts_zyx, "voxel", props, new_sh, tuple(cm.spacing), pl.transpose_forward, voxel_coordinate_frame="full")
        pts_pad = map_points_zyx_unpadded_to_padded(pre, slicer_revert)
        stride = grid_stride(patch_size, tile_step_size)
        _, seed_slices = seed_slices_for_points(pts_pad, patch_size, padded_shape, cluster_margin_frac, mode)
        canvas_sl = canvas_bbox_for_seeds(seed_slices, stride, max_border_expand_extra, border_expand, padded_shape)
        real_lo = tuple(max(canvas_sl[a].start, slicer_revert[a + 1].start) for a in range(3))
        real_hi = tuple(min(canvas_sl[a].stop, slicer_revert[a + 1].stop) for a in range(3))
        if all(real_hi[a] > real_lo[a] for a in range(3)):
            out_lo = tuple(real_lo[a] - pad_below[a] for a in range(3))
            out_hi = tuple(real_hi[a] - pad_below[a] for a in range(3))
            dev = _rs.RESAMPLE_DEVICE
            assert dev is not None, (
                "RESAMPLE_DEVICE is unset -- preprocess_loaded_click_aabb needs the GPU resample "
                "device that cli/segtrack.py sets via set_resample_device(dev) before the case loop."
            )
            native = torch.as_tensor(np.ascontiguousarray(data), device=dev)
            aabb = interpolate_aabb(native, out_lo, out_hi, new_sh).to(torch.float32).cpu()
            pad[:, real_lo[0]:real_hi[0], real_lo[1]:real_hi[1], real_lo[2]:real_hi[2]] = aabb

    bl_points = None  # segtrack always calls this with is_longi=False in segment_native
    return pad, slicer_revert, props, points_xyz, bl_points
```

`_rs` is `nanounet.data.resampling` imported as `import nanounet.data.resampling as _rs` at the
top of this file — read `_rs.RESAMPLE_DEVICE` **at call time** (inside the function), not as a
`from ... import RESAMPLE_DEVICE` at module load time, since it is a module-global that
`set_resample_device` mutates *after* import (`cli/segtrack.py:126`, before the case loop
starts) — a top-level `from` import would freeze the pre-mutation value (`None`).

Check file length once written; if this pushes past 200 LOC, the click→pp-coordinate mapping
block (the `if points_xyz:` body) is the natural extraction into its own `_click_aabb(...)`
helper in the same file.

### B.5 — Wire into `nanounet/infer/segtrack.py`

In `run_case`, both branches currently call:
```python
pack = preprocess_loaded(fu_data, fu_props, str(case.fu_clicks), pl, cm, dj)
```
and
```python
pack_bl = preprocess_loaded(bl_data, bl_raw, str(case.bl_clicks), pl, cm, dj)
...
pack_fu, ... = fut.result()   # from _fu_pack(), which calls preprocess_loaded(d, p, ..., pl, cm, dj)
```

Replace every `preprocess_loaded(X, Y, Z, pl, cm, dj)` call in this file with
`preprocess_loaded_click_aabb(X, Y, Z, pl, cm, dj, tile_step_size=cfg.inference.tile_step_size,
cluster_margin_frac=seg_kw["cluster_margin_frac"], mode=seg_kw["inference_mode"],
border_expand=seg_kw["border_expand"], max_border_expand_extra=seg_kw.get("max_border_extra",
MAX_BORDER_EXTRA))`. `run_case` already receives `cfg` and `seg_kw` as parameters — no new
plumbing needed. Import `preprocess_loaded_click_aabb` from `nanounet.infer.click_resample`;
drop the now-unused `preprocess_loaded` import if nothing else in the file uses it.

This is FU/BL-independent single-timepoint preprocessing only (matches how `segment_native` is
always called with `is_longi=False` in this file) — do not touch `predict_io.preprocess_loaded`
itself or any longi call site.

### B.6 — Verify

Run the verification protocol. If any `cmp` fails: first re-run B.1's self-check standalone
(confirms the grid_sample/interpolate equivalence still holds in isolation); if that still
passes but the real pipeline diverges, the bug is almost certainly in the AABB↔padded-frame
bookkeeping (`pad_below`, `real_lo`/`real_hi` intersection, or the `canvas_sl` vs `slicer_revert`
alignment) — not the interpolation math. **Do not ship Step B non-identical.** If it cannot be
made bit-identical within reasonable effort, fall back to keeping `preprocess_loaded` unchanged
for segtrack and ship Steps A and C alone — they are independent and still a real win.

---

## Step C — prefetch case N+1 while case N runs

**Design constraint:** Step B's AABB resample runs on the GPU (`RESAMPLE_DEVICE`). Prefetching
it on a background thread while the *current* case's `segment_native` is mid-forward-pass would
put two concurrent workloads on the same CUDA device from different Python threads — real
contention risk, not free overlap. So prefetch only the **CPU-bound** half: disk read
(`load_ct`) + `crop_normalize_case` (transpose, crop-to-nonzero, normalize). Do the GPU AABB
resample synchronously, on the main thread, immediately before `segment_native`, using the
prefetched CPU-side result.

### C.1 — Split load from preprocess in `nanounet/infer/segtrack.py`

Extract the per-case disk-load step (already partially split via `load_ct`/`load_instance_zyx`
in `segtrack_case.py`) into one function covering both branches:

```python
def load_case_io(case: SegTrackCase):
    """CPU-only: disk reads. No GPU/preprocess work here -- safe to run in a background thread
    while the GPU is busy on a different case."""
    if case.bl_mask is not None:
        fu = load_ct(case.fu_img)
        bl_zyx, props_bl = load_instance_zyx(case.bl_mask)
        return {"fu": fu, "bl_zyx": bl_zyx, "props_bl": props_bl}
    return {"bl": load_ct(case.bl_img), "fu": load_ct(case.fu_img)}
```

Give `run_case` an optional `preloaded: dict | None = None` parameter; when provided, skip its
own `load_ct`/`load_instance_zyx` calls and use `preloaded["fu"]`/`preloaded["bl"]`/etc.
directly instead. When `None` (unchanged default), `run_case` loads synchronously exactly as it
does today — this keeps `run_case` usable standalone (tests, single-case CLI paths that don't go
through the folder loop) without requiring a caller to prefetch.

### C.2 — One-ahead prefetch in `nanounet/cli/segtrack.py`

In `main()`'s per-case loop, replace the plain `for i, case in enumerate(cases, 1):` body with a
one-ahead prefetch using `ThreadPoolExecutor(max_workers=1)` (same pattern already used inside
`run_case` for FU/BL concurrency — one more level of it, not a new pattern):

```python
from concurrent.futures import ThreadPoolExecutor
from nanounet.infer.segtrack import load_case_io

with ThreadPoolExecutor(max_workers=1) as io_pool:
    next_fut = io_pool.submit(load_case_io, cases[0])
    for i, case in enumerate(cases, 1):
        preloaded = next_fut.result()
        next_fut = io_pool.submit(load_case_io, cases[i]) if i < len(cases) else None
        ...
        r = run_case(case, cdir, net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dj=dj, dev=dev, matcher=matcher,
                     decode=args.decode, overwrite=args.overwrite, keep_pred=args.keep_pred,
                     track_ckpt=track_ckpt, thresh=args.thresh, device=d, seg_kw=seg_kw, on_step=on_step,
                     preloaded=preloaded)
```

`SystemExit` handling in the existing `try/except` around `run_case` stays as-is; on a skip, the
loop still needs `next_fut` already submitted for `i+1` before `continue` — the structure above
submits it right after consuming the current result, before the `try/run_case` call, so this
holds regardless of what `run_case` does or raises.

### C.3 — Verify

Run the verification protocol on the two reference cases *as a folder-mode pair* (not
single-mode — prefetch only exists in the folder loop), so both cases actually run through the
prefetch path back-to-back:

```bash
nanounet_segtrack \
  --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU \
  --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL \
  --patients-csv <(printf "patient\n39059724f7\n3994f23bfb\n") \
  -o /scratch/nielsrocholl/segtrack_prof/ref_after --keep-pred --overwrite
```

`cmp` against the same `ref_before` outputs as Steps A/B (prefetching changes *when* I/O
happens, never *what* gets computed, so output must still be byte-identical). This step is a
pure timing win, not a correctness-risk one — the main thing to actually check numerically is
that `next_fut.result()` for the *first* case still blocks correctly (no prefetch benefit
possible there, matches current cold-start behavior) and that the last case doesn't try to
submit past the end of `cases` (the `if i < len(cases) else None` guard above).

---

## Reporting

After all three steps pass their `cmp` gates, report a single before/after table using the
original ~80s/~84s baseline and the final per-case seconds from the CLI's `cprint` output,
plus one line on where the remaining time goes (SimpleITK read, `crop_to_nonzero`'s
`binary_fill_holes`, forward passes, `.mha` write) so the next round of work has a real number
to aim at instead of a guess. `crop_to_nonzero`'s full-FOV `binary_fill_holes` cost is explicitly
**not** addressed by this plan (Step B still needs the same global crop bbox for coordinate
consistency) — worth flagging as the next candidate, not fixing here.
