# Plan: longitudinal clustered inference (joint 2-channel preprocessing)

Status: implementation spec. A coding agent with no prior context implements this end to end.
**Every code change MUST obey `.claude/skills/nanochat-style/SKILL.md` (hard rule).** The relevant hard
rules are called out inline (R1 <200 LOC/file, R3 no dead code/one way, R5/E1 teaching errors, R11 no
bare print, R12 no silent fallback, R13 CLI procedural, R15 validate at startup, D2/D3/D4 docs).

---

## 1. Goal

Let a longitudinal (DWB two-stream) checkpoint run inference **exactly like the normal promptable
model**: `--inference-mode clustered` (pack all prompts into a covering patch set) plus
`--border-expand` (the large-lesion oversampling used at inference). Output = predicted **follow-up
(FU)** lesion segmentation.

The model takes **two registered scans** (FU + baseline). We assume the baseline is already warped
into the FU frame (`nanounet_register_longi`); no registration happens at inference.

## 2. Why the current longi inference is wrong, and the fix

**Training** (`nanounet/data/sampling_longi.py::build_patch_longi`, `nanounet/cli/longi_build.py`):
the longi case is a **single 2-channel image** — `_0000` = FU CT, `_0001` = warped-BL CT on an
identical grid. `nanounet_preprocess` runs **one nonzero crop across both channels**
(`nanounet/data/crop.py::_nonzero_mask` ORs all channels, one shared bbox), so FU and BL are
voxel-aligned. Each training patch takes **one bbox** and crops both channels at **identical
coordinates**; the BL stream encodes **all in-patch BL clicks** at their true positions.

**Current inference** (`nanounet/infer/predict_io.py::preprocess_case`,
`nanounet/infer/longi_row.py`): preprocesses FU and BL **separately** (`run_case([scan])` and
`run_case([bl_scan])`) → two independent nonzero crops → two grids that are **not voxel-aligned**.
It then fakes alignment with a co-location hack: one "BL partner" per cluster, the BL patch shifted
so a single anchor click matches, and only that one BL prompt encoded. This is a lossy approximation
of training and makes clustered/border-expand alignment fragile.

**Fix (decided: replace the old path):** preprocess **FU + BL jointly as one 2-channel case**
(`run_case([scan, bl_scan])`), reproducing the training grid. Then FU and BL share every patch bbox,
BL clicks map through the **same** `props`/`slicer_revert` as FU, and clustered mode + border-expand
work identically to the normal model — with only two additions per patch: crop channel 1 (BL) at the
same bbox and encode all in-patch BL clicks on the BL stream.

This **deletes** the co-location machinery (R3, no two ways / no dead code) and **changes**
`--baseline-points` from a parallel partner-list to a plain BL-click set (same JSON format as
`--points`).

### 2.1 Stream/channel layout (unchanged model contract)

`nanounet/model/dwb.py::LongiResEncUNet` splits the input row into two halves. Row is 6 channels:
`[FU_CT, FU_hm+, FU_hm-, BL_CT, BL_hm+, BL_hm-]`. Per stream `n_stream = n_img + 2` with
`n_img = 1` (one CT modality per timepoint; `build_net_longi` asserts
`determine_num_input_channels == 2` and builds the base net with `n_in_override=1`). The joint
preprocessed volume therefore has **2 channels**: `data[0]` = FU CT, `data[1]` = BL CT.

### 2.2 Null baseline (no baseline provided, or prompts disabled)

Training's `force_zero_prompt or not has_bl` path sets `bl_stream = fu_stream` (duplicate FU) →
`DWB(x_FU − x_FU) = 0` → identity → single-timepoint behaviour. Inference mirrors this exactly:
when there is no baseline image, or `--no-prompt-encode` is set, duplicate the FU half into the BL
half. A longi checkpoint therefore still runs single-stream when no baseline is given.

Note: "real BL_CT + empty BL heatmap" (a patch containing zero in-patch BL clicks while a baseline
*is* provided) is a training-seen input (any patch not covering a BL click), so it needs **no**
special handling — encode an empty BL heatmap over the real BL_CT crop.

---

## 3. Files changed (summary)

| File | Change | LOC gate |
|------|--------|----------|
| `nanounet/infer/longi_row.py` | **Full rewrite**: joint-volume row encoding; delete `bl_partner_for_cluster` | ~55 |
| `nanounet/infer/predict_case.py` | Edit: new longi params; joint BL point mapping; drop `_map_bl_pts_pad`/`bl_meta`/`pad_bl` | ~175 (must stay <200) |
| `nanounet/infer/roi_slices.py` | Edit: delete now-unused `colocated_spatial_slices` | ~105 |
| `nanounet/infer/predict_io.py` | Edit: joint 2-ch preprocess; geometry precheck; baseline resolver; delete `load_bl_points` | ~90 |
| `nanounet/cli/predict.py` | Edit: `--baseline-dir`; startup validation; `is_longi`/`bl_present` wiring | ~185 (must stay <200) |
| `docs/steps/predict.md` | Edit: args table + errors | <200 |
| `docs/steps/longi.md` | Edit: two-stream inference section | <200 |

**Verification gate (run after edits, R1):** `wc -l` each touched `.py`; every one must be `< 200`.
If `predict_case.py` or `predict.py` exceed 200, move the block flagged "extractable" (below) into
`predict_io.py`/`longi_row.py` rather than trimming logic.

Symbols confirmed safe to delete (only used by the files being rewritten — grep before deleting to
re-confirm):
`colocated_spatial_slices`, `bl_partner_for_cluster`, `_map_bl_pts_pad`, `load_bl_points`.
Symbols that MUST stay (still used): `local_prompt_points_for_patch`,
`map_points_zyx_unpadded_to_padded`, `cluster_prompts_patch_local`, `shift_spatial_slices`.

---

## 4. `nanounet/infer/longi_row.py` — FULL REWRITE

Replace the entire file with exactly this content:

```python
"""Two-stream inference row on a JOINT 2-channel volume (ch0 FU CT, ch1 BL CT sharing one
preprocessing crop, so FU/BL are voxel-aligned — same grid as training's build_patch_longi).
BL stream = same-bbox crop of ch1 + all in-patch BL clicks. Null baseline (no BL image, or prompts
disabled) duplicates the FU stream -> DWB(x_FU - x_FU)=0 -> identity (single-timepoint fallback)."""

from __future__ import annotations

import torch

from nanounet.infer.roi_slices import local_prompt_points_for_patch
from nanounet.prompt.cluster import cluster_prompts_patch_local
from nanounet.prompt.encoding import encode_points_to_heatmap_pair


def encode_inference_row(
    row: torch.Tensor,
    pad: torch.Tensor,
    sz: slice,
    sy: slice,
    sx: slice,
    n_img: int,
    cluster: list[tuple[int, int, int]],
    encode_prompt: bool,
    cfg,
    patch_size: tuple[int, int, int],
    dev: torch.device,
    *,
    is_longi: bool = False,
    bl_present: bool = False,
    bl_pts_pad: list[tuple[int, int, int]] | None = None,
) -> None:
    n_stream = n_img + 2
    row[:n_img] = pad[:n_img, sz, sy, sx]
    if not encode_prompt:
        row[n_img:n_stream].zero_()
    else:
        loc = cluster_prompts_patch_local(cluster, sz, sy, sx)
        if not loc:
            loc = local_prompt_points_for_patch(cluster[0], sz, sy, sx, patch_size)
        pr = encode_points_to_heatmap_pair(
            loc, [], patch_size, cfg.prompt.point_radius_vox, cfg.prompt.encoding,
            device=dev, intensity_scale=cfg.prompt.prompt_intensity_scale,
        )
        row[n_img:n_stream] = pr.float()
    if not is_longi:
        return
    # Null baseline: duplicate FU -> identity DWB (matches training force_zero_prompt / not has_bl).
    if not bl_present or not encode_prompt:
        row[n_stream:] = row[:n_stream]
        return
    # Real baseline: same bbox crops ch1 because the joint 2-ch crop keeps FU/BL voxel-aligned.
    row[n_stream : n_stream + n_img] = pad[n_img : 2 * n_img, sz, sy, sx]
    bl_local = cluster_prompts_patch_local(bl_pts_pad, sz, sy, sx) if bl_pts_pad else []
    bl_pr = encode_points_to_heatmap_pair(
        bl_local, [], patch_size, cfg.prompt.point_radius_vox, cfg.prompt.encoding,
        device=dev, intensity_scale=cfg.prompt.prompt_intensity_scale,
    )
    row[n_stream + n_img :] = bl_pr.float()
```

Notes for the implementer:
- `cluster_prompts_patch_local(bl_pts_pad, sz, sy, sx)` already filters points to the patch and
  returns patch-local `(z,y,x)`. Reuse it for BL exactly as FU uses it (do not write a new filter).
- Do NOT re-add `bl_partner_for_cluster`, `colocated_spatial_slices`, `map_points_zyx_unpadded_to_padded`
  imports here — they are gone from this file.

---

## 5. `nanounet/infer/predict_case.py` — EDITS

### 5.1 Imports
- Line 13: change
  `from nanounet.infer.longi_row import bl_partner_for_cluster, encode_inference_row`
  → `from nanounet.infer.longi_row import encode_inference_row`
- The `roi_slices` import block keeps `map_points_zyx_unpadded_to_padded` (FU still uses it) and drops
  nothing else.

### 5.2 Delete `_map_bl_pts_pad`
Delete the whole function (current lines 35–47) and its trailing blank line.

### 5.3 Signature of `predict_case_logits`
Remove these keyword params: `pad_bl`, `bl_points_xyz`, `bl_props`, `bl_slicer_revert`.
Add these (place after `merge: str = "max",`):

```python
    is_longi: bool = False,
    bl_present: bool = False,
    bl_points_xyz: list | None = None,
```

### 5.4 Channel/row sizing
Replace the current `n_img`/`n_stream`/`row_ch` derivation (current lines 84–86) with:

```python
    # longi net = 1 CT per stream (build_net_longi); joint volume has 2 channels when a baseline is
    # present, else 1 (null-baseline single-timepoint). Non-longi is always pad.shape[0] modalities.
    n_img = pad.shape[0] // 2 if (is_longi and bl_present) else pad.shape[0]
    n_stream = n_img + 2
    row_ch = 2 * n_stream if is_longi else n_stream
```

(Delete the old `pad_bl` reference on the current `row_ch` line.)

### 5.5 BL point mapping (replaces the old `bl_pts_pad` block, current lines 102–104)
After `pts_pad = map_points_zyx_unpadded_to_padded(pre, slicer_revert)`:

```python
    bl_pts_pad = None
    if is_longi and bl_present and bl_points_xyz:
        bl_zyx = [(z, y, x) for x, y, z in bl_points_xyz]
        bl_pre = points_to_centers_zyx(
            bl_zyx, "voxel", props, unpadded_shape, tuple(cm.spacing), pl.transpose_forward,
            voxel_coordinate_frame="full",
        )
        bl_pts_pad = map_points_zyx_unpadded_to_padded(bl_pre, slicer_revert)
```

BL points use the **same** `props`, `unpadded_shape`, and `slicer_revert` as FU — that is the whole
point of joint preprocessing (identical grid).

### 5.6 Seed-row build (current lines 121–128)
Replace with (drop `bl_meta`, drop `bl_partner_for_cluster`):

```python
    rows = []
    for cl, (sz, sy, sx) in zip(seeds_pts, seed_slices):
        row = torch.empty((row_ch, *patch_size), device=dev, dtype=torch.float32)
        encode_inference_row(
            row, pad, sz, sy, sx, n_img, cl, encode_prompt, cfg, patch_size, dev,
            is_longi=is_longi, bl_present=bl_present, bl_pts_pad=bl_pts_pad,
        )
        rows.append(row)
```

### 5.7 Border-expand row build (current lines ~164, 175–177)
- Delete the line `bp, anchor = bl_meta[i]`.
- Replace the `encode_inference_row(workon[0], ...)` call with:

```python
                encode_inference_row(
                    workon[0], pad, sze, sye, sxe, n_img, cl, encode_prompt, cfg, patch_size, dev,
                    is_longi=is_longi, bl_present=bl_present, bl_pts_pad=bl_pts_pad,
                )
```

Everything else in `predict_case.py` (gaussian, `accumulate`, merge max/average, TTA, export slice)
is unchanged. **Extractable if over 200 LOC:** move §5.5 into a helper
`points_to_padded(points_xyz, props, unpadded_shape, cm, pl, slicer_revert)` in
`nanounet/infer/roi_slices.py` and call it for both FU and BL.

---

## 6. `nanounet/infer/roi_slices.py` — EDIT

Delete `colocated_spatial_slices` (current lines 39–51). No other change. Confirm with
`grep -rn colocated_spatial_slices nanounet --include="*.py"` that no reference remains after the
`longi_row.py` rewrite.

---

## 7. `nanounet/infer/predict_io.py` — EDITS

### 7.1 Module top
Add imports at the top (nanochat: imports at top, not inside functions):

```python
import os

import numpy as np
import SimpleITK as sitk
```

Keep existing `import csv`, `import torch`, the `pad_nd_image` import, `run_case`, and
`load_points_xyz` imports.

### 7.2 Delete `load_bl_points`
Delete the whole `load_bl_points` function (current lines 14–31). BL points now use the same reader
and JSON format as FU points (`load_points_xyz`).

### 7.3 Geometry precheck (new small helper)
Add:

```python
def _assert_bl_geometry(scan: str, bl_scan: str) -> None:
    fu, bl = sitk.ReadImage(scan), sitk.ReadImage(bl_scan)
    if fu.GetSize() != bl.GetSize() or not np.allclose(fu.GetSpacing(), bl.GetSpacing()):
        raise ValueError(
            f"Baseline geometry does not match follow-up (joint longi preprocess needs one grid).\n"
            f"  FU {scan}: size={fu.GetSize()} spacing={fu.GetSpacing()}\n"
            f"  BL {bl_scan}: size={bl.GetSize()} spacing={bl.GetSpacing()}\n"
            f"Fix: register BL into the FU frame first with nanounet_register_longi "
            f"(see docs/steps/longi.md)."
        )
```

This satisfies E1 (name the problem, the expectation, the fix). Without it the raw
`read_images` error would be a bare `RuntimeError("shape mismatch ...")`.

### 7.4 Baseline resolver (new small helper; keeps `predict.py` lean)
Add:

```python
def baseline_resolver(baseline_image, baseline_points, baseline_dir, end):
    """cid -> (bl_scan|None, bl_json|None) and a bl_present bool, for CLI longi inference.

    Dataset mode: per-case siblings <baseline_dir>/<cid>{end} + <baseline_dir>/<cid>.json.
    Single mode: the two explicit --baseline-* paths (or (None, None) when not longi).
    """
    if baseline_dir is not None:
        def resolve(cid):
            return os.path.join(baseline_dir, cid + end), os.path.join(baseline_dir, cid + ".json")
        return resolve, True
    def resolve(_cid):
        return baseline_image, baseline_points
    return resolve, baseline_image is not None
```

### 7.5 Rewrite `preprocess_case`
Replace the current body with:

```python
def preprocess_case(scan, json_path, pl, cm, dj, bl_scan=None, bl_json=None):
    files = [scan]
    if bl_scan is not None:
        _assert_bl_geometry(scan, bl_scan)  # joint 2-ch crop keeps FU/BL voxel-aligned (design: §2)
        files = [scan, bl_scan]
    data, _seg, props = run_case(files, None, pl, cm, dj, verbose=False)
    data_t = torch.from_numpy(data).float()
    pad, slicer_revert = pad_nd_image(data_t, tuple(cm.patch_size), "constant", {"value": 0}, True, None)
    points = load_points_xyz(json_path)
    bl_points = load_points_xyz(bl_json) if bl_json else None
    return pad, slicer_revert, props, points, bl_points
```

**Return-tuple change:** now `(pad, slicer_revert, props, points, bl_points)` (was
`(..., points, bl_pack)`). Update the consumer in `predict.py` (§8.5). `bl_points` is `None` when no
baseline JSON, else a `list[(x,y,z)]`. When `bl_scan` is given `pad` has 2 channels; else 1.

---

## 8. `nanounet/cli/predict.py` — EDITS

### 8.1 Imports
Add: `from nanounet.model.dwb import LongiResEncUNet` and, from `predict_io`, add `baseline_resolver`
to the existing import (`patient_ids_from_csv, preprocess_case, baseline_resolver`).

### 8.2 New / changed argparse flags
- Change the help text of `--baseline-points`:
  `"BL click set JSON (single mode), same format as --points (native voxel x,y,z, FU-registered frame)"`
- Add (near the baseline flags):

```python
    ap.add_argument("--baseline-dir", default=None,
                    help="dataset mode: dir with per-case BL <cid>.nii.gz + <cid>.json (longi)")
```

Keep `--baseline-image`, `--baseline-points`, `--longi` as they are (only help text of
`--baseline-points` changes).

### 8.3 Startup validation (R15/E3 — all of this before any case runs)
Right after loading `net, lm` and computing `single_mode` / `cases` / `end`, add:

```python
    is_longi = isinstance(net, LongiResEncUNet)
    if single_mode and args.baseline_dir:
        raise SystemExit("--baseline-dir is for dataset mode; single mode uses --baseline-image/--baseline-points")
    if not single_mode and (args.baseline_image or args.baseline_points):
        raise SystemExit("dataset mode uses --baseline-dir (per-case BL); not --baseline-image/--baseline-points")
    resolve_bl, bl_present = baseline_resolver(args.baseline_image, args.baseline_points, args.baseline_dir, end)
    if bl_present and not is_longi:
        raise SystemExit("baseline given but checkpoint is not longi (no dwb.* keys). Drop --baseline-* or pass a longi ckpt.")
    if is_longi and not bl_present:
        cprint("[yellow]longi checkpoint without a baseline: running null-baseline (single-timepoint identity)[/yellow]")
    if args.baseline_dir:
        missing = []
        for cid, *_ in cases:
            bs, bj = resolve_bl(cid)
            if not os.path.isfile(bs):
                missing.append(bs)
            if not os.path.isfile(bj):
                missing.append(bj)
        if missing:
            raise FileNotFoundError(
                "Missing baseline files for longi dataset inference:\n  "
                + "\n  ".join(missing[:10])
                + ("\n  ..." if len(missing) > 10 else "")
                + f"\nExpected per FU case <cid>: {args.baseline_dir}/<cid>{end} and <cid>.json.\n"
                  "Fix: build them with nanounet_register_longi (see docs/steps/longi.md)."
            )
```

Keep the existing both-or-neither guard for the single-mode flags (current lines 70–73).

### 8.4 Header/config table
Add a row to the `config_table(...)` list so the run reports longi state:
`("longi", "on" if bl_present else ("null-baseline" if is_longi else "off"), "cli/ckpt"),`

### 8.5 Wire per-case baseline into preprocess + predict
- Change the `preprocess_case(...)` submit calls (single loop and threaded loop) to pass per-case BL:

  Single loop (current line 153):
  ```python
              bs, bj = resolve_bl(cid)
              consume(i, cid, out, preprocess_case(scan, jp, pl, cm, dj, bs, bj), bs is not None)
  ```
  Threaded loop (current line 163):
  ```python
          bs, bj = resolve_bl(cid)
          inflight.append((i, cid, out, bs is not None,
                           pool.submit(preprocess_case, scan, jp, pl, cm, dj, bs, bj)))
  ```
  and where the future is drained (current lines 165–166 and 167–169), unpack the extra
  `bl_case` element and pass it to `consume`.

- Delete the module-level `bl_scan = args.baseline_image` / `bl_json = args.baseline_points`
  lines (current 145–146).

- Change `consume` / `gpu_export` to take a per-case `bl_case: bool` and rebuild kwargs (drop the
  old `bl_pack` branch entirely):

  ```python
      def gpu_export(case_id, idx, out_trunc, pack, bl_case):
          t0 = time.perf_counter()
          pad_cpu, slicer_revert, props, points_xyz, bl_points = pack
          logits = predict_case_logits(
              net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dj=dj, dev=dev,
              pad=pad_cpu.to(dev), slicer_revert=slicer_revert, props=props, points_xyz=points_xyz,
              encode_prompt=not args.no_prompt_encode, use_tta=use_tta,
              border_expand=args.border_expand, max_border_expand_extra=args.max_border_extra,
              batch_size=args.batch_size, use_amp=not args.no_amp,
              cluster_margin_frac=args.cluster_margin_frac, mode=args.inference_mode, merge=args.merge,
              is_longi=is_longi, bl_present=bl_case, bl_points_xyz=bl_points,
          )
          export_prediction_from_logits(logits, props, cm, pl, dj, out_trunc)
          cprint(f"[bold green][{idx}/{n}] {case_id} ({time.perf_counter() - t0:.1f}s)[/bold green]")

      def consume(idx, case_id, out_trunc, pack, bl_case):
          gpu_export(case_id, idx, out_trunc, pack, bl_case)
  ```

`bl_case` is a per-case bool (in dataset mode all cases share the same value after §8.3 validation;
in single mode it equals `bl_present`). It exists so `predict_case_logits` knows whether `pad` has 2
channels (real baseline) or 1 (null baseline).

**Extractable if over 200 LOC:** move the §8.3 `--baseline-dir` existence check into a
`predict_io.check_baseline_files(cases_ids, resolve_bl, end)` helper.

### 8.6 GPU efficiency (G1/G4)
The joint 2-channel preprocess adds one extra `sitk.ReadImage` per case in the CPU prefetch worker
(`preprocess_case` runs in the `ThreadPoolExecutor`), not on the GPU thread — the data path is
unchanged in structure. No new CPU→GPU sync points are introduced (BL point mapping is pure
Python/host, done once per case; per-patch BL heatmaps are built on-device exactly like FU).
**Measure (G4):** run the §9 dataset smoke both single-stream and longi and report cases/s in the PR;
longi is expected to cost ~2× encoder FLOPs (two encoder passes) — that is the model, not a
data-path regression, and must be stated explicitly (G5).

---

## 9. Test plan (temporary — write, validate, delete; R16)

No permanent test files. Use the scratchpad dir. Validate three things:

1. **Alignment parity (unit):** in a throwaway script, joint-preprocess a real registered FU/BL pair
   with `preprocess_case(scan, json, pl, cm, dj, bl_scan, bl_json)`; assert `pad.shape[0] == 2` and
   `pad.shape[1:]` matches the single-stream FU `pad.shape[1:]` (same crop). Map one FU click and its
   BL partner through `points_to_centers_zyx(..., "voxel", props, ...)`; assert both land inside the
   volume and that a BL click co-located with an FU click maps within a few voxels of it.
2. **End-to-end longi clustered:** run
   `nanounet_predict -i fu.nii.gz -o /tmp/.../pred.nii.gz -m <longi_model> --ckpt <finetune.ckpt>
   --points fu.json --baseline-image bl.nii.gz --baseline-points bl.json
   --inference-mode clustered --merge max --border-expand`.
   Assert an output file is written and is non-empty foreground where expected.
3. **Null-baseline fallback:** same command **without** the two `--baseline-*` flags; assert it runs
   (yellow null-baseline notice) and produces output (single-timepoint identity path).
4. **Dataset mode:** a folder of FU scans + sibling `<cid>.json`, plus `--baseline-dir` with
   `<cid>.nii.gz`/`<cid>.json`; assert one output per case and that a missing BL sibling triggers the
   §8.3 `FileNotFoundError` with the fix line.

Delete all throwaway scripts/outputs afterward.

---

## 10. Docs (D2/D3/D4 — update in the SAME change)

### 10.1 `docs/steps/predict.md`
- In the arguments table, replace the `--baseline-points` row and add a `--baseline-dir` row:

  | `--baseline-points` | str | none | BL click set JSON (**single mode**), same format as `--points`; native voxel `(x,y,z)` in the FU-registered frame |
  | `--baseline-dir` | str | none | **Dataset mode** longi: dir with per-case BL `<cid>.nii.gz` + `<cid>.json` |

- In "Common errors", update/extend:
  - `--baseline-points requires --baseline-image` (single mode, keep).
  - `--baseline-dir is for dataset mode` / `dataset mode uses --baseline-dir` (mode mismatch).
  - `baseline given but checkpoint is not longi` → drop `--baseline-*` or use a longi ckpt.
  - `Baseline geometry does not match follow-up` → register BL→FU first.
  - `Missing baseline files for longi dataset inference` → build with `nanounet_register_longi`.

### 10.2 `docs/steps/longi.md` — rewrite the "Two-stream inference" section

Replace the current section body with the joint-preprocessing story:

- FU + baseline are preprocessed **jointly as one 2-channel case** at inference (same as training):
  one nonzero crop → FU/BL voxel-aligned → clustered mode and `--border-expand` (large-lesion
  oversampling) behave exactly like the normal promptable model, and output is the **FU**
  segmentation.
- `--baseline-image`: sibling BL `.nii.gz`, **already registered into the FU frame** (same
  size/spacing as FU; `nanounet_register_longi` produces this). It is NOT re-preprocessed
  separately.
- `--baseline-points`: a **plain BL click set** (same JSON as `--points`), native voxel `(x,y,z)` in
  the FU-registered frame — one entry per baseline lesion, **not** a partner list parallel to
  `--points`.
- Without both flags a longi checkpoint runs **null-baseline** (duplicated FU stream → identity DWB
  → single-timepoint behaviour).
- Dataset mode: use `--baseline-dir <dir>` with `<cid>.nii.gz` + `<cid>.json` per FU case.

Update the example command block to:

```bash
nanounet_predict \
  -i followup.nii.gz -o out/pred.nii.gz \
  -m "$MODEL_DIR" --ckpt finetune/best.ckpt \
  --points fu_points.json \
  --baseline-image baseline.nii.gz --baseline-points bl_clicks.json \
  --inference-mode clustered --merge max --border-expand
```

Keep all other sections of `longi.md` (register/build/clicks/finetune) unchanged.

---

## 11. Review checklist (run before done — SKILL §"Review checklist")

- [ ] `wc -l` every touched `.py` < 200 (R1). `predict_case.py` and `predict.py` are the risk files.
- [ ] `grep -rn` for `colocated_spatial_slices`, `bl_partner_for_cluster`, `_map_bl_pts_pad`,
      `load_bl_points`, `pad_bl`, `bl_slicer_revert`, `bl_props` in `nanounet/` returns **nothing**
      (R3 no dead code / one way).
- [ ] Every touched file keeps a *why* module docstring; no banner comments; comments explain *why*.
- [ ] Every new failure path (geometry mismatch, mode/flag mismatch, missing BL files, non-longi ckpt
      with baseline) names problem + expectation + fix command (E1); all validated at startup (E3/R15).
- [ ] No bare `print`; all output via `cprint`/`nano_header`/rich (R11). Null-baseline notice is a
      single calm yellow line (U7).
- [ ] Data-path: extra BL read happens in the prefetch worker, not the GPU thread (G1/G3); report
      before/after cases/s for single-stream vs longi in the PR (G4), and state the ~2× encoder cost
      is the model (G5), not a regression.
- [ ] `docs/steps/predict.md` + `docs/steps/longi.md` updated in this same change (D4), argument
      tables use the mandatory format (D3), commands are literal/runnable (D5).
```
