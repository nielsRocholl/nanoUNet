# Radiom in-process embed API

Library hooks on `predict_case_logits` for desktop viewers that map clicks in world mm
before calling inference (Radiom keeps slice-click geometry authoritative).

## `points_zyx_unpadded`

Preprocessed-volume indices `(z, y, x)` in **unpadded** crop space — same frame as
`points_to_centers_zyx(..., "world"|"voxel", ...)` output before `map_points_zyx_unpadded_to_padded`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points_zyx_unpadded` | `list[tuple[int,int,int]] \| None` | `None` | When set, skip `points_xyz` JSON mapping; mutual exclusive with non-empty `points_xyz` |

## `on_forward`

Optional `(done, total)` callback after each completed patch forward (seed or border-expand).
Not called inside TTA mirror passes — once per stacked batch output patch.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `on_forward` | `Callable[[int,int],None] \| None` | `None` | Progress hook for embedders (e.g. Radiom `/api/nanounet/infer-progress`) |

Remote CLI / `nanounet_predict` unchanged: omit both parameters.

## Interactive patch API

For warm-session click inference (Radiom remote `/session/click`):

| Function | Module | Description |
|----------|--------|-------------|
| `predict_patch_logits` | `nanounet.infer.predict_patch` | Single centered patch; `(logits CPU, padded slices)`. Prefer `use_tta=False`. |
| `patch_logits_to_native_seg` | `nanounet.infer.patch_export` | Patch logits → native seg `np.ndarray` |
| `native_seg_to_nifti_bytes` | `nanounet.infer.patch_export` | Native seg → gzip `.nii.gz` bytes |

Map viewer clicks to padded `(z,y,x)` via `resolve_pts_pad` / `points_to_centers_zyx` (same as batch), then call `predict_patch_logits`.
