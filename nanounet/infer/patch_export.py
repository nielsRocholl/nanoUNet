"""Patch / tile segs → native scanner-space array + optional gzip NIfTI bytes.

Native paste is per-tile (nnInteractive-style), not a full-volume logit resample.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch


def _axis_overlap(patch_sl: slice, rev_sl: slice) -> tuple[slice, slice] | None:
    i0 = max(patch_sl.start, rev_sl.start)
    i1 = min(patch_sl.stop, rev_sl.stop)
    if i1 <= i0:
        return None
    return slice(i0 - rev_sl.start, i1 - rev_sl.start), slice(i0 - patch_sl.start, i1 - patch_sl.start)


def patch_unpadded_overlap(
    sz: slice, sy: slice, sx: slice, slicer_revert: tuple
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    """(unpadded slices, patch-local slices) overlapping slicer_revert crop."""
    rz, ry, rx = slicer_revert[1], slicer_revert[2], slicer_revert[3]
    z = _axis_overlap(sz, rz)
    y = _axis_overlap(sy, ry)
    x = _axis_overlap(sx, rx)
    if z is None or y is None or x is None:
        return None
    return (z[0], y[0], x[0]), (z[1], y[1], x[1])


def _unpadded_shape(slicer_revert: tuple) -> tuple[int, int, int]:
    return (
        slicer_revert[1].stop - slicer_revert[1].start,
        slicer_revert[2].stop - slicer_revert[2].start,
        slicer_revert[3].stop - slicer_revert[3].start,
    )


def _map_ix(i: int, p: int, n: int) -> int:
    return int(round(i * n / p)) if p else 0


def _paste_max(full: np.ndarray, crop: np.ndarray, bbox: list[list[int]]) -> None:
    img_sl, cr_sl = [], []
    for i, (mn, mx) in enumerate(bbox):
        t0, t1 = max(0, mn), min(mx, full.shape[i])
        c0 = max(0, -mn)
        if t1 <= t0:
            return
        img_sl.append(slice(t0, t1))
        cr_sl.append(slice(c0, c0 + (t1 - t0)))
    sub = full[tuple(img_sl)]
    src = crop[tuple(cr_sl)].astype(full.dtype, copy=False)
    np.maximum(sub, src, out=sub)


def tiles_to_native_seg(
    crops: list[tuple[np.ndarray, tuple[slice, slice, slice]]],
    pl,
    cm,
    props: dict,
    pp_shape: tuple[int, int, int],
) -> np.ndarray:
    """Nearest-resample each plan-space crop into native scanner zeros (per-tile paste)."""
    sh = props["shape_after_cropping_and_before_resampling"]
    sp_t = [props["spacing"][i] for i in pl.transpose_forward]
    cur_sp = cm.spacing if len(cm.spacing) == len(sh) else [sp_t[0], *cm.spacing]
    tgt_sp = [props["spacing"][i] for i in pl.transpose_forward]
    P, N = np.array(pp_shape, dtype=np.int64), np.array(sh, dtype=np.int64)
    crop_bb = props["bbox_used_for_cropping"]
    full = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
    for crop, sl in crops:
        if crop.size == 0 or not np.any(crop):
            continue
        lo = [sl[d].start for d in range(3)]
        hi = [sl[d].stop for d in range(3)]
        nlo = [_map_ix(lo[d], int(P[d]), int(N[d])) for d in range(3)]
        nhi = [_map_ix(hi[d], int(P[d]), int(N[d])) for d in range(3)]
        ns = tuple(max(1, min(int(N[d]), nhi[d]) - max(0, nlo[d])) for d in range(3))
        nlo_c = [max(0, nlo[d]) for d in range(3)]
        nhi_c = [nlo_c[d] + ns[d] for d in range(3)]
        rs = np.asarray(cm.resampling_fn_seg(crop[None].astype(np.float32), ns, cur_sp, tgt_sp))[0]
        bb = [[int(crop_bb[d][0] + nlo_c[d]), int(crop_bb[d][0] + nhi_c[d])] for d in range(3)]
        _paste_max(full, rs, bb)
    return full.transpose(tuple(pl.transpose_backward))


def patch_logits_to_native_seg(
    patch_logits: torch.Tensor,
    sz: slice,
    sy: slice,
    sx: slice,
    *,
    slicer_revert: tuple,
    props: dict,
    pl,
    cm,
) -> np.ndarray:
    """Argmax patch → per-tile native paste (no full pp zeros volume)."""
    ov = patch_unpadded_overlap(sz, sy, sx, slicer_revert)
    if ov is None:
        raise ValueError(
            "patch ROI does not overlap preprocessed crop (slicer_revert).\n"
            "Expected centered click inside the padded case volume.\n"
            "Fix: re-prepare the interactive session, then click again."
        )
    (uz, uy, ux), (pz, py, px) = ov
    m = patch_logits.detach()
    if m.ndim == 5 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 4:
        patch_seg = m.argmax(dim=0)
    elif m.ndim == 3:
        patch_seg = m.argmax(dim=0)
    else:
        raise ValueError(f"unexpected patch_logits shape {tuple(patch_logits.shape)}")
    crop = patch_seg.to(torch.uint8).cpu().numpy()[pz, py, px]
    return tiles_to_native_seg([(crop, (uz, uy, ux))], pl, cm, props, _unpadded_shape(slicer_revert))


def native_seg_to_nifti_bytes(seg: np.ndarray, props: dict) -> bytes:
    """Write gzip NIfTI in-memory; return file bytes."""
    import SimpleITK as sitk

    if "sitk_stuff" not in props:
        raise KeyError(
            "props missing 'sitk_stuff' (spacing/origin/direction).\n"
            "Expected output of nanoUNet run_case preprocessing.\n"
            "Fix: re-run prepare on a valid NIfTI/MHA input."
        )
    st = props["sitk_stuff"]
    dtype = np.uint8 if int(seg.max()) < 255 else np.int16
    itk = sitk.GetImageFromArray(seg.astype(dtype, copy=False))
    itk.SetSpacing(st["spacing"])
    itk.SetOrigin(st["origin"])
    itk.SetDirection(st["direction"])
    fd, path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    try:
        sitk.WriteImage(itk, path, useCompression=True)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
