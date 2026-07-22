"""Patch logits → native scanner-space seg array + optional gzip NIfTI bytes."""

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


def _convert_preprocessed_seg_to_native(seg_pp: np.ndarray, pl, cm, props: dict) -> np.ndarray:
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    sp_t = [props["spacing"][i] for i in pl.transpose_forward]
    sh = props["shape_after_cropping_and_before_resampling"]
    cur_sp = cm.spacing if len(cm.spacing) == len(sh) else [sp_t[0], *cm.spacing]
    tgt_sp = [props["spacing"][i] for i in pl.transpose_forward]
    seg_rs = cm.resampling_fn_seg(seg_pp[np.newaxis, ...], sh, cur_sp, tgt_sp)[0]
    full = np.zeros(props["shape_before_cropping"], dtype=seg_rs.dtype)
    full = insert_crop_into_image(full, seg_rs, props["bbox_used_for_cropping"])
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
    """Argmax patch → scatter into unpadded pp volume → resample to native array."""
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
    patch_np = patch_seg.to(torch.uint8).cpu().numpy()
    seg_pp = np.zeros(_unpadded_shape(slicer_revert), dtype=np.uint8)
    seg_pp[uz, uy, ux] = patch_np[pz, py, px]
    return _convert_preprocessed_seg_to_native(seg_pp, pl, cm, props)


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
