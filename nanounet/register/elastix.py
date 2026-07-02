"""Classical BL->FU registration via itk-elastix: rigid->affine->bspline multi-res, MI.

register() warps the moving image into the fixed frame; body_mask() builds a body-only mask so the
metric ignores air/table; resample_seg() re-applies a transform to a label map with nearest-neighbour
(FinalBSplineInterpolationOrder=0); resample_to() pulls an image onto a reference grid with a rigid
transform. MIN_COMP_VOX lives here so warp_case and refine share it.
"""

from __future__ import annotations

import cc3d
import itk
import numpy as np
import scipy.ndimage as ndi

MIN_COMP_VOX = 5  # drop connected-component speckle below this many voxels
BODY_HU = -300.0  # everything above this is body/tissue; below is air


def frame_z_overlap_mm(fixed: "itk.Image", moving: "itk.Image") -> float:
    org_f = np.array(fixed.GetOrigin(), dtype=float)
    org_m = np.array(moving.GetOrigin(), dtype=float)
    ext_f = org_f + np.array(fixed.GetSpacing()) * np.array(fixed.GetLargestPossibleRegion().GetSize())
    ext_m = org_m + np.array(moving.GetSpacing()) * np.array(moving.GetLargestPossibleRegion().GetSize())
    return float(max(0.0, min(ext_f[2], ext_m[2]) - max(org_f[2], org_m[2])))


def resample_to(img, tf, ref, *, default: float, interp=None):
    kw = dict(Input=img, Transform=tf, ReferenceImage=ref, UseReferenceImage=True)
    if interp is not None:
        kw["Interpolator"] = interp
    r = itk.ResampleImageFilter.New(**kw)
    r.SetDefaultPixelValue(default)
    r.Update()
    return r.GetOutput()


def default_params(*, geometric_init: bool = True) -> "itk.ParameterObject":
    po = itk.ParameterObject.New()
    # Disjoint DICOM frames need click pre-alignment first; otherwise centre bounding boxes so elastix
    # samples valid voxels instead of dying on zero overlap.
    rigid = po.GetDefaultParameterMap("rigid")
    if geometric_init:
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
    geometric_init: bool = True,
    threads: int | None = None,
    verbose: bool = False,
):
    # fixed=FU, moving=BL, both itk.Image[itk.F]. warped is moving resampled into the fixed frame.
    kw: dict = dict(parameter_object=default_params(geometric_init=geometric_init), log_to_console=verbose)
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
