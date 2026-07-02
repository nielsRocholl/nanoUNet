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
