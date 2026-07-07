"""Warp one BL/FU pair into the FU frame: image, mask, CSV-keyed clicks.

Registers FU<-BL, warps the BL image and BL mask, and derives BL clicks in the FU frame from a
ball-stamped lesion-id map (train: meta CSV lesion_id; val: sequential json order). Two backends:
`elastix` (classical, default) runs per-lesion VOI `refine` afterwards; `unigradicon` (deep learning)
uses its own native instance optimization (`io_iterations`) as the refinement instead. Standalone: no
dependency on nanounet.data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import itk
import numpy as np

from nanounet.register.elastix import body_mask, frame_z_overlap_mm
from nanounet.register.elastix import warp_pair as elastix_warp_pair
from nanounet.register.landmarks import (
    correspondence,
    landmark_align,
    read_pairs,
    robust_rigid,
    stamp_ids,
)
from nanounet.register.refine import refine_clicks
from nanounet.register.unigradicon import warp_pair as unigradicon_warp_pair

StepFn = Callable[[str], None]


@dataclass
class WarpResult:
    warped_img: "itk.Image"
    warped_seg: np.ndarray
    bl_clicks: list[tuple[int, list]]       # (lesion_id, [x,y,z]) in FU frame
    fu_out: list[tuple[int, list]] | None   # train: cog_fu by lesion_id; val: None (copy FU json)
    fu_ref: "itk.Image"
    bl_seg_vox: int


def _load_pts(path: str) -> list:
    with open(path) as f:
        return [p["point"] for p in json.load(f)["points"]]


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
    backend: str = "elastix",
    io_iterations: int = 0,
) -> WarpResult:
    assert backend in ("elastix", "unigradicon"), backend

    def step(name: str) -> None:
        if on_step is not None:
            on_step(name)

    fu_path = os.path.join(data_root, "inputsTrFU", f"{pid}_{idx}.nii.gz")
    bl_path = os.path.join(data_root, "inputsTrBL", f"{pid}_{idx}.nii.gz")
    bl_seg_path = os.path.join(data_root, "targetsTrBL", f"{pid}_{idx}.nii.gz")
    fu_json = fu_path.replace(".nii.gz", ".json")
    bl_json = bl_path.replace(".nii.gz", ".json")
    meta_csv = os.path.join(data_root, "meta", f"{pid}.csv")
    assert os.path.isfile(fu_path), fu_path
    assert os.path.isfile(bl_path), bl_path
    assert os.path.isfile(bl_seg_path), bl_seg_path

    step("load")
    fu = itk.imread(fu_path, itk.F)
    bl = itk.imread(bl_path, itk.F)
    bl_seg = itk.imread(bl_seg_path, itk.F)
    bl_seg_vox = int((itk.array_from_image(bl_seg) > 0.5).sum())

    has_csv = os.path.isfile(meta_csv)
    pairs = read_pairs(meta_csv) if has_csv else None
    fu_pts = _load_pts(fu_json) if os.path.isfile(fu_json) else None
    bl_pts = _load_pts(bl_json) if os.path.isfile(bl_json) else None

    if has_csv:
        bl_src = [(lid, cb) for lid, (cb, cf) in pairs.items() if cb is not None]
        fu_out = [(lid, cf.tolist()) for lid, (cb, cf) in pairs.items() if cf is not None]
    else:
        assert bl_pts is not None, bl_json
        bl_src = [(i + 1, p) for i, p in enumerate(bl_pts)]
        fu_out = None

    assert bl_src, "no BL lesion clicks to stamp"
    bl_ids = stamp_ids(bl_seg, [p for _, p in bl_src], [lid for lid, _ in bl_src])

    geometric_init = True
    if frame_z_overlap_mm(fu, bl) <= 0:
        F, B = correspondence(fu, bl, pairs, fu_pts, bl_pts)
        tf = robust_rigid(F, B)
        bl, bl_seg, bl_ids = landmark_align(fu, bl, bl_seg, bl_ids, tf)
        geometric_init = False

    step("register")
    if backend == "elastix":
        fu_mask = body_mask(fu) if body_mask_metric else None
        bl_mask = body_mask(bl) if body_mask_metric else None
        warped_img, warped_seg_itk, warped_ids_itk = elastix_warp_pair(
            fu, bl, bl_seg, bl_ids,
            fixed_mask=fu_mask,
            moving_mask=bl_mask,
            geometric_init=geometric_init,
            threads=threads,
            verbose=verbose,
        )
    else:
        warped_img, warped_seg_itk, warped_ids_itk = unigradicon_warp_pair(
            fu, bl, bl_seg, bl_ids, io_iterations=io_iterations, verbose=verbose,
        )

    step("clicks")
    seg = (itk.array_from_image(warped_seg_itk) > 0.5).astype(np.uint8)
    ids = np.rint(itk.array_from_image(warped_ids_itk)).astype(np.int32)
    bl_clicks: list[tuple[int, list]] = []
    for lbl, _ in bl_src:
        m = ids == lbl
        if not m.any():
            continue
        zz, yy, xx = np.nonzero(m)
        bl_clicks.append((lbl, [float(xx.mean()), float(yy.mean()), float(zz.mean())]))
    assert bl_clicks, "all lesions vanished after registration"

    if refine and backend == "elastix":
        step("refine")
        xyz = [c for _, c in bl_clicks]
        xyz = refine_clicks(fu, warped_img, seg, xyz, verbose=verbose)
        bl_clicks = [(lbl, c) for (lbl, _), c in zip(bl_clicks, xyz)]

    return WarpResult(warped_img, seg, bl_clicks, fu_out, fu, bl_seg_vox)
