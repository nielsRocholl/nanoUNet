"""Lesion-click correspondence, ball-stamp identity maps, and robust disjoint-frame rigid pre-align.

read_pairs() loads meta-CSV lesion_id keys; correspondence() builds FU<-BL physical point pairs;
robust_rigid() fits a trimmed landmark rigid transform; stamp_ids() paints label balls that survive
off-mask clicks; landmark_align() resamples BL + mask + id map onto the FU grid.
"""

from __future__ import annotations

import csv

import itk
import numpy as np

from nanounet.register.elastix import resample_to

MIN_LANDMARKS = 3
MIN_SPREAD_MM = 50.0
BALL_MM = 5.0
TRIM_FRAC = 0.2


def read_pairs(meta_csv: str) -> dict[int, tuple[np.ndarray | None, np.ndarray | None]]:
    out: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
    with open(meta_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lid = int(row["lesion_id"])
            bl = row["cog_bl"].strip()
            fu = row["cog_fu"].strip()
            cb = np.array([float(x) for x in bl.split()], dtype=float) if bl else None
            cf = np.array([float(x) for x in fu.split()], dtype=float) if fu else None
            if cb is not None:
                assert cb.shape == (3,)
            if cf is not None:
                assert cf.shape == (3,)
            out[lid] = (cb, cf)
    return out


def _phys(img: "itk.Image", pt: list | np.ndarray) -> np.ndarray:
    return np.array(img.TransformIndexToPhysicalPoint([int(round(c)) for c in pt]), dtype=float)


def correspondence(
    fu: "itk.Image",
    bl: "itk.Image",
    pairs: dict[int, tuple[np.ndarray | None, np.ndarray | None]] | None,
    fu_pts: list | None,
    bl_pts: list | None,
) -> tuple[np.ndarray, np.ndarray]:
    if pairs is not None:
        F, B = [], []
        for cb, cf in pairs.values():
            if cb is None or cf is None:
                continue
            F.append(_phys(fu, cf))
            B.append(_phys(bl, cb))
        assert len(F) > 0, "no CSV lesion pairs with both cog_bl and cog_fu"
        return np.array(F), np.array(B)
    assert fu_pts is not None and bl_pts is not None
    assert len(fu_pts) == len(bl_pts) and len(fu_pts) > 0
    F = np.array([_phys(fu, p) for p in fu_pts])
    B = np.array([_phys(bl, p) for p in bl_pts])
    return F, B


def _landmarks(pts: np.ndarray):
    v = itk.vector[itk.Point[itk.D, 3]]()
    for p in pts:
        v.push_back(itk.Point[itk.D, 3]([float(p[0]), float(p[1]), float(p[2])]))
    return v


def _fit_rigid(F: np.ndarray, B: np.ndarray):
    tf = itk.VersorRigid3DTransform[itk.D].New()
    init = itk.LandmarkBasedTransformInitializer[itk.Transform[itk.D, 3, 3]].New()
    init.SetFixedLandmarks(_landmarks(F))
    init.SetMovingLandmarks(_landmarks(B))
    init.SetTransform(tf)
    init.InitializeTransform()
    return tf


def _residuals(tf, F: np.ndarray, B: np.ndarray) -> np.ndarray:
    out = np.empty(len(F))
    for i, (f, b) in enumerate(zip(F, B)):
        p = tf.TransformPoint(itk.Point[itk.D, 3]([float(f[0]), float(f[1]), float(f[2])]))
        out[i] = np.linalg.norm(np.array([p[0], p[1], p[2]]) - b)
    return out


def robust_rigid(F: np.ndarray, B: np.ndarray):
    spread = np.linalg.svd(F - F.mean(axis=0), compute_uv=False)
    if len(F) < MIN_LANDMARKS or spread[1] < MIN_SPREAD_MM:
        tf = itk.VersorRigid3DTransform[itk.D].New()
        tf.SetTranslation([float(x) for x in np.median(B - F, axis=0)])
        return tf
    tf = _fit_rigid(F, B)
    n_drop = int(len(F) * TRIM_FRAC)
    if n_drop > 0 and len(F) - n_drop >= MIN_LANDMARKS:
        keep = np.argsort(_residuals(tf, F, B))[: len(F) - n_drop]
        tf = _fit_rigid(F[keep], B[keep])
    return tf


def stamp_ids(
    seg: "itk.Image",
    pts_vox: list,
    labels: list[int],
    *,
    r_mm: float = BALL_MM,
) -> "itk.Image":
    sp = np.array(seg.GetSpacing(), dtype=float)
    shape = itk.array_from_image(seg).shape  # z, y, x
    nz, ny, nx = shape
    ids = np.zeros(shape, dtype=np.float32)
    dist = np.full(shape, np.inf, dtype=np.float32)
    seen_vox: set[tuple[int, int, int]] = set()
    rz = max(1, int(round(r_mm / sp[2])))
    ry = max(1, int(round(r_mm / sp[1])))
    rx = max(1, int(round(r_mm / sp[0])))
    r2 = r_mm * r_mm
    for pt, lbl in zip(pts_vox, labels):
        x, y, z = [int(round(c)) for c in pt]
        xi = min(max(x, 0), nx - 1)
        yi = min(max(y, 0), ny - 1)
        zi = min(max(z, 0), nz - 1)
        assert (zi, yi, xi) not in seen_vox, f"duplicate click voxel at ({x},{y},{z})"
        seen_vox.add((zi, yi, xi))
        z0, z1 = max(0, zi - rz), min(nz, zi + rz + 1)
        y0, y1 = max(0, yi - ry), min(ny, yi + ry + 1)
        x0, x1 = max(0, xi - rx), min(nx, xi + rx + 1)
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        d = ((xx - x) * sp[0]) ** 2 + ((yy - y) * sp[1]) ** 2 + ((zz - z) * sp[2]) ** 2
        m = d <= r2
        sub = dist[z0:z1, y0:y1, x0:x1]
        closer = m & (d < sub)
        sub[closer] = d[closer]
        ids[z0:z1, y0:y1, x0:x1][closer] = float(lbl)
    out = itk.image_from_array(ids)
    out.CopyInformation(seg)
    return out


def landmark_align(
    fu: "itk.Image",
    bl: "itk.Image",
    bl_seg: "itk.Image",
    bl_ids: "itk.Image",
    tf,
):
    bl_al = resample_to(bl, tf, fu, default=-1000.0)
    nn = itk.NearestNeighborInterpolateImageFunction.New(bl_seg)
    seg_al = resample_to(bl_seg, tf, fu, default=0.0, interp=nn)
    nn_ids = itk.NearestNeighborInterpolateImageFunction.New(bl_ids)
    ids_al = resample_to(bl_ids, tf, fu, default=0.0, interp=nn_ids)
    return bl_al, seg_al, ids_al
