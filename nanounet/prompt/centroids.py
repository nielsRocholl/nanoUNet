"""cc3d centroids/bboxes JSON next to *_seg.b2nd; propagation-offset for simulated longitudinal COG error.

seed_zyx/volume_vox exist because the plain centroid falls outside its own lesion ~12% of the time
(irregular/concave shapes): training needs a point guaranteed inside the component (the argmax-EDT
voxel) to identify which lesion a click refers to, and the voxel count to pick a size bin in the
registration-error offset table.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from typing import Any, Dict, List, Tuple

import blosc2
import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt


def centroids_from_seg(seg: np.ndarray) -> List[Tuple[int, int, int]]:
    s = np.asarray(seg)
    if s.ndim == 4:
        s = s[0]
    s = np.maximum(s, 0).astype(np.uint8)
    u = np.unique(s)
    u = u[u > 0]
    if len(u) == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    if len(u) == 1 and u[0] == 1:
        lab = cc3d.connected_components((s > 0).astype(np.uint8))
        stats = cc3d.statistics(lab, no_slice_conversion=True)
        for j in range(1, int(lab.max()) + 1):
            c = stats["centroids"][j]
            out.append((int(round(c[0])), int(round(c[1])), int(round(c[2]))))
    else:
        stats = cc3d.statistics(s, no_slice_conversion=True)
        for idx, c in enumerate(stats["centroids"][1:], start=1):
            if idx in u:
                out.append((int(round(c[0])), int(round(c[1])), int(round(c[2]))))
    return out


def filter_centroids_in_patch(
    centroids: List[Tuple[int, int, int]],
    patch_slices: Tuple[slice, slice, slice],
) -> List[Tuple[int, int, int]]:
    slz, sly, slx = patch_slices
    o: List[Tuple[int, int, int]] = []
    for z, y, x in centroids:
        if slz.start <= z < slz.stop and sly.start <= y < sly.stop and slx.start <= x < slx.stop:
            o.append((z - slz.start, y - sly.start, x - slx.start))
    return o


_EMPTY_CENTROIDS: Dict[str, Any] = {
    "centroids_zyx": [],
    "bboxes_zyx": [],
    "seed_zyx": [],
    "volume_vox": [],
}


def _one_case(seg_path: str) -> Dict[str, Any]:
    seg = np.asarray(blosc2.open(seg_path, mode="r")[:])
    if seg.ndim == 4:
        seg = seg[0]
    seg = np.maximum(seg, 0).astype(np.uint8)
    if not np.any(seg > 0):
        return dict(_EMPTY_CENTROIDS)
    lab = cc3d.connected_components((seg > 0).astype(np.uint8))
    stats = cc3d.statistics(lab, no_slice_conversion=False)
    centroids: List[List[int]] = []
    bboxes: List[List[int]] = []
    seeds: List[List[int]] = []
    volumes: List[int] = []
    for i in range(1, int(lab.max()) + 1):
        sl = stats["bounding_boxes"][i]
        sub_lab = lab[sl]
        mask = sub_lab == i
        if not mask.any():
            continue
        off = np.array([s.start for s in sl])
        cz, cy, cx = stats["centroids"][i]
        centroids.append([int(round(cz)), int(round(cy)), int(round(cx))])
        bboxes.append(
            [sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1, sl[2].start, sl[2].stop - 1]
        )
        edt = distance_transform_edt(mask)
        seed_local = np.array(np.unravel_index(np.argmax(edt), edt.shape))
        seeds.append((seed_local + off).astype(int).tolist())
        volumes.append(int(mask.sum()))
    return {
        "centroids_zyx": centroids,
        "bboxes_zyx": bboxes,
        "seed_zyx": seeds,
        "volume_vox": volumes,
    }


def _write_centroids_for_case(folder: str, case_id: str) -> None:
    suf = "_seg.b2nd"
    out = os.path.join(folder, f"{case_id}_centroids.json")
    d = _one_case(os.path.join(folder, case_id + suf))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f)


def precompute_folder(folder: str, num_processes: int, resume: bool) -> None:
    suf = "_seg.b2nd"
    ids = sorted(i[: -len(suf)] for i in os.listdir(folder) if i.endswith(suf))
    if resume:
        ids = [c for c in ids if not os.path.isfile(os.path.join(folder, f"{c}_centroids.json"))]
    if not ids:
        return
    if num_processes <= 1:
        for case_id in ids:
            _write_centroids_for_case(folder, case_id)
        return
    args = [(folder, c) for c in ids]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        pool.starmap(_write_centroids_for_case, args)


def apply_propagation_offset(
    centroid_zyx: Tuple[int, int, int],
    sigma_per_axis: Tuple[float, float, float],
    max_vox: float,
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    """Gaussian-jitter a GLOBAL centroid (mode='gaussian'). No patch clamping: a displaced click
    that lands outside the patch is simply not rendered (see filter_centroids_in_patch), matching
    how a real registration-propagated click behaves at inference."""
    cz, cy, cx = centroid_zyx
    dz, dy, dx = rng.normal(0, sigma_per_axis)
    mag = float(np.sqrt(dz * dz + dy * dy + dx * dx))
    if mag > max_vox and mag > 1e-8:
        s = max_vox / mag
        dz, dy, dx = dz * s, dy * s, dx * s
    return (int(round(cz + dz)), int(round(cy + dy)), int(round(cx + dx)))
