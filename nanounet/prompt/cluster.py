"""Greedy point clustering for patch-sized 'infer all' seed batches."""

from __future__ import annotations

from typing import List, Tuple

ZYX = Tuple[int, int, int]


def _margin_vox(patch_size: Tuple[int, int, int], margin_frac: float) -> Tuple[int, int, int]:
    m = max(0.0, min(0.5, float(margin_frac)))
    return (
        int(round(m * patch_size[0])),
        int(round(m * patch_size[1])),
        int(round(m * patch_size[2])),
    )


def bbox_fits_in_patch(
    points: List[ZYX],
    patch_size: Tuple[int, int, int],
    margin: Tuple[int, int, int],
) -> bool:
    if not points:
        return True
    z = [p[0] for p in points]
    y = [p[1] for p in points]
    x = [p[2] for p in points]
    e0 = max(z) - min(z) + 1 + 2 * margin[0]
    e1 = max(y) - min(y) + 1 + 2 * margin[1]
    e2 = max(x) - min(x) + 1 + 2 * margin[2]
    return e0 <= patch_size[0] and e1 <= patch_size[1] and e2 <= patch_size[2]


def cluster_points_for_patch_size(
    points: List[ZYX],
    patch_size: Tuple[int, int, int],
    margin_frac: float = 0.1,
) -> List[List[ZYX]]:
    if not points:
        return []
    margin = _margin_vox(patch_size, margin_frac)
    sorted_pts = sorted(points, key=lambda t: (t[0], t[1], t[2]))
    clusters: List[List[ZYX]] = []
    for p in sorted_pts:
        for c in clusters:
            if bbox_fits_in_patch(c + [p], patch_size, margin):
                c.append(p)
                break
        else:
            clusters.append([p])
    return clusters


def spatial_slices_covering_points(
    points: List[ZYX],
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    """Patch placement that includes every point (bbox-centered, volume-clamped).

    Centroid-centered tiles can drop cluster members when the bbox fits in patch_size
    but the mean sits away from an extremal lesion (common on large volumes).
    """
    if not points:
        raise ValueError("spatial_slices_covering_points requires at least one point")
    out: List[slice] = []
    for axis in range(3):
        ps, dim = patch_size[axis], padded_shape[axis]
        lo = min(p[axis] for p in points)
        hi = max(p[axis] for p in points)
        s_lo = hi - ps + 1
        s_hi = lo
        center = (lo + hi) // 2
        s = center - ps // 2
        if s_lo <= s_hi:
            s = max(s_lo, min(s, s_hi))
        s = max(0, min(s, dim - ps))
        out.append(slice(s, s + ps))
    return tuple(out)


def cluster_prompts_patch_local(
    pts_pad: List[ZYX],
    sz: slice,
    sy: slice,
    sx: slice,
) -> List[ZYX]:
    out: List[ZYX] = []
    for pz, py, px in pts_pad:
        if sz.start <= pz < sz.stop and sy.start <= py < sy.stop and sx.start <= px < sx.stop:
            out.append((pz - sz.start, py - sy.start, px - sx.start))
    return out
