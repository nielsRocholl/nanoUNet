"""Di Veroli et al. (Medical Image Analysis 97, 2024) lesion matching, reimplemented for two scans: iterative greedy overlap.

Rule (paper Sec. 3.1-3.3, Fig. 2a): lesions are vertices, matchings are edges. Repeat `r` times: dilate the still-unmatched BL-frame and FU
lesion masks by `d` voxels, score every remaining (BL, FU) pair by overlap = max(|A and B|/|A|, |A and B|/|B|) on the dilated masks, add
an edge for every pair with overlap >= p (all pairs that clear p in the same iteration, which is how merges and splits arise), and
remove every lesion that received an edge. Classes come from degrees: a BL lesion with out-degree 0/1/>=2 is Disappeared/Persistent/
Split, an FU lesion with in-degree 0/1/>=2 is New/Persistent/Merged.

Non-obvious choices (the paper's Table A1 pseudo-code lives in an appendix we do not have): dilation is CUMULATIVE on still-unmatched
lesions (iteration k has grown them by k*d voxels), the only reading under which `r` matters; the structuring element is scipy's default
cross, so k iterations of a d-voxel dilation equal an L1 ball of radius k*d, which lets the overlap of every pair be tabulated ONCE for
all radii from two taxicab distance transforms (`overlap_table`) and the (d, p, r) search become table lookups. No lesion is filtered by
size (the paper drops lesions under 20 voxels); every annotated lesion is scored.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

PUBLISHED = {"d": 1, "p": 0.10, "r": 7}  # middle of the published r = 5 / 7 / 10 (lung / liver / brain)
GRID = {"d": (1, 2), "p": (0.05, 0.10, 0.20, 0.30), "r": (3, 5, 7, 10, 15)}
RHO_MAX = max(GRID["d"]) * max(GRID["r"])  # largest cumulative radius any grid point needs
MAX_CROP_VOXELS = 60_000_000


def _gap(a: np.ndarray, b: np.ndarray) -> int:
    """Lower bound of the L1 distance between two point sets from their bounding boxes."""
    lo_a, hi_a, lo_b, hi_b = a.min(0), a.max(0), b.min(0), b.max(0)
    return int(np.maximum(0, np.maximum(lo_b - hi_a, lo_a - hi_b)).sum())


def overlap_table(bl_pts: list[np.ndarray | None], fu_pts: list[np.ndarray | None]) -> np.ndarray:
    """(n_bl, n_fu, RHO_MAX + 1) overlap of the two masks each dilated by rho voxels (cross element), rho = 0..RHO_MAX. Points are int voxel
    coordinates in ONE frame (BL already moved into FU space); None (no mask or no position) gives overlap 0."""
    out = np.zeros((len(bl_pts), len(fu_pts), RHO_MAX + 1), dtype=np.float32)
    for i, a in enumerate(bl_pts):
        for j, b in enumerate(fu_pts):
            if a is None or b is None or _gap(a, b) > 2 * RHO_MAX:
                continue
            lo = np.minimum(a.min(0), b.min(0)) - RHO_MAX
            shape = tuple(int(s) for s in np.maximum(a.max(0), b.max(0)) + RHO_MAX + 1 - lo)
            assert np.prod(shape) <= MAX_CROP_VOXELS, f"crop {shape} too large for one lesion pair"
            dist = []
            for pts in (a, b):
                m = np.zeros(shape, dtype=bool)
                m[tuple((pts - lo).T)] = True
                dist.append(ndi.distance_transform_cdt(~m, metric="taxicab"))
            ha, hb = (np.cumsum(np.bincount(d.ravel(), minlength=RHO_MAX + 1)[: RHO_MAX + 1]) for d in dist)
            hab = np.cumsum(np.bincount(np.maximum(*dist).ravel(), minlength=RHO_MAX + 1)[: RHO_MAX + 1])
            out[i, j] = np.maximum(hab / np.maximum(ha, 1), hab / np.maximum(hb, 1))
    return out


def match(table: np.ndarray, d: int, p: float, r: int) -> list[tuple[int, int]]:
    """Edges (bl index, fu index) of the iterative greedy rule for parameters (d, p, r)."""
    assert d * r <= RHO_MAX, f"d*r = {d * r} exceeds the tabulated radius {RHO_MAX}"
    free_b, free_f = np.ones(table.shape[0], bool), np.ones(table.shape[1], bool)
    edges: list[tuple[int, int]] = []
    for k in range(1, r + 1):
        hit = (table[:, :, d * k] >= p) & free_b[:, None] & free_f[None, :]
        pairs = np.argwhere(hit)
        edges += [(int(i), int(j)) for i, j in pairs]
        free_b[pairs[:, 0]], free_f[pairs[:, 1]] = False, False
    return edges
