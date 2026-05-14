"""BFS neighbors from hull-shell contact of foreground at patch faces (cc3d shell labels)."""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import cc3d
import numpy as np
import torch
from nanounet.plan.labels import Labels


def plan_border_expansion_centers_from_fg(
    fg: np.ndarray,
    sz: slice,
    sy: slice,
    sx: slice,
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
    seed_key: Tuple[int, int, int, int, int, int],
    max_centers: int,
    quant_vox: Optional[int] = None,
    skip_keys: Optional[Set[Tuple[int, int, int, int, int, int]]] = None,
) -> List[Tuple[int, int, int]]:
    if max_centers <= 0 or not fg.any():
        return []
    dz, dy, dx = fg.shape
    hull = np.zeros((dz, dy, dx), dtype=bool)
    hull[0] = hull[dz - 1] = True
    hull[:, 0] = hull[:, dy - 1] = True
    hull[:, :, 0] = hull[:, :, dx - 1] = True
    shell = fg & hull
    if not shell.any():
        return []
    lab = cc3d.connected_components(shell.astype(np.uint8), connectivity=26)
    n_lab = int(lab.max())
    if n_lab == 0:
        return []
    sizes = [(lab == i).sum() for i in range(1, n_lab + 1)]
    order = sorted(range(1, n_lab + 1), key=lambda i: sizes[i - 1], reverse=True)
    qv = quant_vox if quant_vox is not None else max(1, min(patch_size) // 8)
    skip = skip_keys or set()
    seen: set = set()
    out: List[Tuple[int, int, int]] = []
    face_opts = [(0, True), (0, False), (1, True), (1, False), (2, True), (2, False)]
    for li in order:
        if len(out) >= max_centers:
            break
        zz_i, yy_i, xx_i = np.where(lab == li)
        if zz_i.size == 0:
            continue
        mz, my, mx = float(zz_i.mean()), float(yy_i.mean()), float(xx_i.mean())
        c_fac = [
            np.sum(zz_i == 0),
            np.sum(zz_i == dz - 1),
            np.sum(yy_i == 0),
            np.sum(yy_i == dy - 1),
            np.sum(xx_i == 0),
            np.sum(xx_i == dx - 1),
        ]
        face_indices = [i for i in range(6) if c_fac[i] > 0]
        face_indices.sort(key=lambda i: c_fac[i], reverse=True)
        for fi in face_indices:
            if len(out) >= max_centers:
                break
            axis, is_low = face_opts[fi]
            nsz, nsy, nsx = shift_spatial_slices(sz, sy, sx, axis, is_low, patch_size, padded_shape)
            Z0 = int(np.clip(sz.start + int(round(mz)), 0, padded_shape[0] - 1))
            Y0 = int(np.clip(sy.start + int(round(my)), 0, padded_shape[1] - 1))
            X0 = int(np.clip(sx.start + int(round(mx)), 0, padded_shape[2] - 1))
            if axis == 0:
                pz = int(np.clip(nsz.start + patch_size[0] // 2, 0, padded_shape[0] - 1))
                py, px = Y0, X0
            elif axis == 1:
                pz, py, px = Z0, int(np.clip(nsy.start + patch_size[1] // 2, 0, padded_shape[1] - 1)), X0
            else:
                pz, py, px = Z0, Y0, int(np.clip(nsx.start + patch_size[2] // 2, 0, padded_shape[2] - 1))
            if qv > 1:
                pz, py, px = (pz // qv) * qv, (py // qv) * qv, (px // qv) * qv
            cz, cy, cx = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
            s2 = spatial_slices_to_tuple(cz, cy, cx)
            if s2 == seed_key or s2 in seen or s2 in skip:
                continue
            seen.add(s2)
            out.append((pz, py, px))
    return out


def plan_border_expansion_centers_from_logits(
    patch_logits: torch.Tensor,
    lm: Labels,
    sz: slice,
    sy: slice,
    sx: slice,
    patch_size: Tuple[int, int, int],
    padded_shape: Tuple[int, int, int],
    seed_key: Tuple[int, int, int, int, int, int],
    max_centers: int,
    quant_vox: Optional[int] = None,
    skip_keys: Optional[Set[Tuple[int, int, int, int, int, int]]] = None,
) -> List[Tuple[int, int, int]]:
    from nanounet.infer.roi_slices import patch_fg_mask_from_logits

    fg = patch_fg_mask_from_logits(patch_logits, lm)
    return plan_border_expansion_centers_from_fg(
        fg, sz, sy, sx, patch_size, padded_shape, seed_key, max_centers, quant_vox, skip_keys
    )


DEFAULT_MAX_BORDER_EXPAND_EXTRA = 16
