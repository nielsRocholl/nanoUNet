"""Variant keypoint concat/split, heatmap rendering, and click-inside bookkeeping for
PatchIterable. Batch collate lives in data/loader/workers.py. Split out of iterable.py
to keep that file under the 200-LOC limit.

Click-inside bookkeeping: for every real (non-diagnostic) variant, `click_inside` records whether
the majority of its rendered positive clicks land on foreground in the post-augmentation
segmentation (-1 no positive click, 0 outside, 1 inside) -- no extra forward pass, pure indexing.
"""

from __future__ import annotations

import numpy as np
import torch

from nanounet.prompt.encoding import encode_points_to_heatmap


def _point_list(pts: torch.Tensor) -> list:
    return [] if pts.numel() == 0 else [tuple(v) for v in torch.round(pts).long().tolist()]


def concat_variant_keypoints(variants: list) -> torch.Tensor:
    """Concat every variant's clicks into one (N,3) tensor so one augmentation pass moves all."""
    parts = [v["points_pos"] for v in variants]
    if not parts:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.from_numpy(np.concatenate(parts, axis=0)).float()


def split_variant_keypoints(kp: torch.Tensor, variants: list) -> list:
    """Inverse of concat_variant_keypoints: slice augmented `keypoints` back per variant."""
    out, off = [], 0
    for v in variants:
        n_pp = v["points_pos"].shape[0]
        pp, off = kp[off : off + n_pp], off + n_pp
        out.append({"pp": pp, "n_fp": int(v.get("n_false_pos", 0))})
    return out


def click_inside_flags(entries: list, seg0: torch.Tensor) -> list:
    """Per real-variant row: 1 if a strict majority of its positive (FU) LESION clicks land on
    foreground in the post-augmentation, finest-resolution segmentation, 0 if the majority land
    on background (including clicks pushed outside the patch entirely), -1 if the row has no
    lesion click at all (excluded from both the inside and outside buckets by the caller).

    The trailing `n_fp` false-positive decoys are EXCLUDED from the vote. They are background by
    construction, so counting them made the majority test depend on lesion count: with L correctly
    placed lesion clicks plus one decoy the test `2*n_in > len(idx)` reduces to `L > 1`, so every
    single-lesion patch was flagged "outside" no matter where its click landed. Validation forces
    false_pos_probability=1.0, so that mislabelled every single-lesion val patch."""
    seg_arr = seg0[0] if seg0.ndim == 4 else seg0
    shp = seg_arr.shape
    flags = []
    for e in entries:
        pp = e["pp"]
        n_fp = int(e.get("n_fp", 0))
        n_les = pp.shape[0] - n_fp  # decoys are always the trailing entries (select_prompt_points)
        if n_les <= 0:
            flags.append(-1)
            continue
        idx = torch.round(pp[:n_les]).long().tolist()
        n_in = 0
        for z, y, x in idx:
            if 0 <= z < shp[0] and 0 <= y < shp[1] and 0 <= x < shp[2] and seg_arr[z, y, x] > 0:
                n_in += 1
        flags.append(1 if 2 * n_in > n_les else 0)
    return flags


def render_variant(o: dict, entry: dict, final_patch_size, pr) -> torch.Tensor:
    shape = tuple(int(s) for s in final_patch_size)
    hm = encode_points_to_heatmap(
        _point_list(entry["pp"]), shape, pr.point_radius_vox, pr.encoding, None,
        pr.prompt_intensity_scale,
    ).unsqueeze(0)
    return torch.cat([o["image"][0:1], hm], dim=0)
