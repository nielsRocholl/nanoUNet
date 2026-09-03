"""Variant keypoint concat/split, heatmap rendering, click-inside bookkeeping, and batch collate
for PatchIterable. Split out of patch_iterable.py to keep that file under the 200-LOC limit; the
concept boundary is "iteration/sharding" (patch_iterable.py) vs. "render a variant into a tensor /
flatten a batch" (here).

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


def concat_variant_keypoints(variants: list, longi: bool) -> torch.Tensor:
    """Concat every variant's clicks into one (N,3) tensor so one augmentation pass moves all."""
    parts = []
    for v in variants:
        parts += [v["points_pos"]] + ([v["bl_points_pos"]] if longi else [])
    if not parts:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.from_numpy(np.concatenate(parts, axis=0)).float()


def split_variant_keypoints(kp: torch.Tensor, variants: list, longi: bool) -> list:
    """Inverse of concat_variant_keypoints: slice augmented `keypoints` back per variant."""
    out, off = [], 0
    for v in variants:
        n_pp = v["points_pos"].shape[0]
        pp, off = kp[off : off + n_pp], off + n_pp
        entry = {"pp": pp, "n_fp": int(v.get("n_false_pos", 0))}
        if longi:
            n_bp = v["bl_points_pos"].shape[0]
            entry["bp"], off = kp[off : off + n_bp], off + n_bp
        out.append(entry)
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


def render_variant(o: dict, entry: dict, raw: dict, longi: bool, final_patch_size, pr) -> torch.Tensor:
    shape = tuple(int(s) for s in final_patch_size)
    fu_hm = encode_points_to_heatmap(
        _point_list(entry["pp"]), shape, pr.point_radius_vox, pr.encoding, None,
        pr.prompt_intensity_scale,
    ).unsqueeze(0)
    if not longi:
        return torch.cat([o["image"][0:1], fu_hm], dim=0)
    fu_stream = torch.cat([o["image"][0:1], fu_hm], dim=0)
    if raw["null_baseline"]:
        bl_stream = fu_stream  # duplicate rendered FU stream -> identity DWB
    else:
        bl_hm = encode_points_to_heatmap(
            _point_list(entry["bp"]), shape, pr.point_radius_vox, pr.encoding, None,
            pr.prompt_intensity_scale,
        ).unsqueeze(0)
        bl_stream = torch.cat([o["image"][1:2], bl_hm], dim=0)
    return torch.cat([fu_stream, bl_stream], dim=0)


_META_KEYS = ("scenario", "cohort", "size_bucket", "has_subset", "draws_matched")


def collate_patches(batch: list) -> dict:
    """Flatten each item's variants into rows; `pair_id` groups rows from the same raw patch.
    `click_inside` (-1/0/1, see click_inside_flags above) rides along per row. When items carry
    `data_prompt2` (val emit_prompt2 only, always prompts_per_patch==1 there so this is a 1:1
    per-item tensor, not exploded), it is stacked separately under the same key. `scenario` /
    `cohort` / `size_bucket` / `has_subset` / `draws_matched` are row-aligned integer codes from
    the fixed val manifest (ValPatchDataset) -- absent during training."""
    t0 = batch[0]["target"]
    is_list = isinstance(t0, list)
    have_prompt2 = "data_prompt2" in batch[0]
    rows_data, rows_target, pair_ids, click_inside, prompt2_rows = [], [], [], [], []
    meta = {k: [] for k in _META_KEYS if k in batch[0]}
    for pid, item in enumerate(batch):
        for v, ci in zip(item["data_variants"], item["click_inside"]):
            rows_data.append(v)
            rows_target.append(item["target"])
            pair_ids.append(pid)
            click_inside.append(ci)
            for k in meta:
                meta[k].append(item[k])
        if have_prompt2:
            prompt2_rows.append(item["data_prompt2"])
    data = torch.stack(rows_data)
    if is_list:
        target = [torch.stack([t[i] for t in rows_target], dim=0) for i in range(len(t0))]
    else:
        target = torch.stack(rows_target)
    out = {
        "data": data,
        "target": target,
        "pair_id": torch.tensor(pair_ids, dtype=torch.long),
        "click_inside": torch.tensor(click_inside, dtype=torch.long),
    }
    if have_prompt2:
        out["data_prompt2"] = torch.stack(prompt2_rows)
    for k, v in meta.items():
        out[k] = torch.tensor(v, dtype=torch.long)
    if "target_subset" in batch[0]:
        out["target_subset"] = torch.stack([item["target_subset"] for item in batch])
    return out
