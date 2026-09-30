"""DataLoader worker startup and supervised-batch collate.

worker_init is pickled by module path on spawn (macOS). train/patches/iterable.py
re-exports it so existing pickles still load. collate_patches flattens prompt
variants into rows.
"""

from __future__ import annotations

import torch

from nanounet.data.loader.prefs import pin_worker_threads


def worker_init(worker_id: int) -> None:
    from nanounet.runtime import set_safe_tmpdir
    set_safe_tmpdir()
    pin_worker_threads()


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
