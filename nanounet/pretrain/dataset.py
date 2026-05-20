"""MAE pretrain patches: random crop from b2nd, mirror aug, no prompts.

Cases with spatial shape smaller than the plan 3d_fullres patch on any axis are excluded;
build raises if train or val has no qualifying case after that filter.
"""

from __future__ import annotations

from functools import partial
from typing import List

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from torch.utils.data import DataLoader, IterableDataset

from nanounet.common import preprocessed_dir, print0, raw_dir
from nanounet.dataloader_prefs import DataloaderBucket, mae_keep_workers
from nanounet.data.blosc2_dataset import Blosc2Folder, case_spatial_shape
from nanounet.mem_diag import (
    log_snapshot,
    mem_diag_enabled,
    worker_diag_init,
    worker_diag_iter_end,
    worker_diag_tick,
)
from nanounet.plan.plans import Plans
from nanounet.plan.splits import fold_keys, load_or_create_splits


def _pretrain_collate(batch: list) -> dict:
    return {"data": torch.stack([b["data"] for b in batch])}


def _worker_init(worker_id: int, out_dir: str) -> None:
    worker_diag_init(worker_id, out_dir)


class PretrainPatchIterable(IterableDataset):
    def __init__(
        self,
        folder: str,
        keys: List[str],
        patch_size: np.ndarray,
        num_batches: int,
        batch_size: int,
        base_seed: int,
        mem_diag_dir: str | None = None,
    ):
        self.folder = folder
        self.keys = keys
        self.patch_size = patch_size
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.mem_diag_dir = mem_diag_dir

    def __len__(self) -> int:
        return self.num_batches * self.batch_size

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw = 1 if wi is None else wi.num_workers
        wid = 0 if wi is None else wi.id
        total = self.num_batches * self.batch_size
        n_here = total // nw + (1 if wid < (total % nw) else 0)
        seed = self.base_seed + wid * 10007
        rng = np.random.default_rng(seed)
        ds = Blosc2Folder(self.folder, identifiers=self.keys)
        ps = self.patch_size
        remaining = n_here
        opens = 0
        patches = 0
        if mem_diag_enabled() and self.mem_diag_dir:
            log_snapshot(
                f"worker_{wid}_iter_start",
                self.mem_diag_dir,
                extra={"worker_id": wid, "n_here": n_here, "num_keys": len(self.keys), "batch_size": self.batch_size},
                filename=f"mem_diag_worker_{wid}.jsonl",
            )
        while remaining > 0:
            cid = self.keys[int(rng.integers(0, len(self.keys)))]
            k = min(self.batch_size, remaining)
            with ds.open_case(cid, need_seg=False) as (data, _, _, _):
                opens += 1
                shp = data.shape[1:]
                assert all(int(shp[i]) >= int(ps[i]) for i in range(3))
                for _ in range(k):
                    lb = [int(rng.integers(0, shp[i] - ps[i] + 1)) for i in range(3)]
                    patch = np.asarray(
                        data[
                            :,
                            lb[0] : lb[0] + ps[0],
                            lb[1] : lb[1] + ps[1],
                            lb[2] : lb[2] + ps[2],
                        ],
                        dtype=np.float32,
                    )
                    for ax in (0, 1, 2):
                        if rng.random() < 0.5:
                            patch = np.flip(patch, axis=ax + 1).copy()
                    yield {"data": torch.from_numpy(patch).float()}
                    remaining -= 1
                    patches += 1
            if mem_diag_enabled() and self.mem_diag_dir:
                worker_diag_tick(wid, {"opens": opens, "patches": patches, "worker_id": wid})
        if mem_diag_enabled() and self.mem_diag_dir:
            worker_diag_iter_end(
                wid,
                {"opens": opens, "patches": patches, "worker_id": wid, "n_here": n_here},
            )


def _keys_fit_patch(case_dir: str, keys: List[str], ps: np.ndarray, mem_diag_dir: str | None, tag: str) -> List[str]:
    if mem_diag_enabled() and mem_diag_dir:
        log_snapshot(f"keys_fit_patch_before_{tag}", mem_diag_dir, extra={"n_keys": len(keys)})
    out: List[str] = []
    for k in keys:
        shp = case_spatial_shape(case_dir, k)
        if all(int(shp[i]) >= int(ps[i]) for i in range(3)):
            out.append(k)
    if mem_diag_enabled() and mem_diag_dir:
        log_snapshot(f"keys_fit_patch_after_{tag}", mem_diag_dir, extra={"n_keys": len(out)})
    return out


def build_pretrain_dataloaders(
    dataset_name: str,
    fold: int,
    plans_identifier: str,
    batch_size: int,
    num_iterations_per_epoch: int,
    num_val_iterations: int,
    base_seed_train: int,
    base_seed_val: int,
    bucket: DataloaderBucket,
    pin_memory: bool | None = None,
    mem_diag_dir: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    pp = preprocessed_dir()
    raw = raw_dir()
    pl_path = join(pp, dataset_name, plans_identifier + ".json")
    pm = Plans(pl_path)
    cm = pm.get_configuration("3d_fullres")
    dj = load_json(join(raw, dataset_name, "dataset.json"))
    fold_dir = join(pp, dataset_name)
    case_dir = join(pp, dataset_name, cm.data_identifier)
    all_ids = Blosc2Folder.get_identifiers(case_dir)
    ntr = dj.get("numTraining")
    tr_keys = all_ids[: int(ntr)] if ntr is not None else list(all_ids)
    sp = join(fold_dir, "splits_final.json")
    spl = load_or_create_splits(sp, tr_keys, 5, 12345)
    tr_k, va_k = fold_keys(spl, fold)
    ps = np.array(cm.patch_size)
    tr_0, va_0 = tr_k, va_k
    tr_k = _keys_fit_patch(case_dir, tr_k, ps, mem_diag_dir, "train")
    va_k = _keys_fit_patch(case_dir, va_k, ps, mem_diag_dir, "val")
    if len(tr_k) < len(tr_0) or len(va_k) < len(va_0):
        print0(
            "[dim]MAE pretrain: skipped cases smaller than patch "
            f"{tuple(ps.tolist())} — train {len(tr_0)}→{len(tr_k)}, val {len(va_0)}→{len(va_k)}[/dim]"
        )
    if not tr_k or not va_k:
        raise ValueError(
            "MAE pretrain needs at least one case per split with spatial shape >= patch; "
            f"patch={tuple(ps.tolist())} train_ok={len(tr_k)} val_ok={len(va_k)}"
        )
    if mae_keep_workers():
        import torch.multiprocessing as tmp

        if tmp.get_sharing_strategy() != "file_system":
            tmp.set_sharing_strategy("file_system")

    nw_tr = bucket.nw_train
    nw_va = bucket.nw_val
    pin_mem = False if nw_tr == 0 else (pin_memory if pin_memory is not None else False)
    tr_it = PretrainPatchIterable(
        case_dir, tr_k, ps, num_iterations_per_epoch, batch_size, base_seed_train, mem_diag_dir
    )
    va_it = PretrainPatchIterable(case_dir, va_k, ps, num_val_iterations, batch_size, base_seed_val, mem_diag_dir)
    winit_tr = partial(_worker_init, out_dir=mem_diag_dir or ".") if mem_diag_enabled() and mem_diag_dir and nw_tr else None
    winit_va = partial(_worker_init, out_dir=mem_diag_dir or ".") if mem_diag_enabled() and mem_diag_dir and nw_va else None

    def _dl(it, nw, pf, winit, pin):
        kw: dict = {
            "batch_size": batch_size,
            "num_workers": nw,
            "pin_memory": pin,
            "collate_fn": _pretrain_collate,
        }
        if nw:
            kw["persistent_workers"] = False
            kw["prefetch_factor"] = pf
            kw["worker_init_fn"] = winit
        return DataLoader(it, **kw)

    tr = _dl(tr_it, nw_tr, bucket.prefetch_train if nw_tr else None, winit_tr, pin_mem)
    va = _dl(va_it, nw_va, bucket.prefetch_val if nw_va else None, winit_va, False if nw_va == 0 else pin_mem)
    return tr, va
