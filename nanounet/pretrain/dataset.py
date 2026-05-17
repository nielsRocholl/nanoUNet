"""Pretrain patch iterable: random crop from preprocessed b2nd, mirror aug, no prompts."""

from __future__ import annotations

from typing import List

import os

import blosc2
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from torch.utils.data import DataLoader, IterableDataset

from nanounet.common import dataloader_num_workers, preprocessed_dir, raw_dir
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.plan.plans import Plans
from nanounet.plan.splits import fold_keys, load_or_create_splits


def _pretrain_collate(batch: list) -> dict:
    return {"data": torch.stack([b["data"] for b in batch])}


class PretrainPatchIterable(IterableDataset):
    def __init__(
        self,
        folder: str,
        keys: List[str],
        patch_size: np.ndarray,
        num_batches: int,
        batch_size: int,
        base_seed: int,
    ):
        self.folder = folder
        self.keys = keys
        self.patch_size = patch_size
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.base_seed = base_seed

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
        for _ in range(n_here):
            cid = self.keys[int(rng.integers(0, len(self.keys)))]
            data, _, _, _ = ds.load_case(cid)
            shp = data.shape[1:]
            for i in range(3):
                if shp[i] < ps[i]:
                    raise ValueError(f"case {cid} spatial {shp} smaller than patch {tuple(ps.tolist())}")
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


def _keys_fit_patch(case_dir: str, keys: List[str], ps: np.ndarray) -> List[str]:
    dparams = {"nthreads": 1}
    mmap = {} if os.name == "nt" else {"mmap_mode": "r"}
    out: List[str] = []
    for k in keys:
        data = blosc2.open(urlpath=join(case_dir, k + ".b2nd"), mode="r", dparams=dparams, **mmap)
        shp = data.shape[1:]
        if all(int(shp[i]) >= int(ps[i]) for i in range(3)):
            out.append(k)
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
    pin_memory: bool | None = None,
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
    tr_k = _keys_fit_patch(case_dir, tr_k, ps)
    va_k = _keys_fit_patch(case_dir, va_k, ps)
    if not tr_k or not va_k:
        raise ValueError(
            "MAE pretrain needs at least one case per split with spatial shape >= patch; "
            f"patch={tuple(ps.tolist())} train_ok={len(tr_k)} val_ok={len(va_k)}"
        )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    nw_tr = dataloader_num_workers(train=True)
    nw_va = dataloader_num_workers(train=False)
    tr_it = PretrainPatchIterable(case_dir, tr_k, ps, num_iterations_per_epoch, batch_size, base_seed_train)
    va_it = PretrainPatchIterable(case_dir, va_k, ps, num_val_iterations, batch_size, base_seed_val)
    tr = DataLoader(
        tr_it,
        batch_size=batch_size,
        num_workers=nw_tr,
        pin_memory=pin_memory,
        persistent_workers=nw_tr > 0,
        prefetch_factor=4 if nw_tr else None,
        collate_fn=_pretrain_collate,
    )
    va = DataLoader(
        va_it,
        batch_size=batch_size,
        num_workers=nw_va,
        pin_memory=pin_memory,
        persistent_workers=nw_va > 0,
        prefetch_factor=2 if nw_va else None,
        collate_fn=_pretrain_collate,
    )
    return tr, va
