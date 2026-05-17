"""Iterable patch dataset + LightningDataModule (prefetch, pin_memory)."""

from __future__ import annotations

from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from torch.utils.data import DataLoader, IterableDataset

from nanounet.common import (
    ANISO_THRESHOLD,
    dataloader_num_workers,
    preprocessed_dir,
    raw_dir,
    setup_logging,
)
from nanounet.config import RoiPromptConfig, load_config
from nanounet.data import augment
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.data.sampling import build_patch
from nanounet.plan.plans import Plans
from nanounet.plan.splits import fold_keys, load_or_create_splits
from nanounet.train.patch_size import get_patch_size

setup_logging()


def _ds_scales(cm, enable: bool):
    if not enable:
        return None
    p = cm.pool_op_kernel_sizes
    return list(list(i) for i in 1 / np.cumprod(np.vstack(p), axis=0))[:-1]


def _rotation_dummy_mirroring(patch_size: List[int]):
    patch_size = list(patch_size)
    assert len(patch_size) == 3
    do_dummy = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
    if do_dummy:
        rotation_for_DA = (-180.0 / 360 * 2.0 * np.pi, 180.0 / 360 * 2.0 * np.pi)
    else:
        rotation_for_DA = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
    mirror_axes = (0, 1, 2)
    initial_patch_size = get_patch_size(patch_size, rotation_for_DA, rotation_for_DA, rotation_for_DA, (0.85, 1.25))
    if do_dummy:
        initial_patch_size[0] = patch_size[0]
    return rotation_for_DA, do_dummy, tuple(initial_patch_size), mirror_axes


class _PatchIterable(IterableDataset):
    def __init__(
        self,
        folder: str,
        keys: List[str],
        roi_cfg: RoiPromptConfig,
        patch_size: np.ndarray,
        final_patch_size: np.ndarray,
        annotated_key,
        tf,
        force_zero_prompt: bool,
        num_batches: int,
        batch_size: int,
        base_seed: int,
    ):
        self.folder = folder
        self.keys = keys
        self.roi_cfg = roi_cfg
        self.patch_size = patch_size
        self.final_patch_size = final_patch_size
        self.annotated_key = annotated_key
        self.tf = tf
        self.force_zero_prompt = force_zero_prompt
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
        for _ in range(n_here):
            cid = self.keys[int(rng.integers(0, len(self.keys)))]
            data, seg, _, prop = ds.load_case(cid)
            raw = build_patch(
                data,
                seg,
                prop,
                self.roi_cfg,
                self.patch_size,
                self.final_patch_size,
                self.annotated_key,
                self.force_zero_prompt,
                rng,
            )
            im = torch.from_numpy(raw["image"]).float()
            se = torch.from_numpy(raw["segmentation"]).short()
            with torch.no_grad():
                o = self.tf(**{"image": im, "segmentation": se})
            yield {"data": o["image"], "target": o["segmentation"]}


def _collate(batch: list) -> dict:
    data = torch.stack([b["data"] for b in batch])
    t0 = batch[0]["target"]
    if isinstance(t0, list):
        target = [torch.stack([b["target"][i] for b in batch], dim=0) for i in range(len(t0))]
    else:
        target = torch.stack([b["target"] for b in batch])
    return {"data": data, "target": target}


class NanoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        fold: int,
        plans_identifier: str,
        roi_cfg_path: str,
        batch_size: int | None = None,
        num_iterations_per_epoch: int = 250,
        num_val_iterations: int = 50,
        oversample_foreground: float = 0.33,
        enable_deep_supervision: bool = True,
        pin_memory: bool | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.fold = fold
        self.plans_identifier = plans_identifier
        self.roi_cfg = load_config(roi_cfg_path)
        self.batch_size = batch_size
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_val_iterations = num_val_iterations
        self.oversample_foreground = oversample_foreground
        self.enable_ds = enable_deep_supervision
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        pp = preprocessed_dir()
        raw = raw_dir()
        pl_path = join(pp, self.dataset_name, self.plans_identifier + ".json")
        self.pm = Plans(pl_path)
        self.cm = self.pm.get_configuration("3d_fullres")
        dj = load_json(join(raw, self.dataset_name, "dataset.json"))
        self.label_manager = self.pm.get_label_manager(dj)
        if self.batch_size is None:
            self.batch_size = self.cm.batch_size
        fold_dir = join(pp, self.dataset_name)
        case_dir = join(pp, self.dataset_name, self.cm.data_identifier)
        all_ids = Blosc2Folder.get_identifiers(case_dir)
        ntr = dj.get("numTraining")
        tr_keys = all_ids[: int(ntr)] if ntr is not None else list(all_ids)
        sp = join(fold_dir, "splits_final.json")
        spl = load_or_create_splits(sp, tr_keys, 5, 12345)
        self.tr_keys, self.val_keys = fold_keys(spl, self.fold)
        self.case_folder = case_dir
        ps = np.array(self.cm.patch_size)
        dss = _ds_scales(self.cm, self.enable_ds)
        rot, do_dum, init_ps, mirrors = _rotation_dummy_mirroring(list(ps))
        umn = self.cm.use_mask_for_norm
        fl = self.label_manager.foreground_labels
        reg = self.label_manager.foreground_regions if self.label_manager.has_regions else None
        ign = self.label_manager.ignore_label
        self.train_tf = augment.train_transforms(ps, rot, dss, mirrors, do_dum, umn, False, fl, reg, ign)
        self.val_tf = augment.val_transforms(dss, False, fl, reg, ign)
        self.patch_size = ps
        self.final_ps = ps
        self.init_patch_size = np.array(init_ps)

    def train_dataloader(self) -> DataLoader:
        it = _PatchIterable(
            self.case_folder,
            self.tr_keys,
            self.roi_cfg,
            self.init_patch_size,
            self.final_ps,
            self.label_manager.annotated_classes_key,
            self.train_tf,
            False,
            self.num_iterations_per_epoch,
            self.batch_size,
            self.fold + 1000 * self.num_iterations_per_epoch,
        )
        nw = dataloader_num_workers(train=True)
        return DataLoader(
            it,
            batch_size=self.batch_size,
            num_workers=nw,
            pin_memory=self.pin_memory,
            persistent_workers=nw > 0,
            prefetch_factor=4 if nw else None,
            collate_fn=_collate,
        )

    def val_dataloader(self) -> DataLoader:
        it = _PatchIterable(
            self.case_folder,
            self.val_keys,
            self.roi_cfg,
            self.final_ps,
            self.final_ps,
            self.label_manager.annotated_classes_key,
            self.val_tf,
            not self.roi_cfg.prompt.validation_use_prompt,
            self.num_val_iterations,
            self.batch_size,
            self.fold + 2000,
        )
        nw = dataloader_num_workers(train=False)
        return DataLoader(
            it,
            batch_size=self.batch_size,
            num_workers=nw,
            pin_memory=self.pin_memory,
            persistent_workers=nw > 0,
            prefetch_factor=2 if nw else None,
            collate_fn=_collate,
        )
