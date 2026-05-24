"""LightningDataModule for supervised training (prefetch, pin_memory)."""

from __future__ import annotations

from functools import partial
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from torch.utils.data import DataLoader

from nanounet.common import ANISO_THRESHOLD, preprocessed_dir, raw_dir, setup_logging
from nanounet.config import load_config
from nanounet.data import augment
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.dataloader_prefs import DataloaderBucket, build_iter_dataloader, init_dataloader_ipc
from nanounet.mem_diag import mem_diag_enabled
from nanounet.plan.plans import Plans
from nanounet.plan.splits import fold_keys, load_or_create_splits
from nanounet.train.patch_iterable import PatchIterable, collate_patches, worker_init
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


class NanoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        fold: int,
        plans_identifier: str,
        roi_cfg_path: str,
        dl_bucket: DataloaderBucket,
        batch_size: int | None = None,
        num_iterations_per_epoch: int = 250,
        num_val_iterations: int = 50,
        oversample_foreground: float = 0.33,
        enable_deep_supervision: bool = True,
        pin_memory: bool | None = None,
        mem_diag_dir: str | None = None,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.fold = fold
        self.plans_identifier = plans_identifier
        self.roi_cfg = load_config(roi_cfg_path)
        self.batch_size = batch_size
        self.dl_bucket = dl_bucket
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_val_iterations = num_val_iterations
        self.oversample_foreground = oversample_foreground
        self.enable_ds = enable_deep_supervision
        self.mem_diag_dir = mem_diag_dir
        self.persistent_workers = persistent_workers
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
        init_dataloader_ipc()
        it = PatchIterable(
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
            self.mem_diag_dir,
        )
        b = self.dl_bucket
        nw = b.nw_train
        winit = partial(worker_init, out_dir=self.mem_diag_dir or ".") if nw else None
        return build_iter_dataloader(
            it,
            batch_size=self.batch_size,
            bucket=b,
            nw=nw,
            prefetch=b.prefetch_train,
            collate_fn=collate_patches,
            pin_memory=self.pin_memory,
            worker_init_fn=winit,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        init_dataloader_ipc()
        it = PatchIterable(
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
            self.mem_diag_dir,
        )
        b = self.dl_bucket
        nw = b.nw_val
        winit = partial(worker_init, out_dir=self.mem_diag_dir or ".") if nw else None
        return build_iter_dataloader(
            it,
            batch_size=self.batch_size,
            bucket=b,
            nw=nw,
            prefetch=b.prefetch_val,
            collate_fn=collate_patches,
            pin_memory=self.pin_memory,
            worker_init_fn=winit,
            persistent_workers=self.persistent_workers,
        )
