"""Lightning module: prompt-aware ResEnc, val Dice, SGD + poly LR."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from torch import autocast

from nanounet.config import RoiPromptConfig, load_config, save_config
from nanounet.mem_diag import (
    cgroup_epoch_deltas,
    cgroup_mem_bytes,
    log_snapshot,
    log_wandb_scalars,
    mem_diag_enabled,
    purge_torch_tmp,
)
from nanounet.model.dice_helpers import get_tp_fp_fn_tn, pooled_fg_dice, val_split_metrics
from nanounet.model.losses import build_loss
from nanounet.model.lr_schedule import PolyLRScheduler, StretchedTailPolyLRScheduler
from nanounet.model.mae_transfer import load_full_net, load_mae_encoder
from nanounet.model.network import build_net, build_net_longi
from nanounet.plan.plans import Plans

class NanoUNetLM(pl.LightningModule):
    def __init__(
        self,
        plans_path: str,
        dataset_json_path: str,
        roi_cfg_path: str,
        output_dir: str,
        initial_lr: float = 0.01,
        weight_decay: float = 3e-5,
        num_epochs: int = 1000,
        lr_schedule: str = "poly",
        stretched_k: int = 750,
        stretched_ref: int = 1000,
        stretched_exp: float = 0.9,
        enable_deep_supervision: bool = True,
        loss_type: str = "dc_ce",
        optimizer: str = "sgd",
        mae_ckpt: str | None = None,
        init_weights: str | None = None,
        longi: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.plans_path = plans_path
        self.dataset_json_path = dataset_json_path
        self.output_dir = output_dir
        self.roi_cfg: RoiPromptConfig = load_config(roi_cfg_path)
        self.pm = Plans(plans_path)
        self.cm = self.pm.get_configuration("3d_fullres")
        self.dj = load_json(dataset_json_path)
        self.label_manager = self.pm.get_label_manager(self.dj)
        build = build_net_longi if longi else build_net
        self.net = build(self.cm, self.label_manager, self.dj, enable_deep_supervision)
        if init_weights is not None:
            load_full_net(self.net, init_weights)
        elif mae_ckpt is not None:
            load_mae_encoder(self.net, mae_ckpt)
        self.loss = build_loss(self.cm, self.label_manager, enable_deep_supervision, loss_type=loss_type, is_ddp=False)
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr_schedule = lr_schedule
        self.stretched_k = stretched_k
        self.stretched_ref = stretched_ref
        self.stretched_exp = stretched_exp
        self.enable_deep_supervision = enable_deep_supervision
        self._val_buf: List[Dict[str, Any]] = []
        self._prev_cgroup_file: int | None = None
        self._prev_cgroup_shmem: int | None = None

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self) -> None:
        maybe_mkdir_p(self.output_dir)
        shutil.copyfile(self.plans_path, join(self.output_dir, "plans.json"))
        shutil.copyfile(self.dataset_json_path, join(self.output_dir, "dataset.json"))
        save_config(self.roi_cfg, join(self.output_dir, "nano_config.json"))
        wid = os.environ.get("WANDB_RUN_ID", "").strip()
        if wid:
            open(join(self.output_dir, "wandb_run_id.txt"), "w", encoding="utf-8").write(wid + "\n")

    def on_train_epoch_start(self) -> None:
        self._epoch_t0 = time.perf_counter()
        purge_torch_tmp()

    def training_step(self, batch: dict, _bidx: int):
        x = batch["data"].to(self.device, non_blocking=True)
        y = batch["target"]
        if isinstance(y, list):
            y = [i.to(self.device, non_blocking=True) for i in y]
        else:
            y = y.to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            out = self.net(x)
            loss = self.loss(out, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch: dict, _bidx: int):
        x = batch["data"].to(self.device, non_blocking=True)
        y = batch["target"]
        if isinstance(y, list):
            y = [i.to(self.device, non_blocking=True) for i in y]
        else:
            y = y.to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            out = self.net(x)
            loss = self.loss(out, y)
        if self.enable_deep_supervision:
            out = out[0]
            y = y[0]
        axes = list(range(2, out.ndim))
        output_seg = out.argmax(1)[:, None]
        oh = torch.zeros_like(out, dtype=torch.float32, device=out.device)
        oh.scatter_(1, output_seg, 1)
        if self.label_manager.has_ignore_label:
            mask = (y != self.label_manager.ignore_label).float()
            y = y.clone()
            y[y == self.label_manager.ignore_label] = 0
        else:
            # Out-of-FOV voxels are -1 (crop_to_nonzero) with no ignore label -> background;
            # clamp so the metric one-hot scatter stays in bounds (no-op without -1).
            mask = None
            y = y.clamp_min(0)
        tp, fp, fn, _ = get_tp_fp_fn_tn(oh, y, axes=axes, mask=mask)
        tg, pg, ng, da, fb = val_split_metrics(tp[:, 1:], fp[:, 1:], fn[:, 1:], y, output_seg)
        self._val_buf.append(
            {"tp": tg, "fp": pg, "fn": ng, "dice_a": da, "fp_b": fb, "loss": float(loss.detach())}
        )

    def on_validation_epoch_start(self) -> None:
        self._val_buf.clear()

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "_epoch_t0") and not self.trainer.sanity_checking:
            self.log("epoch_wall_time_sec", float(time.perf_counter() - self._epoch_t0))
        if not self._val_buf:
            return
        da = torch.cat([v["dice_a"] for v in self._val_buf])
        fb = torch.cat([v["fp_b"] for v in self._val_buf])
        self.log("val_dice", pooled_fg_dice(self._val_buf), prog_bar=True)
        self.log("val_dice_macro", float(da.mean()) if da.numel() else float("nan"), prog_bar=True)
        self.log("val_fp", float(fb.mean()) if fb.numel() else 0.0, prog_bar=False)
        self.log("val_n_a", float(da.numel()))
        self.log("val_n_b", float(fb.numel()))
        self.log("val_loss", float(np.mean([v["loss"] for v in self._val_buf])), prog_bar=False)
        if mem_diag_enabled():
            ep = int(self.current_epoch)
            fd, sd = cgroup_epoch_deltas(self._prev_cgroup_file, self._prev_cgroup_shmem)
            cg = cgroup_mem_bytes()
            self._prev_cgroup_file = cg.get("file")
            self._prev_cgroup_shmem = cg.get("shmem")
            extra: dict = {"epoch": ep, "stage": "supervised"}
            if fd is not None:
                extra["cgroup_file_delta_bytes"] = fd
            if sd is not None:
                extra["cgroup_shmem_delta_bytes"] = sd
            row = log_snapshot(f"sup_epoch_{ep}", self.output_dir, extra=extra)
            log_wandb_scalars(self, row)

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(self.net.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.SGD(
                self.net.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
        if self.lr_schedule == "stretched_tail_poly":
            sched = StretchedTailPolyLRScheduler(
                opt,
                self.initial_lr,
                self.num_epochs,
                k_transition=self.stretched_k,
                ref_poly_steps=self.stretched_ref,
                exponent=self.stretched_exp,
            )
        else:
            sched = PolyLRScheduler(opt, self.initial_lr, self.num_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def on_exception(self, exception: BaseException) -> None:
        if mem_diag_enabled():
            log_snapshot(
                "sup_exception",
                self.output_dir,
                extra={"epoch": int(self.current_epoch), "error": type(exception).__name__},
            )
