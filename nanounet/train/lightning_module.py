"""Lightning module: prompt-aware ResEnc, val Dice, SGD + poly LR."""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from torch import autocast

from nanounet.config import RoiPromptConfig, load_config, save_config
from nanounet.model.dice_helpers import get_tp_fp_fn_tn
from nanounet.model.losses import build_loss
from nanounet.model.lr_schedule import PolyLRScheduler, StretchedTailPolyLRScheduler
from nanounet.model.mae_transfer import load_mae_encoder
from nanounet.model.network import build_net
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
        mae_ckpt: str | None = None,
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
        self.net = build_net(self.cm, self.label_manager, self.dj, enable_deep_supervision)
        if mae_ckpt is not None:
            load_mae_encoder(self.net, mae_ckpt)
        self.loss = build_loss(self.cm, self.label_manager, enable_deep_supervision, loss_type=loss_type, is_ddp=False)
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.lr_schedule = lr_schedule
        self.stretched_k = stretched_k
        self.stretched_ref = stretched_ref
        self.stretched_exp = stretched_exp
        self.enable_deep_supervision = enable_deep_supervision
        self._val_buf: List[Dict[str, Any]] = []

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
            mask = None
        tp, fp, fn, _ = get_tp_fp_fn_tn(oh, y, axes=axes, mask=mask)
        tp = tp[:, 1:].detach().cpu()
        fp = fp[:, 1:].detach().cpu()
        fn = fn[:, 1:].detach().cpu()
        self._val_buf.append({"tp": tp, "fp": fp, "fn": fn, "loss": float(loss.detach())})

    def on_validation_epoch_start(self) -> None:
        self._val_buf.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_buf:
            return
        tp = torch.cat([v["tp"] for v in self._val_buf], dim=0)
        fp = torch.cat([v["fp"] for v in self._val_buf], dim=0)
        fn = torch.cat([v["fn"] for v in self._val_buf], dim=0)
        tg = tp.sum(0).numpy()
        pg = fp.sum(0).numpy()
        ng = fn.sum(0).numpy()
        dg = np.array([2 * float(a) / (2 * a + b + c) if (2 * a + b + c) > 0 else np.nan for a, b, c in zip(tg, pg, ng)])
        self.log("val_dice", float(np.nanmean(dg)), prog_bar=True)
        self.log("val_loss", float(np.mean([v["loss"] for v in self._val_buf])), prog_bar=False)

    def configure_optimizers(self):
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
