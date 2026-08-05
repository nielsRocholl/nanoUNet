"""Lightning module: prompt-aware ResEnc, val Dice, SGD + poly LR."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from torch import autocast

from nanounet.config import RoiPromptConfig, load_config, save_config
from nanounet.diag import purge_torch_tmp
from nanounet.model.dice_helpers import prompt_pair_dice, subset_dice_row, val_step_row
from nanounet.model.losses import build_loss, consistency_dice_term
from nanounet.model.lr_schedule import PolyLRScheduler, StretchedTailPolyLRScheduler
from nanounet.model.mae_transfer import load_full_net, load_mae_encoder
from nanounet.model.network import build_net, build_net_longi
from nanounet.plan.plans import Plans
from nanounet.train.val_metrics import log_val_metrics

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
        consistency_weight: float = 0.0,
        consistency_warmup_epochs: int = 50,
        warmup_epochs: int = 0,
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
        # is_ddp=False is correct, not a stub: plans set batch_dice=False, so dice is per-sample
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
        self.longi = longi
        self.consistency_weight_max = consistency_weight
        self.consistency_warmup_epochs = consistency_warmup_epochs
        self.warmup_epochs = warmup_epochs
        # Prompt-heatmap channel indices, per the fixed layouts documented in patch_iterable.py:
        # supervised [CT, hm+, hm-], longi [FU_CT, FU_hm+, FU_hm-, BL_CT, BL_hm+, BL_hm-].
        self._prompt_ch = [1, 2, 4, 5] if longi else [1, 2]
        self._val_buf: List[Dict[str, Any]] = []
        self._val_buf_ablated: List[Dict[str, Any]] = []
        self._agreement_buf: List[torch.Tensor] = []
        self._meta_buf: List[Dict[str, Any]] = []

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
            loss_seg = self.loss(out, y)
            if self.consistency_weight_max > 0:
                pair_id = batch["pair_id"].to(self.device, non_blocking=True)
                w = self.consistency_warmup_epochs  # epoch+1: epoch 0 must not be a dead epoch
                ramp = 1.0 if w <= 0 else min(1.0, (self.current_epoch + 1) / w)
                lam = self.consistency_weight_max * ramp
                loss_consistency = consistency_dice_term(out, pair_id)
                loss = loss_seg + lam * loss_consistency
            else:
                loss_consistency = torch.zeros((), device=loss_seg.device)
                loss = loss_seg
        self.log("train_loss_seg", loss_seg, batch_size=x.shape[0])
        self.log("train_loss_consistency", loss_consistency, batch_size=x.shape[0])
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
            # Collapse diagnostic: zero the prompt-heatmap channels and re-run. If val_dice here
            # closes the gap to the normal val_dice, the net (and/or the consistency term) is
            # learning to ignore the click -- lambda is too high.
            x_ablated = x.clone()
            x_ablated[:, self._prompt_ch] = 0.0
            out_ablated = self.net(x_ablated)
        ds, lm = self.enable_deep_supervision, self.label_manager
        self._val_buf.append(val_step_row(out, y, lm, ds, float(loss.detach()), batch["click_inside"]))
        self._val_buf_ablated.append(val_step_row(out_ablated, y, lm, ds, 0.0))
        if "scenario" in batch:
            meta = {
                k: batch[k].cpu()
                for k in ("scenario", "cohort", "size_bucket", "has_subset", "draws_matched")
            }
            meta["click_inside"] = batch["click_inside"].cpu()
            if bool(meta["has_subset"].any()):
                ys = batch["target_subset"].to(self.device, non_blocking=True)
                meta["subset_row"] = subset_dice_row(out, ys, lm, ds)
            self._meta_buf.append(meta)
        # val_prompt_agreement: 3rd forward, only when the val dataloader emits a 2nd independent
        # prompt on the same patch (data_module sets emit_prompt2=True for validation only).
        if "data_prompt2" in batch:
            x2 = batch["data_prompt2"].to(self.device, non_blocking=True)
            with autocast(self.device.type, enabled=self.device.type == "cuda"):
                out2 = self.net(x2)
            self._agreement_buf.append(prompt_pair_dice(out, out2, self.enable_deep_supervision))

    def on_validation_epoch_start(self) -> None:
        self._val_buf.clear()
        self._val_buf_ablated.clear()
        self._agreement_buf.clear()
        self._meta_buf.clear()

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "_epoch_t0") and not self.trainer.sanity_checking:
            self.log("epoch_wall_time_sec", float(time.perf_counter() - self._epoch_t0))
        if not self._val_buf:
            return
        log_val_metrics(self)

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(self.net.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.SGD(self.net.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        if self.lr_schedule == "stretched_tail_poly":
            sched = StretchedTailPolyLRScheduler(
                opt,
                self.initial_lr,
                self.num_epochs,
                k_transition=self.stretched_k,
                ref_poly_steps=self.stretched_ref,
                exponent=self.stretched_exp,
                warmup_epochs=self.warmup_epochs,
            )
        else:
            sched = PolyLRScheduler(opt, self.initial_lr, self.num_epochs, warmup_epochs=self.warmup_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
