"""MAE Lightning module: same ResEnc, masked-only L2 reconstruction, SGD + poly LR."""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from torch import autocast

from nanounet.model.lr_schedule import PolyLRScheduler
from nanounet.model.network import build_net
from nanounet.plan.plans import Plans, determine_num_input_channels
from nanounet.pretrain.masking import bottleneck_mask


class NanoMAELM(pl.LightningModule):
    def __init__(
        self,
        plans_path: str,
        dataset_json_path: str,
        output_dir: str,
        mask_ratio: float = 0.75,
        initial_lr: float = 1e-2,
        weight_decay: float = 3e-5,
        num_epochs: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["plans_path", "dataset_json_path"])
        self.output_dir = output_dir
        self.plans_path = plans_path
        self.dataset_json_path = dataset_json_path
        self.pm = Plans(plans_path)
        self.cm = self.pm.get_configuration("3d_fullres")
        self.dj = load_json(dataset_json_path)
        self.label_manager = self.pm.get_label_manager(self.dj)
        n_in = determine_num_input_channels(self.cm, self.dj)
        self.n_in = n_in
        self.net = build_net(
            self.cm,
            self.label_manager,
            self.dj,
            enable_deep_supervision=False,
            n_extra_in=0,
            num_classes_override=n_in,
        )
        p = np.vstack(self.cm.pool_op_kernel_sizes)
        self.total_stride = tuple(np.prod(p, axis=0).astype(int).tolist())

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _loss(self, batch: dict) -> torch.Tensor:
        x = batch["data"].to(self.device, non_blocking=True)
        assert x.shape[1] == self.n_in
        m = bottleneck_mask(self.cm.patch_size, self.total_stride, self.hparams.mask_ratio, x.shape[0], x.device)
        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            pred = self.forward(x * (1.0 - m))
            loss = ((pred.float() - x.float()).pow(2) * m).sum() / m.sum().clamp_min(1.0)
        return loss

    def training_step(self, batch: dict, _bidx: int):
        loss = self._loss(batch)
        self.log("train_recon_loss", loss, prog_bar=True, batch_size=batch["data"].shape[0])
        return loss

    def validation_step(self, batch: dict, _bidx: int):
        loss = self._loss(batch)
        self.log("val_recon_loss", loss, prog_bar=False, batch_size=batch["data"].shape[0])
        return loss

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.net.parameters(),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        sched = PolyLRScheduler(opt, self.hparams.initial_lr, self.hparams.num_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
