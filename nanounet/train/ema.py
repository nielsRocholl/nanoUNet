"""Weight EMA callback: shadow copy of net params/buffers, updated after each train batch.

In a noise-dominated regime (small batches, SGD momentum 0.99) EMA typically buys the equivalent
of a few hundred epochs of averaging for free. The shadow rides along in the checkpoint via
Callback.state_dict/load_state_dict -- Lightning calls these automatically, and skips restoring
state that isn't present in an older checkpoint, so checkpoints saved before this callback existed
still load unmodified. Validation swaps the shadow in for one extra pass over the val set and logs
val_dice_ema next to val_dice so the human can compare before trusting it; lightning_module.py
needs no change for this.

A Callback is the correct extension point here (not a wrapper layer): it hooks Lightning's own
train/validation loop rather than sitting between it and the model.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import autocast

from nanounet.model.dice_helpers import pooled_fg_dice, val_step_row


class EMACallback(pl.Callback):
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.decay <= 0:
            return
        if not self.shadow:
            # Lazy init on the first live batch: by now the net is on its final device, and a
            # resumed run has already had load_state_dict populate self.shadow before this fires.
            self.shadow = {k: v.detach().clone() for k, v in pl_module.net.state_dict().items()}
            return
        for k, v in pl_module.net.state_dict().items():
            s = self.shadow[k]
            if s.is_floating_point():
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.decay <= 0 or not self.shadow or trainer.sanity_checking:
            return
        raw = {k: v.detach().clone() for k, v in pl_module.net.state_dict().items()}
        pl_module.net.load_state_dict(self.shadow)
        buf = []
        with torch.no_grad():
            for batch in trainer.datamodule.val_dataloader():
                x = batch["data"].to(pl_module.device, non_blocking=True)
                y = batch["target"]
                y = (
                    [t.to(pl_module.device, non_blocking=True) for t in y]
                    if isinstance(y, list)
                    else y.to(pl_module.device, non_blocking=True)
                )
                with autocast(pl_module.device.type, enabled=pl_module.device.type == "cuda"):
                    out = pl_module.net(x)
                buf.append(val_step_row(out, y, pl_module.label_manager, pl_module.enable_deep_supervision, 0.0))
        pl_module.net.load_state_dict(raw)
        pl_module.log("val_dice_ema", pooled_fg_dice(buf), sync_dist=True)

    def state_dict(self) -> dict[str, Any]:
        return {"shadow": self.shadow}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.shadow = state_dict["shadow"]
