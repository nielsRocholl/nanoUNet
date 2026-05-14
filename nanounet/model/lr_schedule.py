"""Polynomial LR + stretched-tail variant (nnU-Net)."""

from __future__ import annotations

from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]


class StretchedTailPolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        num_epochs: int,
        k_transition: int = 750,
        ref_poly_steps: int = 1000,
        exponent: float = 0.9,
        last_epoch: int = None,
        epoch_offset: int = 0,
    ):
        self.initial_lr = initial_lr
        self.num_epochs = num_epochs
        self.k_transition = k_transition
        self.ref_poly_steps = ref_poly_steps
        self.exponent = exponent
        self.epoch_offset = epoch_offset
        self.ctr = 0
        if not (0 < k_transition < ref_poly_steps):
            raise ValueError("require 0 < k_transition < ref_poly_steps")
        if not (0 <= epoch_offset < num_epochs):
            raise ValueError("epoch_offset")
        super().__init__(optimizer, last_epoch if last_epoch is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        s = max(0, current_step - self.epoch_offset)
        n_last = (self.num_epochs - 1) - self.epoch_offset
        ref, exp, lr0, k = self.ref_poly_steps, self.exponent, self.initial_lr, self.k_transition
        if s < k or n_last <= k:
            new_lr = lr0 * (1.0 - s / ref) ** exp
        else:
            lr_k = lr0 * (1.0 - k / ref) ** exp
            lr_end = lr0 * (1.0 - (ref - 1) / ref) ** exp
            denom = max(n_last - k, 1)
            t = (s - k) / denom
            new_lr = lr_k + (lr_end - lr_k) * t
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
