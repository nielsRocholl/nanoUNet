"""Lightning 2.x checkpoint metadata: fit epoch index vs saved ``num_epochs``.

``epoch`` in the file is ``trainer.current_epoch`` at save (PL 2.2+). Training is
complete when that index has reached ``num_epochs`` (next epoch would be ``num_epochs``)."""

from __future__ import annotations

from typing import Any

import torch


def pl_ckpt_epoch_and_target(path: str) -> tuple[int, int]:
    d: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    hp = d.get("hyper_parameters")
    if hp is None or not isinstance(hp, dict) or "num_epochs" not in hp:
        raise ValueError(f"not a nanoUNet Lightning checkpoint: {path}")
    target = int(hp["num_epochs"])
    ep = d.get("epoch")
    if ep is not None:
        return int(ep), target
    try:
        return int(d["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]), target
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"checkpoint missing top-level epoch: {path}") from None


def pl_ckpt_stage_done(epoch_idx: int, num_epochs: int) -> bool:
    return epoch_idx >= num_epochs


def pl_ckpt_assert_epochs_match(path: str, expected_target_epochs: int) -> None:
    _, tgt = pl_ckpt_epoch_and_target(path)
    if tgt != expected_target_epochs:
        raise ValueError(f"checkpoint num_epochs={tgt}, CLI expects {expected_target_epochs}")
