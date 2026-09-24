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
        raise ValueError(
            f"{path} has no hyper_parameters['num_epochs']: not a nanoUNet Lightning checkpoint.\n"
            f"Expected a .ckpt saved by nanounet_train, passed via --resume or --mae-resume.\n"
            f"Fix: point --resume/--mae-resume at a checkpoint under out/checkpoints or out/mae_pretrain/checkpoints   (see docs/steps/train.md)"
        )
    target = int(hp["num_epochs"])
    ep = d.get("epoch")
    if ep is not None:
        return int(ep), target
    try:
        return int(d["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]), target
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{path} has neither a top-level 'epoch' key nor loops.fit_loop.epoch_progress.current.completed.\n"
            f"Expected the standard PyTorch Lightning 2.x checkpoint layout written by nanounet_train.\n"
            f"Fix: point --resume/--mae-resume at an unmodified nanounet_train checkpoint   (see docs/steps/train.md)"
        ) from None


def pl_ckpt_stage_done(epoch_idx: int, num_epochs: int) -> bool:
    return epoch_idx >= num_epochs


def pl_ckpt_assert_epochs_match(path: str, expected_target_epochs: int) -> None:
    _, tgt = pl_ckpt_epoch_and_target(path)
    if tgt != expected_target_epochs:
        raise ValueError(
            f"{path} was trained for num_epochs={tgt}, but the CLI expects {expected_target_epochs}.\n"
            f"Expected --epochs/--mae-epochs to match the value the checkpoint was created with.\n"
            f"Fix: pass --epochs {tgt} (or --mae-epochs {tgt} for an MAE checkpoint) to match it, or start a fresh run"
        )
