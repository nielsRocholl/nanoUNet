"""Train bipartite lesion matcher (Lightning).

`RichProgressBar` requires `rich` (see `requirements.txt`); Lightning errors if it is missing.
W&B metrics require `wandb` in the environment (`pip install -r requirements.txt`).

DataLoader workers on macOS use spawn: keep ``trainer.fit`` under ``if __name__ == "__main__"`` so
re-imported worker modules do not recurse into training.
"""

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from tracking.common import CACHE_ROOT, DATASET_ROOT, seed_all
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import MatcherModule


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    num_workers: int = 2
    seed: int = 0
    d: int = 128
    layers: int = 4
    heads: int = 4
    pair_w: float = 1.0
    row_w: float = 0.5
    none_w: float = 0.2


def _accelerator() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "auto"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="lightning_logs")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--d", type=int, default=TrainConfig.d)
    ap.add_argument("--layers", type=int, default=TrainConfig.layers)
    ap.add_argument("--heads", type=int, default=TrainConfig.heads)
    ap.add_argument("--pair-w", type=float, default=TrainConfig.pair_w)
    ap.add_argument("--row-w", type=float, default=TrainConfig.row_w)
    ap.add_argument("--none-w", type=float, default=TrainConfig.none_w)
    ap.add_argument("--wandb", action="store_true", help="log to W&B (also on if --wandb-run-name is set)")
    ap.add_argument("--wandb-project", default="lesion-tracking")
    ap.add_argument("--wandb-run-name", default="", type=str)
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        d=args.d,
        layers=args.layers,
        heads=args.heads,
        pair_w=args.pair_w,
        row_w=args.row_w,
        none_w=args.none_w,
    )
    seed_all(cfg.seed)
    dm = MatcherDataModule(
        cache_root=Path(args.cache),
        dataset_root=Path(args.root),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    dm.prepare_data()
    dm.setup()
    mod = MatcherModule(
        d=cfg.d,
        layers=cfg.layers,
        heads=cfg.heads,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        pos_weight=dm.pos_weight,
        pair_w=cfg.pair_w,
        row_w=cfg.row_w,
        none_w=cfg.none_w,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=args.out, monitor="val_loss", save_top_k=3, mode="min")
    use_wandb = args.wandb or bool(args.wandb_run_name.strip())
    logger = False
    if use_wandb:
        if importlib.util.find_spec("wandb") is None:
            raise SystemExit("wandb requested but not installed: pip install 'wandb>=0.12.10'")
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=args.wandb_project,
            name=(args.wandb_run_name.strip() or None),
        )
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=args.out,
        accelerator=_accelerator(),
        devices=1,
        log_every_n_steps=10,
        callbacks=[ckpt, RichProgressBar()],
        logger=logger,
    )
    trainer.fit(mod, dm)
