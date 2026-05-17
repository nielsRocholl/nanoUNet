"""Train bipartite lesion matcher (Lightning).

Uses `TQDMProgressBar`. W&B: `pip install wandb`.

DataLoader workers on macOS use spawn: keep ``trainer.fit`` under ``if __name__ == "__main__``.
"""

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from tracking.common import CACHE_ROOT, DATASET_ROOT, seed_all
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import MatcherModule


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    num_workers: int = 2
    seed: int = 0
    d: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.2
    sinkhorn_w: float = 1.0
    pair_w: float = 0.1
    nce_w: float = 0.3
    dust_w: float = 0.30
    dust_pos_w: float = 1.0
    nce_tau: float = 0.1
    sinkhorn_iters: int = 20
    fu_jitter: float = 0.3
    p_drop_fu: float = 0.10
    p_drop_bl: float = 0.10
    desc_jitter_frac: float = 0.0
    dust_pair_summary: bool = True
    dust_legacy_linear: bool = False
    set_attn_blocks: int = 0
    sinkhorn_uniform_fu: bool = True
    ema_decay: float = 0.999
    ema_start_epoch: int = 5
    tta_n: int = 0
    dust_tau: float = 0.2
    early_stop_patience: int = 60


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
    ap.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    ap.add_argument("--sinkhorn-w", type=float, default=TrainConfig.sinkhorn_w)
    ap.add_argument("--pair-w", type=float, default=TrainConfig.pair_w)
    ap.add_argument("--nce-w", type=float, default=TrainConfig.nce_w)
    ap.add_argument("--dust-w", type=float, default=TrainConfig.dust_w)
    ap.add_argument("--dust-pos-w", type=float, default=TrainConfig.dust_pos_w)
    ap.add_argument("--dust-tau", type=float, default=TrainConfig.dust_tau, help="Hungarian decode row-normalized threshold")
    ap.add_argument("--nce-tau", type=float, default=TrainConfig.nce_tau)
    ap.add_argument("--sinkhorn-iters", type=int, default=TrainConfig.sinkhorn_iters)
    ap.add_argument("--fu-jitter", type=float, default=TrainConfig.fu_jitter, help="FU jitter scale; 0 disables FU noise")
    ap.add_argument("--p-drop-fu", type=float, default=TrainConfig.p_drop_fu)
    ap.add_argument("--p-drop-bl", type=float, default=TrainConfig.p_drop_bl)
    ap.add_argument("--desc-jitter-frac", type=float, default=TrainConfig.desc_jitter_frac)
    ap.add_argument("--set-attn-blocks", type=int, default=TrainConfig.set_attn_blocks, help="0 disables SetAttn stack")
    ap.add_argument("--ema-decay", type=float, default=TrainConfig.ema_decay, help="0 disables EMA")
    ap.add_argument("--ema-start", type=int, default=TrainConfig.ema_start_epoch)
    ap.add_argument("--tta-n", type=int, default=TrainConfig.tta_n, help="val TTA passes (0=off)")
    ap.add_argument(
        "--single-pos-fu",
        action="store_true",
        help="Sinkhorn loss: legacy one-argmax FU target (ablates Round 7.1 uniform merge supervision)",
    )
    ap.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="run all --epochs (no stop on val_match_score plateau)",
    )
    ap.add_argument(
        "--dust-no-pair-summary",
        action="store_true",
        help="ablation: zero pair-logit summaries for DustHead (structural lever off)",
    )
    ap.add_argument(
        "--dust-legacy-linear",
        action="store_true",
        help="Round-5-style nn.Linear(d,1) dust head (for aug-only smoke vs DustHead)",
    )
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
        dropout=args.dropout,
        sinkhorn_w=args.sinkhorn_w,
        pair_w=args.pair_w,
        nce_w=args.nce_w,
        dust_w=args.dust_w,
        dust_pos_w=args.dust_pos_w,
        nce_tau=args.nce_tau,
        sinkhorn_iters=args.sinkhorn_iters,
        fu_jitter=args.fu_jitter,
        p_drop_fu=args.p_drop_fu,
        p_drop_bl=args.p_drop_bl,
        desc_jitter_frac=args.desc_jitter_frac,
        dust_pair_summary=not args.dust_no_pair_summary,
        dust_legacy_linear=args.dust_legacy_linear,
        set_attn_blocks=args.set_attn_blocks,
        ema_decay=args.ema_decay,
        ema_start_epoch=args.ema_start,
        tta_n=args.tta_n,
        dust_tau=args.dust_tau,
        early_stop_patience=args.early_stop_patience,
        sinkhorn_uniform_fu=not args.single_pos_fu,
    )
    seed_all(cfg.seed)
    dm = MatcherDataModule(
        cache_root=Path(args.cache),
        dataset_root=Path(args.root),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        fu_jitter_scale=cfg.fu_jitter,
        p_drop_fu=cfg.p_drop_fu,
        p_drop_bl=cfg.p_drop_bl,
        desc_jitter_frac=cfg.desc_jitter_frac,
    )
    dm.prepare_data()
    dm.setup()
    mod = MatcherModule(
        d=cfg.d,
        layers=cfg.layers,
        heads=cfg.heads,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        dropout=cfg.dropout,
        sinkhorn_w=cfg.sinkhorn_w,
        pair_w=cfg.pair_w,
        nce_w=cfg.nce_w,
        dust_w=cfg.dust_w,
        dust_pos_w=cfg.dust_pos_w,
        nce_tau=cfg.nce_tau,
        sinkhorn_iters=cfg.sinkhorn_iters,
        max_epochs=cfg.epochs,
        dust_pair_summary=cfg.dust_pair_summary,
        dust_legacy_linear=cfg.dust_legacy_linear,
        set_attn_blocks=cfg.set_attn_blocks,
        ema_decay=cfg.ema_decay,
        ema_start_epoch=cfg.ema_start_epoch,
        tta_n=cfg.tta_n,
        dust_tau=cfg.dust_tau,
        k_intra=8,
        fu_jitter_scale=cfg.fu_jitter,
        desc_jitter_frac=cfg.desc_jitter_frac,
        sinkhorn_uniform_fu=cfg.sinkhorn_uniform_fu,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=args.out, monitor="val_match_score", save_top_k=3, mode="max")
    callbacks = [ckpt, TQDMProgressBar()]
    if not args.no_early_stop:
        callbacks.insert(1, EarlyStopping(monitor="val_match_score", mode="max", patience=cfg.early_stop_patience))
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
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(mod, dm)
