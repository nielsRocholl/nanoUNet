"""Train best-val matcher, evaluate val/test, run nearest-mask baseline, write report.json."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from tracking.common import CACHE_ROOT, DATASET_ROOT, print0, seed_all
from tracking.report import eval_baseline, eval_gnn
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import MatcherModule


@dataclass
class ReportConfig:
    epochs: int = 400
    batch_size: int = 8
    num_workers: int = 2
    seed: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-2
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
    dust_tau: float = 0.2
    ema_decay: float = 0.999
    ema_start_epoch: int = 5
    tta_n: int = 0
    early_stop_patience: int = 60


def _accelerator() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "auto"


ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(DATASET_ROOT))
ap.add_argument("--cache", default=str(CACHE_ROOT))
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=ReportConfig.epochs)
ap.add_argument("--batch-size", type=int, default=ReportConfig.batch_size)
ap.add_argument("--num-workers", type=int, default=ReportConfig.num_workers)
ap.add_argument("--seed", type=int, default=ReportConfig.seed)
ap.add_argument("--lr", type=float, default=ReportConfig.lr)
ap.add_argument("--weight-decay", type=float, default=ReportConfig.weight_decay)
ap.add_argument("--d", type=int, default=ReportConfig.d)
ap.add_argument("--layers", type=int, default=ReportConfig.layers)
ap.add_argument("--heads", type=int, default=ReportConfig.heads)
ap.add_argument("--dropout", type=float, default=ReportConfig.dropout)
ap.add_argument("--sinkhorn-w", type=float, default=ReportConfig.sinkhorn_w)
ap.add_argument("--pair-w", type=float, default=ReportConfig.pair_w)
ap.add_argument("--nce-w", type=float, default=ReportConfig.nce_w)
ap.add_argument("--dust-w", type=float, default=ReportConfig.dust_w)
ap.add_argument("--dust-pos-w", type=float, default=ReportConfig.dust_pos_w)
ap.add_argument("--nce-tau", type=float, default=ReportConfig.nce_tau)
ap.add_argument("--sinkhorn-iters", type=int, default=ReportConfig.sinkhorn_iters)
ap.add_argument("--fu-jitter", type=float, default=ReportConfig.fu_jitter)
ap.add_argument("--p-drop-fu", type=float, default=ReportConfig.p_drop_fu)
ap.add_argument("--p-drop-bl", type=float, default=ReportConfig.p_drop_bl)
ap.add_argument("--desc-jitter-frac", type=float, default=ReportConfig.desc_jitter_frac)
ap.add_argument("--dust-tau", type=float, default=ReportConfig.dust_tau)
ap.add_argument("--ema-decay", type=float, default=ReportConfig.ema_decay)
ap.add_argument("--ema-start", type=int, default=ReportConfig.ema_start_epoch)
ap.add_argument("--tta-n", type=int, default=ReportConfig.tta_n)
ap.add_argument("--early-stop-patience", type=int, default=ReportConfig.early_stop_patience)
ap.add_argument("--no-early-stop", action="store_true")
ap.add_argument("--no-ema", action="store_true")

if __name__ == "__main__":
    args = ap.parse_args()
    cfg = ReportConfig(
        epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        lr=args.lr, weight_decay=args.weight_decay, d=args.d, layers=args.layers, heads=args.heads,
        dropout=args.dropout, sinkhorn_w=args.sinkhorn_w, pair_w=args.pair_w, nce_w=args.nce_w,
        dust_w=args.dust_w, dust_pos_w=args.dust_pos_w, nce_tau=args.nce_tau,
        sinkhorn_iters=args.sinkhorn_iters, fu_jitter=args.fu_jitter, p_drop_fu=args.p_drop_fu,
        p_drop_bl=args.p_drop_bl, desc_jitter_frac=args.desc_jitter_frac, dust_tau=args.dust_tau,
        ema_decay=args.ema_decay, ema_start_epoch=args.ema_start, tta_n=args.tta_n,
        early_stop_patience=args.early_stop_patience,
    )
    root = Path(args.root)
    cache = Path(args.cache)
    out = Path(args.out)
    for split in ("train", "val", "test"):
        assert (cache / "processed" / f"{split}_v4.pt").is_file(), f"missing cached {split} graph"
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)

    dm = MatcherDataModule(
        cache_root=cache, dataset_root=root, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        fu_jitter_scale=cfg.fu_jitter, p_drop_fu=cfg.p_drop_fu, p_drop_bl=cfg.p_drop_bl,
        desc_jitter_frac=cfg.desc_jitter_frac,
    )
    dm.prepare_data()
    dm.setup()
    mod = MatcherModule(
        d=cfg.d, layers=cfg.layers, heads=cfg.heads, lr=cfg.lr, weight_decay=cfg.weight_decay,
        dropout=cfg.dropout, sinkhorn_w=cfg.sinkhorn_w, pair_w=cfg.pair_w, nce_w=cfg.nce_w,
        dust_w=cfg.dust_w, dust_pos_w=cfg.dust_pos_w, nce_tau=cfg.nce_tau,
        sinkhorn_iters=cfg.sinkhorn_iters, max_epochs=cfg.epochs, ema_decay=cfg.ema_decay,
        ema_start_epoch=cfg.ema_start_epoch, tta_n=cfg.tta_n, dust_tau=cfg.dust_tau,
        k_intra=8, fu_jitter_scale=cfg.fu_jitter, desc_jitter_frac=cfg.desc_jitter_frac,
    )
    ckpt = ModelCheckpoint(dirpath=out / "checkpoints", monitor="val_match_score", save_top_k=1, mode="max")
    callbacks = [ckpt, TQDMProgressBar()]
    if not args.no_early_stop:
        callbacks.insert(1, EarlyStopping(monitor="val_match_score", mode="max", patience=cfg.early_stop_patience))
    trainer = pl.Trainer(
        max_epochs=cfg.epochs, default_root_dir=out, accelerator=_accelerator(), devices=1,
        log_every_n_steps=10, callbacks=callbacks, logger=False,
    )
    trainer.fit(mod, dm)
    best = Path(ckpt.best_model_path)
    assert best.is_file(), "training finished without a best checkpoint"

    print0(f"best checkpoint: {best}")
    report = {
        "best_checkpoint": str(best),
        "config": asdict(cfg),
        "gnn": {
            "val": eval_gnn(best, root, cache, "val", cfg.batch_size, cfg.num_workers, cfg.tta_n, not args.no_ema),
            "test": eval_gnn(best, root, cache, "test", cfg.batch_size, cfg.num_workers, cfg.tta_n, not args.no_ema),
        },
        "baseline": {
            "val": eval_baseline(root, cache, "val"),
            "test": eval_baseline(root, cache, "test"),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print0(f"wrote {out / 'report.json'}")
