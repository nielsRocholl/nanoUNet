"""Train bipartite lesion matcher with step-clock Lightning."""

import argparse
import importlib.util
from dataclasses import dataclass, fields
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from tracking.common import CACHE_ROOT, DATASET_ROOT, dump_json, seed_all
from tracking.data.features import add_feat_args, desc_dim, feat_from_args
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import MatcherModule

CKPT_MONITOR = "val_match_score_ema"


@dataclass
class TrainConfig:
    epochs: int = 400
    max_steps: int = 8000
    val_check_steps: int = 250
    warmup_steps: int = 1000
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    val_batch_size: int = 1
    num_workers: int = 2
    seed: int = 0
    d: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.2
    sinkhorn_w: float = 1.0
    pair_w: float = 0.1
    nce_w: float = 0.3
    hard_pair_w: float = 0.0
    dust_w: float = 0.30
    dust_pos_w: float = 1.0
    nce_tau: float = 0.1
    sinkhorn_iters: int = 20
    fu_jitter: float = 0.3
    p_drop_fu: float = 0.10
    p_drop_bl: float = 0.10
    desc_jitter_frac: float = 0.0
    set_attn_blocks: int = 0
    ema_decay: float = 0.999
    ema_start_step: int = 1000
    tta_n: int = 0
    dust_tau: float = 0.2
    hard_k: int = 4
    early_stop_patience: int = 8
    nce_scope: str = "graph"
    desc_norm: bool = False
    dust_pair_summary: bool = True
    dust_legacy_linear: bool = False
    sinkhorn_uniform_fu: bool = True
    fold: int | None = None
    n_folds: int = 5
    cv_seed: int = 0
    val_score_ema_beta: float = 0.3
    edge_cross_attn: bool = False


ARG_FIELDS = (
    ("epochs", int), ("max_steps", int), ("val_check_steps", int), ("warmup_steps", int),
    ("lr", float), ("weight_decay", float), ("batch_size", int), ("val_batch_size", int),
    ("num_workers", int), ("seed", int), ("d", int), ("layers", int), ("heads", int),
    ("dropout", float), ("sinkhorn_w", float), ("pair_w", float), ("nce_w", float),
    ("hard_pair_w", float), ("dust_w", float), ("dust_pos_w", float), ("dust_tau", float),
    ("nce_tau", float), ("sinkhorn_iters", int), ("fu_jitter", float), ("p_drop_fu", float),
    ("p_drop_bl", float), ("desc_jitter_frac", float), ("set_attn_blocks", int),
    ("ema_decay", float), ("ema_start_step", int), ("tta_n", int), ("hard_k", int),
    ("early_stop_patience", int), ("n_folds", int), ("cv_seed", int), ("val_score_ema_beta", float),
)


def _accelerator() -> str:
    return "mps" if torch.backends.mps.is_available() else "auto"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="lightning_logs")
    for name, typ in ARG_FIELDS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=typ, default=getattr(TrainConfig, name))
    ap.add_argument("--fold", type=int, default=None, help="CV fold index (enables patient-level k-fold val)")
    ap.add_argument("--nce-scope", choices=("graph", "batch"), default=TrainConfig.nce_scope)
    ap.add_argument("--single-pos-fu", action="store_true", help="legacy one-argmax FU Sinkhorn target")
    ap.add_argument("--desc-norm", action="store_true", help="enable descriptor LayerNorm (R8 bundle)")
    ap.add_argument("--no-desc-norm", action="store_true", help="force descriptor LayerNorm off")
    ap.add_argument("--no-early-stop", action="store_true", help="run full step/epoch budget")
    ap.add_argument("--dust-no-pair-summary", action="store_true", help="zero pair-logit summaries for dust head")
    ap.add_argument("--dust-legacy-linear", action="store_true", help="Round-5-style linear dust head")
    ap.add_argument("--edge-cross-attn", action="store_true", help="edge-conditioned cross-attn refine layer")
    ap.add_argument("--wandb", action="store_true", help="log to W&B")
    ap.add_argument("--wandb-project", default="lesion-tracking")
    ap.add_argument("--wandb-run-name", default="", type=str)
    add_feat_args(ap)
    args = ap.parse_args()

    feat = feat_from_args(args)
    cfg = TrainConfig()
    for f in fields(TrainConfig):
        if hasattr(args, f.name):
            setattr(cfg, f.name, getattr(args, f.name))
    if args.desc_norm:
        cfg.desc_norm = True
    elif args.no_desc_norm:
        cfg.desc_norm = False
    cfg.dust_pair_summary = not args.dust_no_pair_summary
    cfg.dust_legacy_linear = args.dust_legacy_linear
    cfg.sinkhorn_uniform_fu = not args.single_pos_fu
    cfg.edge_cross_attn = args.edge_cross_attn
    cfg.fold = args.fold
    if cfg.fold is not None:
        assert cfg.fold in range(cfg.n_folds)
    seed_all(cfg.seed)

    dm = MatcherDataModule(
        cache_root=Path(args.cache), dataset_root=Path(args.root), feat=feat,
        batch_size=cfg.batch_size, val_batch_size=cfg.val_batch_size, num_workers=cfg.num_workers,
        fu_jitter_scale=cfg.fu_jitter, p_drop_fu=cfg.p_drop_fu, p_drop_bl=cfg.p_drop_bl,
        desc_jitter_frac=cfg.desc_jitter_frac, fold=cfg.fold, n_folds=cfg.n_folds, cv_seed=cfg.cv_seed,
    )
    dm.prepare_data()
    dm.setup()
    mod = MatcherModule(
        d=cfg.d, layers=cfg.layers, heads=cfg.heads, lr=cfg.lr, weight_decay=cfg.weight_decay,
        dropout=cfg.dropout, sinkhorn_w=cfg.sinkhorn_w, pair_w=cfg.pair_w, nce_w=cfg.nce_w,
        hard_pair_w=cfg.hard_pair_w, dust_w=cfg.dust_w, dust_pos_w=cfg.dust_pos_w,
        nce_tau=cfg.nce_tau, sinkhorn_iters=cfg.sinkhorn_iters, max_epochs=cfg.epochs,
        max_steps=cfg.max_steps, warmup_steps=cfg.warmup_steps, dust_pair_summary=cfg.dust_pair_summary,
        dust_legacy_linear=cfg.dust_legacy_linear, set_attn_blocks=cfg.set_attn_blocks,
        ema_decay=cfg.ema_decay, ema_start_step=cfg.ema_start_step, tta_n=cfg.tta_n,
        dust_tau=cfg.dust_tau, k_intra=8, hard_k=cfg.hard_k, fu_jitter_scale=cfg.fu_jitter,
        desc_jitter_frac=cfg.desc_jitter_frac, desc_dim=desc_dim(feat), desc_norm=cfg.desc_norm,
        nce_scope=cfg.nce_scope, sinkhorn_uniform_fu=cfg.sinkhorn_uniform_fu,
        edge_cross_attn=cfg.edge_cross_attn, val_score_ema_beta=cfg.val_score_ema_beta,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.ckpt"):
        if p.name not in ("best.ckpt", "last.ckpt"):
            p.unlink(missing_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=str(out), monitor=CKPT_MONITOR, mode="max", save_top_k=1, save_last=True,
        filename="best", auto_insert_metric_name=False, enable_version_counter=False,
    )
    callbacks = [ckpt, TQDMProgressBar()]
    if not args.no_early_stop:
        callbacks.insert(1, EarlyStopping(monitor=CKPT_MONITOR, mode="max", patience=cfg.early_stop_patience))

    logger = False
    if args.wandb or bool(args.wandb_run_name.strip()):
        if importlib.util.find_spec("wandb") is None:
            raise SystemExit("wandb requested but not installed: pip install 'wandb>=0.12.10'")
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=(args.wandb_run_name.strip() or None))
    trainer_kw = dict(default_root_dir=args.out, accelerator=_accelerator(), devices=1, log_every_n_steps=10, callbacks=callbacks, logger=logger)
    if cfg.max_steps > 0:
        trainer_kw.update(max_steps=cfg.max_steps, max_epochs=-1, val_check_interval=cfg.val_check_steps, check_val_every_n_epoch=None)
    else:
        trainer_kw["max_epochs"] = cfg.epochs
    pl.Trainer(**trainer_kw).fit(mod, dm)
    fold_metrics = {
        "fold": cfg.fold,
        "best_ckpt": str(ckpt.best_model_path),
        "val_match_score_ema": mod._best_ema_score,
        "val_match_score_raw": mod._best_raw_score,
        "val_match_score_peak": mod._val_score_peak,
        **{f"val_acc_{k}": v for k, v in mod._best_sub.items()},
    }
    dump_json(out / "fold_metrics.json", fold_metrics)
