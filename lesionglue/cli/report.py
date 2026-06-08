"""Train/evaluate matcher and nearest-mask baseline, then write report.json."""

import argparse
import gc
import importlib.util
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from tracking.common import CACHE_ROOT, DATASET_ROOT, print0, seed_all
from tracking.data.features import add_feat_args, cache_tag, desc_dim, feat_from_args
from tracking.report import eval_baseline, eval_gnn
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import MatcherModule


@dataclass
class ReportConfig:
    epochs: int = 400
    max_steps: int = 40000
    val_check_steps: int = 500
    warmup_steps: int = 1000
    batch_size: int = 8
    val_batch_size: int = 1
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
    hard_pair_w: float = 0.0
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
    ema_start_step: int = 1000
    tta_n: int = 0
    hard_k: int = 4
    early_stop_patience: int = 20
    nce_scope: str = "graph"
    desc_norm: bool = False


ARG_FIELDS = (
    ("epochs", int), ("max_steps", int), ("val_check_steps", int), ("warmup_steps", int),
    ("batch_size", int), ("val_batch_size", int), ("num_workers", int), ("seed", int),
    ("lr", float), ("weight_decay", float), ("d", int), ("layers", int), ("heads", int),
    ("dropout", float), ("sinkhorn_w", float), ("pair_w", float), ("nce_w", float),
    ("hard_pair_w", float), ("dust_w", float), ("dust_pos_w", float), ("nce_tau", float),
    ("sinkhorn_iters", int), ("fu_jitter", float), ("p_drop_fu", float), ("p_drop_bl", float),
    ("desc_jitter_frac", float), ("dust_tau", float), ("ema_decay", float),
    ("ema_start_step", int), ("tta_n", int), ("hard_k", int), ("early_stop_patience", int),
)


def _accelerator() -> str:
    return "mps" if torch.backends.mps.is_available() else "auto"


def _cuda_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(DATASET_ROOT))
ap.add_argument("--cache", default=str(CACHE_ROOT))
ap.add_argument("--out", required=True)
ap.add_argument("--checkpoint", default="", type=str, help="skip training and only eval from this .ckpt")
for name, typ in ARG_FIELDS:
    ap.add_argument(f"--{name.replace('_', '-')}", type=typ, default=getattr(ReportConfig, name))
ap.add_argument("--nce-scope", choices=("graph", "batch"), default=ReportConfig.nce_scope)
ap.add_argument("--no-desc-norm", action="store_true")
ap.add_argument("--no-early-stop", action="store_true")
ap.add_argument("--no-ema", action="store_true")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--wandb-project", default="lesion-tracking")
ap.add_argument("--wandb-run-name", default="", type=str)
ap.add_argument("--eval-batch-size", type=int, default=1)
ap.add_argument("--eval-num-workers", type=int, default=0)
ap.add_argument("--eval-device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--baseline-full-mask-cache", action="store_true")
add_feat_args(ap)

if __name__ == "__main__":
    args = ap.parse_args()
    feat = feat_from_args(args)
    cfg = ReportConfig()
    for f in fields(ReportConfig):
        if hasattr(args, f.name):
            setattr(cfg, f.name, getattr(args, f.name))
    cfg.desc_norm = not args.no_desc_norm
    root, cache, out = Path(args.root), Path(args.cache), Path(args.out)
    tag = cache_tag(feat)
    for split in ("train", "val", "test"):
        assert (cache / "processed" / f"{split}_{tag}.pt").is_file(), f"missing cached {split} graph for feat={feat.mode}"
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)

    if args.checkpoint.strip():
        best = Path(args.checkpoint.strip()).resolve()
        assert best.is_file(), f"checkpoint not found: {best}"
    else:
        dm = MatcherDataModule(
            cache_root=cache, dataset_root=root, feat=feat, batch_size=cfg.batch_size,
            val_batch_size=cfg.val_batch_size, num_workers=cfg.num_workers, fu_jitter_scale=cfg.fu_jitter,
            p_drop_fu=cfg.p_drop_fu, p_drop_bl=cfg.p_drop_bl, desc_jitter_frac=cfg.desc_jitter_frac,
        )
        dm.prepare_data()
        dm.setup()
        mod = MatcherModule(
            d=cfg.d, layers=cfg.layers, heads=cfg.heads, lr=cfg.lr, weight_decay=cfg.weight_decay,
            dropout=cfg.dropout, sinkhorn_w=cfg.sinkhorn_w, pair_w=cfg.pair_w, nce_w=cfg.nce_w,
            hard_pair_w=cfg.hard_pair_w, dust_w=cfg.dust_w, dust_pos_w=cfg.dust_pos_w,
            nce_tau=cfg.nce_tau, sinkhorn_iters=cfg.sinkhorn_iters, max_epochs=cfg.epochs,
            max_steps=cfg.max_steps, warmup_steps=cfg.warmup_steps, ema_decay=cfg.ema_decay,
            ema_start_step=cfg.ema_start_step, tta_n=cfg.tta_n, dust_tau=cfg.dust_tau,
            hard_k=cfg.hard_k, k_intra=8, fu_jitter_scale=cfg.fu_jitter,
            desc_jitter_frac=cfg.desc_jitter_frac, desc_dim=desc_dim(feat), desc_norm=cfg.desc_norm,
            nce_scope=cfg.nce_scope,
        )
        ckpt = ModelCheckpoint(dirpath=out / "checkpoints", monitor="val_match_score", save_top_k=1, mode="max")
        callbacks = [ckpt, TQDMProgressBar()]
        if not args.no_early_stop:
            callbacks.insert(1, EarlyStopping(monitor="val_match_score", mode="max", patience=cfg.early_stop_patience))
        logger = False
        if args.wandb or bool(args.wandb_run_name.strip()):
            if importlib.util.find_spec("wandb") is None:
                raise SystemExit("wandb requested but not installed: pip install 'wandb>=0.12.10'")
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(project=args.wandb_project, name=(args.wandb_run_name.strip() or None))
        trainer_kw = dict(default_root_dir=out, accelerator=_accelerator(), devices=1, log_every_n_steps=10, callbacks=callbacks, logger=logger)
        if cfg.max_steps > 0:
            trainer_kw.update(max_steps=cfg.max_steps, max_epochs=-1, val_check_interval=cfg.val_check_steps, check_val_every_n_epoch=None)
        else:
            trainer_kw["max_epochs"] = cfg.epochs
        pl.Trainer(**trainer_kw).fit(mod, dm)
        best = Path(ckpt.best_model_path)
        assert best.is_file(), "training finished without a best checkpoint"
        del mod, dm
        _cuda_gc()

    print0(f"best checkpoint: {best}")
    prog = not args.quiet
    one_mask = not args.baseline_full_mask_cache
    gnn_val = eval_gnn(best, root, cache, "val", args.eval_batch_size, args.eval_num_workers, cfg.tta_n, not args.no_ema, feat=feat, eval_device=args.eval_device, show_progress=prog)
    _cuda_gc()
    gnn_test = eval_gnn(best, root, cache, "test", args.eval_batch_size, args.eval_num_workers, cfg.tta_n, not args.no_ema, feat=feat, eval_device=args.eval_device, show_progress=prog)
    _cuda_gc()
    bl_val = eval_baseline(root, cache, "val", feat=feat, show_progress=prog, one_mask_cache=one_mask)
    bl_test = eval_baseline(root, cache, "test", feat=feat, show_progress=prog, one_mask_cache=one_mask)
    report = {"best_checkpoint": str(best), "feat_mode": feat.mode, "config": asdict(cfg), "gnn": {"val": gnn_val, "test": gnn_test}, "baseline": {"val": bl_val, "test": bl_test}}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print0(f"wrote {out / 'report.json'}")
