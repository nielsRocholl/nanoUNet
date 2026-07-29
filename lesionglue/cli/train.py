"""Train bipartite lesion matcher with step-clock Lightning."""

import argparse
import importlib.util
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from tracking.common import CACHE_ROOT, DATASET_ROOT, dump_json, seed_all
from tracking.config import CKPT_MONITOR, Config, dump_config, load_config
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import module_from_config


def _accelerator() -> str:
    return "mps" if torch.backends.mps.is_available() else "auto"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="lightning_logs")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lesion-tracking")
    ap.add_argument("--wandb-run-name", default="", type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        assert args.fold in range(cfg.n_folds)
    seed_all(cfg.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out / "config.json")

    dm = MatcherDataModule(
        cache_root=Path(args.cache), dataset_root=Path(args.root),
        batch_size=cfg.batch_size, val_batch_size=cfg.val_batch_size, num_workers=cfg.num_workers,
        fu_jitter_scale=cfg.fu_jitter, p_drop_fu=cfg.p_drop_fu, p_drop_bl=cfg.p_drop_bl,
        k_intra=cfg.k_intra, fold=args.fold, n_folds=cfg.n_folds, cv_seed=cfg.cv_seed,
    )
    dm.prepare_data()
    dm.setup()
    mod = module_from_config(cfg)

    keep_ckpts = ("best.ckpt", "last.ckpt", "best_raw.ckpt", "swa_plateau.ckpt")
    for p in out.glob("*.ckpt"):
        if p.name not in keep_ckpts:
            p.unlink(missing_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=str(out), monitor=CKPT_MONITOR, mode="max", save_top_k=1, save_last=True,
        filename="best", auto_insert_metric_name=False, enable_version_counter=False,
    )
    ckpt_raw = ModelCheckpoint(
        dirpath=str(out), monitor="val_match_score", mode="max", save_top_k=1, save_last=False,
        filename="best_raw", auto_insert_metric_name=False, enable_version_counter=False,
    )
    callbacks = [ckpt, ckpt_raw, TQDMProgressBar()]
    if not args.no_early_stop:
        callbacks.insert(1, EarlyStopping(monitor=CKPT_MONITOR, mode="max", patience=cfg.early_stop_patience))

    logger = False
    if args.wandb or bool(args.wandb_run_name.strip()):
        if importlib.util.find_spec("wandb") is None:
            raise SystemExit("wandb requested but not installed: pip install 'wandb>=0.12.10'")
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=(args.wandb_run_name.strip() or None))

    pl.Trainer(
        default_root_dir=args.out, accelerator=_accelerator(), devices=1, log_every_n_steps=10,
        callbacks=callbacks, logger=logger, max_steps=cfg.max_steps, max_epochs=-1,
        val_check_interval=cfg.val_check_steps, check_val_every_n_epoch=None,
    ).fit(mod, dm)

    fold_metrics = {
        "fold": args.fold,
        "best_ckpt": str(ckpt.best_model_path),
        "val_match_score_ema": mod._best_ema_score,
        "val_match_score_raw": mod._best_raw_score,
        "val_match_score_peak": mod._val_score_peak,
        **{f"val_acc_{k}": v for k, v in mod._best_sub.items()},
        # Stage B.3 selector bake-off: which checkpoint file backs each of the three candidate selectors.
        "selector_ckpts": {
            "best_ema": str(ckpt.best_model_path),
            "best_raw": str(ckpt_raw.best_model_path),
            "swa_plateau": str(out / "swa_plateau.ckpt"),
        },
    }
    dump_json(out / "fold_metrics.json", fold_metrics)
