"""Train bipartite lesion matcher with step-clock Lightning."""

from __future__ import annotations

import argparse
import shlex
import importlib.util
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from core.ui import arg_rows
from lesionglue.common import CACHE_ROOT, DATASET_ROOT, HOLDOUT_CSV, config_table, cprint, dump_json, nano_header, seed_all
from lesionglue.config import CKPT_MONITOR, dump_config, load_config
from lesionglue.data.graph.dense import graph_config
from lesionglue.data.source.splits import POOLS
from lesionglue.train.datamodule import MatcherDataModule
from lesionglue.train.module import module_from_config


def _accelerator() -> str:
    return "mps" if torch.backends.mps.is_available() else "auto"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config (lesionglue/configs/base.json)")
    ap.add_argument("--root", default=str(DATASET_ROOT), help="dataset root directory read by the datamodule")
    ap.add_argument("--cache", default=str(CACHE_ROOT), help="root of the cached lesion graphs")
    ap.add_argument("--out", default="lightning_logs", help="run dir for checkpoints, config.json, fold_metrics.json; refuses to run if it holds *.ckpt")
    ap.add_argument("--pool", choices=POOLS, default="train-val", help="patients that form the fit set and the CV folds: train-val = the 240 train+val patients; all = train+val+test caches, all 300 patients (the holdout is then in the pool)")
    ap.add_argument("--fold", type=int, default=None, help="CV fold in [0, n_folds) held out for validation; None = no-val fit on the whole pool, saves last.ckpt only")
    ap.add_argument("--no-val", action="store_true", help="with --fold: hold that fold out of the fit set but never validate, early-stop or select on it; fit for max_steps and save last.ckpt only (already the case without --fold)")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed; None = use the seed from --config")
    ap.add_argument("--max-steps", type=int, default=None, help="override config max_steps (optimizer steps); None = use the value from --config")
    ap.add_argument("--no-early-stop", action="store_true", help="disable EarlyStopping on the checkpoint metric (only applies with --fold and without --no-val)")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases (needs the wandb package; also on when --wandb-run-name is set)")
    ap.add_argument("--wandb-project", default="lesion-tracking", help="W&B project name, used when W&B logging is on")
    ap.add_argument("--wandb-run-name", default="", type=str, help="W&B run name; a non-blank value also turns on W&B logging, empty = W&B auto-names")
    args = ap.parse_args()
    nano_header("lesionglue_train")
    config_table(arg_rows(ap, args))

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.max_steps is not None:
        cfg.max_steps = int(args.max_steps)
    if args.fold is not None:
        assert args.fold in range(cfg.n_folds)
    seed_all(cfg.seed)
    pool_flag = "" if args.pool == "train-val" else f" --pool {args.pool}"
    no_val = args.fold is None or args.no_val  # no val loop, no EarlyStopping, no selection: last.ckpt only

    out = Path(args.out)
    existing = sorted(out.glob("*.ckpt")) if out.is_dir() else []
    if existing:
        raise SystemExit(
            f"Refusing to overwrite checkpoints in {out}: {[p.name for p in existing]}.\n"
            f"Expected an empty run directory.\n"
            f"Fix: --out /nnunet_data/lesion_tracking/runs/r13_one_retrain/final_seed0"
        )
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out / "config.json")

    dm = MatcherDataModule(
        cache_root=Path(args.cache), dataset_root=Path(args.root),
        batch_size=cfg.batch_size, val_batch_size=cfg.val_batch_size, num_workers=cfg.num_workers,
        fu_jitter_scale=cfg.fu_jitter, p_drop_fu=cfg.p_drop_fu, p_drop_bl=cfg.p_drop_bl,
        graph=graph_config(cfg), fold=args.fold, n_folds=cfg.n_folds, cv_seed=cfg.cv_seed, pool=args.pool, no_val=no_val,
    )
    dm.prepare_data()
    dm.setup()
    n_val = 0 if getattr(dm, "val_ds", None) is None else len(dm.val_ds)
    note = f"holdout {HOLDOUT_CSV.name} is not used for selection" if args.pool == "train-val" else "pool=all: the holdout patients are inside the pool"
    held = f" held_out={dm.n_held_out} (fold {args.fold}, never validated)" if args.fold is not None and no_val else ""
    cprint(f"fit={len(dm.train_ds)} val={n_val}{held} ({note})")
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
    if no_val:
        ckpt = ModelCheckpoint(
            dirpath=str(out), save_top_k=0, save_last=True,
            filename="last", auto_insert_metric_name=False, enable_version_counter=False,
        )
        callbacks = [ckpt]
    else:
        callbacks = [ckpt, ckpt_raw]
        if not args.no_early_stop:
            callbacks.insert(1, EarlyStopping(monitor=CKPT_MONITOR, mode="max", patience=cfg.early_stop_patience))

    logger = False
    if args.wandb or bool(args.wandb_run_name.strip()):
        if importlib.util.find_spec("wandb") is None:
            raise SystemExit(
                "wandb requested but not installed.\n"
                "Expected the wandb package for --wandb.\n"
                "Fix: pip install 'wandb>=0.12.10'"
            )
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=(args.wandb_run_name.strip() or None))

    trainer_kw = dict(
        default_root_dir=args.out, accelerator=_accelerator(), devices=1, log_every_n_steps=10,
        callbacks=callbacks, logger=logger, max_steps=cfg.max_steps, max_epochs=-1,
        val_check_interval=cfg.val_check_steps, check_val_every_n_epoch=None,
        enable_progress_bar=False,
    )
    if no_val:
        trainer_kw.update(limit_val_batches=0, num_sanity_val_steps=0, val_check_interval=None)
    trainer = pl.Trainer(**trainer_kw)
    trainer.fit(mod, dm)

    last = out / "last.ckpt"
    if no_val:
        trainer.save_checkpoint(str(last))
        if not last.is_file():
            raise SystemExit(
                f"No last.ckpt at {last} after no-val fit.\n"
                f"Expected trainer.save_checkpoint to write the final weights.\n"
                f"Fix: lesionglue_train --config lesionglue/configs/complete.json --out {out}"
            )
        fold_metrics = {
            "fold": args.fold, "val_disabled": True, "selector": "last", "best_ckpt": str(last),
            "n_fit": len(dm.train_ds), "max_steps": cfg.max_steps, "seed": cfg.seed,
            "selector_ckpts": {"last": str(last)},
        }
        if args.fold is not None:
            fold_metrics["n_held_out"] = dm.n_held_out
        if args.pool != "train-val":
            fold_metrics["pool"] = args.pool
        dump_json(out / "fold_metrics.json", fold_metrics)
        cprint(f"wrote {last}")
        if args.fold is not None:
            cprint(f"next: lesionglue_oof --ckpt {shlex.quote(str(last))} --fold {args.fold} --config {shlex.quote(str(args.config))} --out {shlex.quote(str(out / 'oof_last'))}{pool_flag}", markup=False, soft_wrap=True)
            return
        cprint(f"next: lesionglue_eval --ckpt {shlex.quote(str(last))} --split test", markup=False, soft_wrap=True)
        return
    fold_metrics = {
        "fold": args.fold,
        "best_ckpt": str(ckpt.best_model_path or last),
        "val_match_score_ema": mod._best_ema_score,
        "val_match_score_raw": mod._best_raw_score,
        "val_match_score_peak": mod._val_score_peak,
        **{f"val_acc_{k}": v for k, v in mod._best_sub.items()},
        "selector_ckpts": {
            "best_ema": str(ckpt.best_model_path or last),
            "best_raw": str(ckpt_raw.best_model_path),
            "last": str(last),
            "swa_plateau": str(out / "swa_plateau.ckpt"),
        },
    }
    if args.pool != "train-val":
        fold_metrics["pool"] = args.pool
    dump_json(out / "fold_metrics.json", fold_metrics)
    cprint(f"wrote {ckpt.best_model_path}")
    cprint(f"next: lesionglue_oof --ckpt {shlex.quote(str(ckpt.best_model_path or last))} --fold {args.fold} --config {shlex.quote(str(args.config))} --out {shlex.quote(str(out / 'oof_best'))}{pool_flag}", markup=False, soft_wrap=True)


if __name__ == "__main__":
    main()
