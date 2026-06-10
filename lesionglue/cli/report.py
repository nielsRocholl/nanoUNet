"""Train/evaluate matcher and nearest-mask baseline, then write report.json."""

import argparse
import gc
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from tracking.common import CACHE_ROOT, DATASET_ROOT, print0, seed_all
from tracking.config import CKPT_MONITOR, load_config
from tracking.data.features import CACHE_TAG
from tracking.report import eval_baseline, eval_gnn
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import module_from_config


def _accelerator() -> str:
    return "mps" if torch.backends.mps.is_available() else "auto"


def _cuda_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--root", default=str(DATASET_ROOT))
ap.add_argument("--cache", default=str(CACHE_ROOT))
ap.add_argument("--out", required=True)
ap.add_argument("--checkpoint", default="", type=str, help="skip training and only eval from this .ckpt")
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

if __name__ == "__main__":
    args = ap.parse_args()
    cfg = load_config(args.config)
    root, cache, out = Path(args.root), Path(args.cache), Path(args.out)
    for split in ("train", "val", "test"):
        assert (cache / "processed" / f"{split}_{CACHE_TAG}.pt").is_file(), f"missing cached {split} graphs"
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)

    if args.checkpoint.strip():
        best = Path(args.checkpoint.strip()).resolve()
        assert best.is_file(), f"checkpoint not found: {best}"
    else:
        dm = MatcherDataModule(
            cache_root=cache, dataset_root=root, batch_size=cfg.batch_size,
            val_batch_size=cfg.val_batch_size, num_workers=cfg.num_workers,
            fu_jitter_scale=cfg.fu_jitter, p_drop_fu=cfg.p_drop_fu, p_drop_bl=cfg.p_drop_bl, k_intra=cfg.k_intra,
        )
        dm.prepare_data()
        dm.setup()
        mod = module_from_config(cfg)
        ckpt = ModelCheckpoint(dirpath=out / "checkpoints", monitor=CKPT_MONITOR, save_top_k=1, mode="max")
        callbacks = [ckpt, TQDMProgressBar()]
        if not args.no_early_stop:
            callbacks.insert(1, EarlyStopping(monitor=CKPT_MONITOR, mode="max", patience=cfg.early_stop_patience))
        logger = False
        if args.wandb or bool(args.wandb_run_name.strip()):
            if importlib.util.find_spec("wandb") is None:
                raise SystemExit("wandb requested but not installed")
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(project=args.wandb_project, name=(args.wandb_run_name.strip() or None))
        pl.Trainer(
            default_root_dir=out, accelerator=_accelerator(), devices=1, log_every_n_steps=10,
            callbacks=callbacks, logger=logger, max_steps=cfg.max_steps, max_epochs=-1,
            val_check_interval=cfg.val_check_steps, check_val_every_n_epoch=None,
        ).fit(mod, dm)
        best = Path(ckpt.best_model_path)
        assert best.is_file(), "training finished without a best checkpoint"
        del mod, dm
        _cuda_gc()

    print0(f"best checkpoint: {best}")
    prog = not args.quiet
    one_mask = not args.baseline_full_mask_cache
    gnn_val = eval_gnn(best, root, cache, "val", args.eval_batch_size, args.eval_num_workers, not args.no_ema, eval_device_pref=args.eval_device, show_progress=prog)
    _cuda_gc()
    gnn_test = eval_gnn(best, root, cache, "test", args.eval_batch_size, args.eval_num_workers, not args.no_ema, eval_device_pref=args.eval_device, show_progress=prog)
    _cuda_gc()
    bl_val = eval_baseline(root, cache, "val", show_progress=prog, one_mask_cache=one_mask)
    bl_test = eval_baseline(root, cache, "test", show_progress=prog, one_mask_cache=one_mask)
    report = {"best_checkpoint": str(best), "config": asdict(cfg), "gnn": {"val": gnn_val, "test": gnn_test}, "baseline": {"val": bl_val, "test": bl_test}}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print0(f"wrote {out / 'report.json'}")
