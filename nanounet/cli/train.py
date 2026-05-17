"""Train NanoUNet (Lightning)."""

from __future__ import annotations

import argparse
import os
import shutil

import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nanounet.common import (
    cprint,
    nano_header,
    preprocessed_dir,
    quiet_lightning_runtime,
    raw_dir,
    resolve_user_config_path,
    results_dir,
    sync_nnunet_env,
)

quiet_lightning_runtime()

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.plans import Plans
from nanounet.pretrain.dataset import build_pretrain_dataloaders
from nanounet.pretrain.module import NanoMAELM
from nanounet.train.data_module import NanoDataModule
from nanounet.train.lightning_module import NanoUNetLM


def main() -> None:
    sync_nnunet_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("-f", "--fold", type=int, default=0)
    ap.add_argument("--plans", dest="plans_identifier", required=True)
    ap.add_argument(
        "--config",
        dest="roi_cfg",
        default="configs/default.json",
        help="ROI/prompt JSON; relative path tries cwd then nanoUNet repo root.",
    )
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--iters-per-epoch", type=int, default=250)
    ap.add_argument("--val-iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lr-schedule", choices=("poly", "stretched_tail_poly"), default="poly")
    ap.add_argument("--stretched-k", type=int, default=750)
    ap.add_argument("--stretched-ref", type=int, default=1000)
    ap.add_argument("--stretched-exp", type=float, default=0.9)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="nanounet")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--loss", "-loss", choices=("dc_ce", "cc_dc_ce"), default="dc_ce", metavar="MODE")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "cpu", "cuda", "gpu", "mps"),
        help="Training device. Use mps on Apple Silicon; cuda/gpu on NVIDIA. Default auto.",
    )
    ap.add_argument("--mae-ckpt", default=None, help="Load encoder weights from MAE Lightning checkpoint.")
    ap.add_argument("--mae-pretrain", action="store_true", help="Run MAE in out/mae_pretrain then supervised.")
    ap.add_argument("--mae-epochs", type=int, default=1000)
    ap.add_argument("--mae-lr", type=float, default=1e-2)
    ap.add_argument("--mae-mask-ratio", type=float, default=0.75)
    ap.add_argument("--mae-iters-per-epoch", type=int, default=None, help="Default: same as --iters-per-epoch.")
    args = ap.parse_args()
    args.roi_cfg = resolve_user_config_path(args.roi_cfg)

    ds = convert_id_to_dataset_name(args.dataset_id)
    nano_header(f"nanoUNet train  {ds}  fold {args.fold}", color="green")
    pp = preprocessed_dir()
    rw = raw_dir()
    plans_path = join(pp, ds, args.plans_identifier + ".json")
    dj_path = join(rw, ds, "dataset.json")
    out = args.out or join(results_dir(), "nanounet", f"{ds}_{args.plans_identifier}_f{args.fold}")
    maybe_mkdir_p(out)
    os.makedirs(join(out, "checkpoints"), exist_ok=True)

    loggers = []
    if not args.no_wandb:
        loggers.append(WandbLogger(project=args.wandb_project, name=args.wandb_name or f"{ds}_f{args.fold}"))
    accel = "cuda" if args.accelerator == "gpu" else args.accelerator

    pm0 = Plans(plans_path)
    batch_mae = args.batch_size if args.batch_size is not None else pm0.get_configuration("3d_fullres").batch_size
    mae_ckpt_arg = args.mae_ckpt
    mae_iters = args.mae_iters_per_epoch if args.mae_iters_per_epoch is not None else args.iters_per_epoch

    if args.mae_pretrain and mae_ckpt_arg is None and args.resume is None:
        pre_out = join(out, "mae_pretrain")
        maybe_mkdir_p(pre_out)
        os.makedirs(join(pre_out, "checkpoints"), exist_ok=True)
        shutil.copyfile(plans_path, join(pre_out, "plans.json"))
        shutil.copyfile(dj_path, join(pre_out, "dataset.json"))
        tr_pre, va_pre = build_pretrain_dataloaders(
            ds,
            args.fold,
            args.plans_identifier,
            batch_mae,
            mae_iters,
            args.val_iters,
            args.fold + 5000 * mae_iters,
            args.fold + 6000,
        )
        pre_lm = NanoMAELM(
            plans_path,
            dj_path,
            pre_out,
            mask_ratio=args.mae_mask_ratio,
            initial_lr=args.mae_lr,
            weight_decay=args.wd,
            num_epochs=args.mae_epochs,
        )
        pre_cb = [
            ModelCheckpoint(dirpath=join(pre_out, "checkpoints"), save_last=True),
            ModelCheckpoint(
                dirpath=join(pre_out, "checkpoints"),
                monitor="val_recon_loss",
                mode="min",
                filename="best-{epoch}-{val_recon_loss:.4f}",
                save_top_k=1,
            ),
        ]
        pre_trnr = Trainer(
            max_epochs=args.mae_epochs,
            accelerator=accel,
            devices=1,
            precision=args.precision,
            callbacks=pre_cb,
            logger=loggers or False,
            default_root_dir=pre_out,
        )
        pre_trnr.fit(pre_lm, train_dataloaders=tr_pre, val_dataloaders=va_pre)
        mae_ckpt_arg = join(pre_out, "checkpoints", "last.ckpt")
        del pre_lm, tr_pre, va_pre, pre_trnr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dm = NanoDataModule(
        ds,
        args.fold,
        args.plans_identifier,
        args.roi_cfg,
        batch_size=args.batch_size,
        num_iterations_per_epoch=args.iters_per_epoch,
        num_val_iterations=args.val_iters,
    )
    lm = NanoUNetLM(
        plans_path,
        dj_path,
        args.roi_cfg,
        out,
        initial_lr=args.lr,
        weight_decay=args.wd,
        num_epochs=args.epochs,
        lr_schedule=args.lr_schedule,
        stretched_k=args.stretched_k,
        stretched_ref=args.stretched_ref,
        stretched_exp=args.stretched_exp,
        loss_type=args.loss,
        mae_ckpt=None if args.resume else mae_ckpt_arg,
    )
    cb = [
        ModelCheckpoint(
            dirpath=join(out, "checkpoints"),
            filename="best-{epoch}-{val_dice:.4f}",
            monitor="val_dice",
            mode="max",
            save_top_k=2,
        ),
        ModelCheckpoint(dirpath=join(out, "checkpoints"), save_last=True),
    ]
    tr = Trainer(
        max_epochs=args.epochs,
        accelerator=accel,
        devices=1,
        precision=args.precision,
        callbacks=cb,
        logger=loggers or False,
        default_root_dir=out,
    )
    cprint(f"[dim]out {out}[/dim]")
    tr.fit(lm, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
