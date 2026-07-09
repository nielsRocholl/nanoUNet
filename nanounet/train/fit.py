"""Fit orchestration for nanounet_train: integrated MAE pretrain stage + supervised stage."""

from __future__ import annotations

import os
import shutil

import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from nanounet.common import cprint
from nanounet.diag import log_snapshot, mem_diag_enabled
from nanounet.lightning_ckpt import (
    pl_ckpt_assert_epochs_match,
    pl_ckpt_epoch_and_target,
    pl_ckpt_stage_done,
)
from nanounet.plan.plans import Plans
from nanounet.plan.splits import fold_seed
from nanounet.pretrain.dataset import build_pretrain_dataloaders
from nanounet.pretrain.module import NanoMAELM
from nanounet.train.data_module import NanoDataModule
from nanounet.train.lightning_module import NanoUNetLM


def run_mae_pretrain(args, ds, pp, plans_path, dj_path, out, accel, loggers, dl_b, pm0) -> str | None:
    """Integrated MAE stage. Returns the mae checkpoint path to transfer, or None."""
    batch_mae = args.batch_size if args.batch_size is not None else pm0.get_configuration("3d_fullres").batch_size
    mae_ckpt_arg = args.mae_ckpt
    mae_iters = args.mae_iters_per_epoch if args.mae_iters_per_epoch is not None else args.iters_per_epoch

    pre_out = join(out, "mae_pretrain")
    mae_last = join(pre_out, "checkpoints", "last.ckpt")
    mem_dir = pre_out if mem_diag_enabled() else None

    maybe_mkdir_p(pre_out)
    os.makedirs(join(pre_out, "checkpoints"), exist_ok=True)
    shutil.copyfile(plans_path, join(pre_out, "plans.json"))
    shutil.copyfile(dj_path, join(pre_out, "dataset.json"))
    run_mae_fit = True
    mae_fit_ckpt: str | None = None
    if args.mae_resume:
        if not os.path.isfile(args.mae_resume):
            raise ValueError(args.mae_resume)
        pl_ckpt_assert_epochs_match(args.mae_resume, args.mae_epochs)
        ep_m, tgt_m = pl_ckpt_epoch_and_target(args.mae_resume)
        if pl_ckpt_stage_done(ep_m, tgt_m):
            mae_ckpt_arg = args.mae_resume
            run_mae_fit = False
        else:
            mae_fit_ckpt = args.mae_resume
    if run_mae_fit:
        if mem_diag_enabled():
            log_snapshot(
                "mae_config",
                pre_out,
                extra={
                    "batch_size": batch_mae,
                    "patch_size": list(pm0.get_configuration("3d_fullres").patch_size),
                    "dl_bucket": args.dl_bucket,
                    "nw_train": dl_b.nw_train,
                    "nw_val": dl_b.nw_val,
                    "prefetch_train": dl_b.prefetch_train,
                    "prefetch_val": dl_b.prefetch_val,
                    "iters_per_epoch": mae_iters,
                    "val_iters": args.val_iters,
                    "preprocessed_dir": pp,
                    "mae_resume": mae_fit_ckpt,
                    "persistent_workers": args.dl_persistent_workers,
                },
            )
        tr_pre, va_pre = build_pretrain_dataloaders(
            ds,
            args.fold,
            args.plans_identifier,
            batch_mae,
            mae_iters,
            args.val_iters,
            fold_seed(args.fold) + 5000 * mae_iters,
            fold_seed(args.fold) + 6000,
            dl_b,
            mem_diag_dir=mem_dir,
            persistent_workers=args.dl_persistent_workers,
        )
        if mem_diag_enabled():
            log_snapshot("mae_dataloaders_built", pre_out)
        pre_lm = NanoMAELM(
            plans_path,
            dj_path,
            pre_out,
            mask_ratio=args.mae_mask_ratio,
            initial_lr=args.mae_lr,
            weight_decay=args.wd,
            num_epochs=args.mae_epochs,
            lr_schedule=args.mae_lr_schedule,
            cosine_t0=args.mae_cosine_t0,
            cosine_t_mult=args.mae_cosine_t_mult,
            cosine_eta_min=args.mae_cosine_eta_min,
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
        pre_trnr.fit(pre_lm, train_dataloaders=tr_pre, val_dataloaders=va_pre, ckpt_path=mae_fit_ckpt)
        mae_ckpt_arg = mae_last
        del pre_lm, tr_pre, va_pre, pre_trnr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if mem_diag_enabled():
            log_snapshot("mae_teardown", pre_out)
    return mae_ckpt_arg


def run_supervised(
    args, ds, plans_path, dj_path, out, ckpt_dir, accel, loggers, dl_b, mae_ckpt_arg, sup_resume
) -> None:
    """Supervised stage."""
    if sup_resume:
        pl_ckpt_assert_epochs_match(sup_resume, args.epochs)
        ep_s, tgt_s = pl_ckpt_epoch_and_target(sup_resume)
        if pl_ckpt_stage_done(ep_s, tgt_s):
            cprint("[dim]supervised training already reached num_epochs; nothing to do.[/dim]")
            return

    dm = NanoDataModule(
        ds,
        args.fold,
        args.plans_identifier,
        args.roi_cfg,
        dl_b,
        args.batch_size,
        args.iters_per_epoch,
        args.val_iters,
        persistent_workers=args.dl_persistent_workers,
        only_prefix=args.only_prefix,
        longi=args.longi,
        longi_null=args.longi_null,
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
        optimizer=args.optimizer,
        mae_ckpt=mae_ckpt_arg,
        init_weights=args.init_weights,
        longi=args.longi,
    )
    # finetune optimizes hard small lesions + FP suppression; select on macro Dice, not the
    # big-lesion-dominated global val_dice that the base run uses.
    mon = "val_dice_macro" if args.init_weights else "val_dice"
    cb = [
        ModelCheckpoint(
            dirpath=join(out, ckpt_dir),
            filename=f"best-{{epoch}}-{{{mon}:.4f}}",
            monitor=mon,
            mode="max",
            save_top_k=2,
        ),
        ModelCheckpoint(dirpath=join(out, ckpt_dir), save_last=True),
    ]
    tr = Trainer(
        max_epochs=args.epochs,
        accelerator=accel,
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip or None,
        callbacks=cb,
        logger=loggers or False,
        default_root_dir=out,
    )
    cprint(f"[dim]out {out}[/dim]")
    tr.fit(lm, datamodule=dm, ckpt_path=sup_resume)
