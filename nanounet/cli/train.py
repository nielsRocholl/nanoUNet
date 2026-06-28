"""Train NanoUNet (Lightning)."""

from __future__ import annotations

from nanounet.dataloader_prefs import init_dataloader_ipc
from nanounet.runtime import set_safe_tmpdir

set_safe_tmpdir()
init_dataloader_ipc()

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
    setup_logging,
)
from nanounet.mem_diag import log_snapshot, mem_diag_enabled, set_mem_diag
from nanounet.runtime import assert_mem_diag_cgroup, runtime_banner

quiet_lightning_runtime()

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nanounet.dataloader_prefs import dataloader_bucket, init_dataloader_ipc
from nanounet.lightning_ckpt import (
    pl_ckpt_assert_epochs_match,
    pl_ckpt_epoch_and_target,
    pl_ckpt_stage_done,
)
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.plans import Plans
from nanounet.plan.splits import fold_seed, parse_fold
from nanounet.pretrain.dataset import build_pretrain_dataloaders
from nanounet.pretrain.module import NanoMAELM
from nanounet.train.data_module import NanoDataModule
from nanounet.train.lightning_module import NanoUNetLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument(
        "-f",
        "--fold",
        type=parse_fold,
        default=0,
        help="Fold index 0-4, or 'all' for full-data training (val=train).",
    )
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
    ap.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Max grad norm; 0 disables. AdamW finetune commonly uses 1.0.",
    )
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
    ap.add_argument(
        "--resume",
        default=None,
        help="Resume supervised from this Lightning ckpt; omit for scratch (no auto last.ckpt).",
    )
    ap.add_argument(
        "--init-weights",
        default=None,
        help="Load full net weights from a supervised Lightning ckpt; fresh optimizer and LR schedule.",
    )
    ap.add_argument(
        "--only-prefix",
        default=None,
        help="Train/val only case keys with this prefix, e.g. d013_ on a merged dataset.",
    )
    ap.add_argument(
        "--longi",
        action="store_true",
        help="Two-stream BL+FU encoder with Difference Weighting at skips (finetune stage 3).",
    )
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "cpu", "cuda", "gpu", "mps"),
        help="Training device. Use mps on Apple Silicon; cuda/gpu on NVIDIA. Default auto.",
    )
    ap.add_argument("--mae-ckpt", default=None, help="Load encoder weights from MAE Lightning checkpoint.")
    ap.add_argument("--mae-pretrain", action="store_true", help="Run MAE in out/mae_pretrain then supervised.")
    ap.add_argument(
        "--mae-resume",
        default=None,
        help="Integrated MAE only: resume or skip-if-done from this ckpt; must match --mae-epochs.",
    )
    ap.add_argument("--mae-epochs", type=int, default=1000)
    ap.add_argument("--mae-lr", type=float, default=1e-2)
    ap.add_argument("--mae-lr-schedule", choices=("cosine_warm_restarts", "poly"), default="cosine_warm_restarts")
    ap.add_argument("--mae-cosine-t0", type=int, default=250)
    ap.add_argument("--mae-cosine-t-mult", type=int, default=1)
    ap.add_argument("--mae-cosine-eta-min", type=float, default=0.0)
    ap.add_argument("--mae-mask-ratio", type=float, default=0.75)
    ap.add_argument("--mae-iters-per-epoch", type=int, default=None, help="Default: same as --iters-per-epoch.")
    ap.add_argument(
        "--dl-bucket",
        choices=("s", "m", "l", "xl"),
        default="m",
        help="DataLoader workers: s=2/1 if TMPDIR off tmpfs else 0, m=4/2, l=8/4, xl=16/8.",
    )
    ap.add_argument(
        "--dl-persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (recommended for long MAE with workers).",
    )
    ap.add_argument(
        "--mem-diag",
        action="store_true",
        help="Log cgroup/process RAM to OUT/mem_diag.jsonl and W&B mem/* metrics.",
    )
    args = ap.parse_args()
    if args.mae_resume and not args.mae_pretrain:
        raise ValueError("--mae-resume requires --mae-pretrain")
    if args.mae_resume and args.mae_ckpt:
        raise ValueError("--mae-resume conflicts with --mae-ckpt")
    if args.init_weights:
        if not os.path.isfile(args.init_weights):
            raise ValueError(args.init_weights)
        if args.resume:
            raise ValueError("--init-weights conflicts with --resume")
        if args.mae_ckpt:
            raise ValueError("--init-weights conflicts with --mae-ckpt")
        if args.mae_pretrain:
            raise ValueError("--init-weights conflicts with --mae-pretrain")
    if args.longi and not args.init_weights:
        raise ValueError("--longi requires --init-weights (warm-start from stage-2 supervised net)")
    args.roi_cfg = resolve_user_config_path(args.roi_cfg)
    set_mem_diag(args.mem_diag)
    setup_logging()
    init_dataloader_ipc()
    dl_b = dataloader_bucket(args.dl_bucket)

    ds = convert_id_to_dataset_name(args.dataset_id)
    nano_header(f"nanoUNet train  {ds}  fold {args.fold}", color="green")
    pp = preprocessed_dir()
    rw = raw_dir()
    plans_path = join(pp, ds, args.plans_identifier + ".json")
    dj_path = join(rw, ds, "dataset.json")
    out = args.out or join(results_dir(), "nanounet", f"{ds}_{args.plans_identifier}_f{args.fold}")
    ckpt_dir = "finetune" if args.init_weights else "checkpoints"
    set_safe_tmpdir(results_tmp=join(out, ".tmp"))
    maybe_mkdir_p(out)
    os.makedirs(join(out, ckpt_dir), exist_ok=True)
    assert_mem_diag_cgroup()
    runtime_banner(join(out, "mae_pretrain") if args.mae_pretrain else out)

    loggers = []
    if not args.no_wandb:
        loggers.append(WandbLogger(project=args.wandb_project, name=args.wandb_name or f"{ds}_f{args.fold}"))
    accel = "cuda" if args.accelerator == "gpu" else args.accelerator

    pm0 = Plans(plans_path)
    batch_mae = args.batch_size if args.batch_size is not None else pm0.get_configuration("3d_fullres").batch_size
    mae_ckpt_arg = args.mae_ckpt
    mae_iters = args.mae_iters_per_epoch if args.mae_iters_per_epoch is not None else args.iters_per_epoch

    pre_out = join(out, "mae_pretrain")
    mae_last = join(pre_out, "checkpoints", "last.ckpt")
    sup_resume = args.resume
    if sup_resume and not os.path.isfile(sup_resume):
        raise ValueError(sup_resume)
    mem_dir = pre_out if mem_diag_enabled() else None

    if args.mae_pretrain and mae_ckpt_arg is None:
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
        mem_diag_dir=out if mem_diag_enabled() else None,
        persistent_workers=args.dl_persistent_workers,
        only_prefix=args.only_prefix,
        longi=args.longi,
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
        mae_ckpt=None if sup_resume or args.init_weights else mae_ckpt_arg,
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


if __name__ == "__main__":
    main()
