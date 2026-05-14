"""Pretrain NanoUNet backbone with CNN-MAE (Lightning)."""

from __future__ import annotations

import argparse
import os
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nanounet.cli.train import _quiet_train_runtime
from nanounet.common import cprint, nano_header, preprocessed_dir, raw_dir, results_dir, sync_nnunet_env
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.pretrain.dataset import build_pretrain_dataloaders
from nanounet.pretrain.module import NanoMAELM


def main() -> None:
    _quiet_train_runtime()
    sync_nnunet_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("-f", "--fold", type=int, default=0)
    ap.add_argument("--plans", dest="plans_identifier", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=3e-5)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--iters-per-epoch", type=int, default=250)
    ap.add_argument("--val-iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="nanounet-mae")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "cpu", "cuda", "gpu", "mps"),
        help="Training device.",
    )
    args = ap.parse_args()

    ds = convert_id_to_dataset_name(args.dataset_id)
    nano_header(f"nanoUNet pretrain MAE  {ds}  fold {args.fold}", color="cyan")
    pp = preprocessed_dir()
    rw = raw_dir()
    plans_path = join(pp, ds, args.plans_identifier + ".json")
    dj_path = join(rw, ds, "dataset.json")
    out = args.out or join(results_dir(), "nanounet", f"{ds}_{args.plans_identifier}_mae_pretrain_f{args.fold}")
    maybe_mkdir_p(out)
    os.makedirs(join(out, "checkpoints"), exist_ok=True)
    shutil.copy(plans_path, join(out, "plans.json"))
    shutil.copy(dj_path, join(out, "dataset.json"))

    from nanounet.plan.plans import Plans

    pm = Plans(plans_path)
    bs = args.batch_size if args.batch_size is not None else pm.get_configuration("3d_fullres").batch_size

    tr_dl, va_dl = build_pretrain_dataloaders(
        ds,
        args.fold,
        args.plans_identifier,
        bs,
        args.iters_per_epoch,
        args.val_iters,
        args.fold + 3000 * args.iters_per_epoch,
        args.fold + 4000,
    )
    lm = NanoMAELM(
        plans_path,
        dj_path,
        out,
        mask_ratio=args.mask_ratio,
        initial_lr=args.lr,
        weight_decay=args.wd,
        num_epochs=args.epochs,
    )
    ck = [
        ModelCheckpoint(dirpath=join(out, "checkpoints"), save_last=True),
        ModelCheckpoint(
            dirpath=join(out, "checkpoints"),
            monitor="val_recon_loss",
            mode="min",
            filename="best-{epoch}-{val_recon_loss:.4f}",
            save_top_k=1,
        ),
    ]
    logs = []
    if not args.no_wandb:
        logs.append(WandbLogger(project=args.wandb_project, name=args.wandb_name or f"{ds}_mae_f{args.fold}"))
    accel = "cuda" if args.accelerator == "gpu" else args.accelerator
    tr = Trainer(
        max_epochs=args.epochs,
        accelerator=accel,
        devices=1,
        precision=args.precision,
        callbacks=ck,
        logger=logs or False,
        default_root_dir=out,
    )
    cprint(f"[dim]MAE pretrain out {out}[/dim]")
    tr.fit(lm, train_dataloaders=tr_dl, val_dataloaders=va_dl, ckpt_path=args.resume)


if __name__ == "__main__":
    main()