"""Pretrain NanoUNet backbone with CNN-MAE (Lightning)."""

from __future__ import annotations

from nanounet.data.loader.prefs import init_dataloader_ipc
from nanounet.runtime import set_safe_tmpdir

set_safe_tmpdir()
init_dataloader_ipc()

import argparse
import os
import shlex
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from core.ui import arg_rows, config_table, cprint, nano_header
from nanounet.common import preprocessed_dir, quiet_lightning_runtime, raw_dir, results_dir

quiet_lightning_runtime()

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nanounet.data.loader.prefs import dataloader_bucket
from nanounet.lightning_ckpt import (
    pl_ckpt_epoch_and_target,
    pl_ckpt_stage_done,
)
from nanounet.plan.dataset.ids import convert_id_to_dataset_name
from nanounet.plan.dataset.splits import fold_seed, parse_fold
from nanounet.diag import set_mem_diag
from nanounet.pretrain.dataset import build_pretrain_dataloaders
from nanounet.pretrain.module import NanoMAELM
from nanounet.runtime import assert_mem_diag_cgroup, runtime_banner


def main() -> None:
    ap = argparse.ArgumentParser()
    # nanochat-style: allow U8 (legacy snake flag; cluster scripts pass it)
    ap.add_argument("-d", "--dataset_id", type=int, required=True, help="Dataset id, matched against Dataset<NNN>_* under raw/preprocessed/results (zero-padded to 3 digits).")
    ap.add_argument(
        "-f",
        "--fold",
        type=parse_fold,
        default=0,
        help="Fold index 0-4, or 'all' for full-data training (val=train).",
    )
    ap.add_argument("--plans", dest="plans_identifier", required=True, help="Plans identifier: basename of the plans JSON under the preprocessed dataset dir (no .json suffix).")
    ap.add_argument("--epochs", type=int, default=1000, help="MAE pretrain epoch budget.")
    ap.add_argument("--lr", type=float, default=1e-2, help="MAE initial learning rate.")
    ap.add_argument("--lr-schedule", choices=("cosine_warm_restarts", "poly"), default="cosine_warm_restarts", help="MAE LR schedule.")
    ap.add_argument("--cosine-t0", type=int, default=250, help="cosine_warm_restarts restart period (epochs).")
    ap.add_argument("--cosine-t-mult", type=int, default=1, help="cosine_warm_restarts period multiplier after each restart.")
    ap.add_argument("--cosine-eta-min", type=float, default=0.0, help="cosine_warm_restarts minimum LR.")
    ap.add_argument("--wd", type=float, default=3e-5, help="Weight decay.")
    ap.add_argument("--mask-ratio", type=float, default=0.75, help="Fraction of voxels masked per patch.")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size; None takes it from plans 3d_fullres.batch_size.")
    ap.add_argument("--iters-per-epoch", type=int, default=250, help="Training batches per epoch.")
    ap.add_argument("--val-iters", type=int, default=50, help="Validation batches per epoch.")
    ap.add_argument("--out", default=None, help="Output directory; default <results-env>/nanounet/<Dataset>_<plans>_mae_pretrain_f<fold>.")
    ap.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    ap.add_argument("--wandb-project", default="nanounet-mae", help="W&B project name.")
    ap.add_argument("--wandb-name", default=None, help="W&B run name; default <Dataset>_mae_f<fold>.")
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
        "--resume",
        default=None,
        help="Resume MAE from this Lightning ckpt; omit for a fresh run (no auto last.ckpt).",
    )
    ap.add_argument("--precision", default="16-mixed", help="Precision passed to the Lightning Trainer.")
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "cpu", "cuda", "gpu", "mps"),
        help="Training device.",
    )
    ap.add_argument(
        "--mem-diag",
        action="store_true",
        help="Log cgroup/process RAM to OUT/mem_diag.jsonl.",
    )
    args = ap.parse_args()
    set_mem_diag(args.mem_diag)

    dl_b = dataloader_bucket(args.dl_bucket)
    ds = convert_id_to_dataset_name(args.dataset_id)
    nano_header(f"nanoUNet pretrain MAE  {ds}  fold {args.fold}", color="cyan")
    config_table(arg_rows(ap, args))
    pp = preprocessed_dir()
    rw = raw_dir()
    plans_path = join(pp, ds, args.plans_identifier + ".json")
    dj_path = join(rw, ds, "dataset.json")
    out = args.out or join(results_dir(), "nanounet", f"{ds}_{args.plans_identifier}_mae_pretrain_f{args.fold}")
    set_safe_tmpdir(results_tmp=join(out, ".tmp"))
    maybe_mkdir_p(out)
    os.makedirs(join(out, "checkpoints"), exist_ok=True)
    assert_mem_diag_cgroup()
    runtime_banner(out)
    shutil.copyfile(plans_path, join(out, "plans.json"))
    shutil.copyfile(dj_path, join(out, "dataset.json"))

    next_cmd = (
        f"nanounet_train -d {args.dataset_id} -f {args.fold} --plans {shlex.quote(args.plans_identifier)} "
        f"--mae-ckpt {shlex.quote(join(out, 'checkpoints', 'last.ckpt'))}"
    )
    ckpt = args.resume
    if ckpt:
        if not os.path.isfile(ckpt):
            raise ValueError(
                f"--resume {ckpt} does not exist.\n"
                f"Expected a Lightning checkpoint written by a previous nanounet_pretrain run.\n"
                f"Fix: pass an existing --resume path, or drop --resume to start a fresh MAE run   (see nanounet/docs/steps/pretrain.md)"
            )
        ep0, tgt0 = pl_ckpt_epoch_and_target(ckpt)
        if pl_ckpt_stage_done(ep0, tgt0):
            cprint("[dim]MAE pretrain already reached num_epochs; nothing to do.[/dim]")
            cprint(f"next: {next_cmd}", markup=False, soft_wrap=True)
            return

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
        fold_seed(args.fold) + 3000 * args.iters_per_epoch,
        fold_seed(args.fold) + 4000,
        dl_b,
        persistent_workers=args.dl_persistent_workers,
    )
    lm = NanoMAELM(
        plans_path,
        dj_path,
        out,
        mask_ratio=args.mask_ratio,
        initial_lr=args.lr,
        weight_decay=args.wd,
        num_epochs=args.epochs,
        lr_schedule=args.lr_schedule,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
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
    tr.fit(lm, train_dataloaders=tr_dl, val_dataloaders=va_dl, ckpt_path=ckpt)
    cprint(f"next: {next_cmd}", markup=False, soft_wrap=True)


if __name__ == "__main__":
    main()