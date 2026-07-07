"""Argparse + validation for nanounet_train."""

from __future__ import annotations

import argparse
import os

from nanounet.common import resolve_user_config_path
from nanounet.plan.splits import parse_fold


def build_train_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("-f", "--fold", type=parse_fold, default=0, help="Fold 0-4 or 'all'.")
    ap.add_argument("--plans", dest="plans_identifier", required=True)
    ap.add_argument("--config", dest="roi_cfg", default="configs/default.json")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=3e-5)
    ap.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    ap.add_argument("--grad-clip", type=float, default=0.0)
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
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--only-prefix", default=None)
    ap.add_argument("--longi", action="store_true")
    ap.add_argument("--longi-null", action="store_true")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "cuda", "gpu", "mps"))
    ap.add_argument("--mae-ckpt", default=None)
    ap.add_argument("--mae-pretrain", action="store_true")
    ap.add_argument("--mae-resume", default=None)
    ap.add_argument("--mae-epochs", type=int, default=1000)
    ap.add_argument("--mae-lr", type=float, default=1e-2)
    ap.add_argument("--mae-lr-schedule", choices=("cosine_warm_restarts", "poly"), default="cosine_warm_restarts")
    ap.add_argument("--mae-cosine-t0", type=int, default=250)
    ap.add_argument("--mae-cosine-t-mult", type=int, default=1)
    ap.add_argument("--mae-cosine-eta-min", type=float, default=0.0)
    ap.add_argument("--mae-mask-ratio", type=float, default=0.75)
    ap.add_argument("--mae-iters-per-epoch", type=int, default=None)
    ap.add_argument("--dl-bucket", choices=("s", "m", "l", "xl"), default="m")
    ap.add_argument("--dl-persistent-workers", action="store_true")
    ap.add_argument("--mem-diag", action="store_true")
    return ap


def validate_train_args(args) -> None:
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
    if args.longi_null and not args.longi:
        raise ValueError("--longi-null requires --longi")
    args.roi_cfg = resolve_user_config_path(args.roi_cfg)


def train_config_rows(args, ds: str, out: str) -> list[tuple[str, object, str]]:
    return [
        ("dataset", ds, "cli"),
        ("fold", args.fold, "cli"),
        ("plans", args.plans_identifier, "cli"),
        ("loss", args.loss, "cli/default"),
        ("optimizer", args.optimizer, "cli/default"),
        ("lr", args.lr, "cli/default"),
        ("epochs", args.epochs, "cli/default"),
        ("batch_size", args.batch_size if args.batch_size is not None else "from plans", "cli/plans"),
        ("precision", args.precision, "cli/default"),
        ("accelerator", args.accelerator, "cli/default"),
        ("dl_bucket", args.dl_bucket, "cli/default"),
        ("mae_pretrain", args.mae_pretrain, "cli"),
        ("longi", args.longi, "cli"),
        ("out", out, "derived"),
    ]
