"""Argparse + validation for nanounet_train."""

from __future__ import annotations

import argparse
import os

from nanounet.common import resolve_user_config_path
from nanounet.plan.splits import parse_fold


def build_train_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True, help="Dataset id, matched against Dataset<NNN>_* under raw/preprocessed/results (zero-padded to 3 digits).")
    ap.add_argument("-f", "--fold", type=parse_fold, default=0, help="Fold 0-4 or 'all'.")
    ap.add_argument("--plans", dest="plans_identifier", required=True, help="Plans identifier: basename of the plans JSON under the preprocessed dataset dir (no .json suffix).")
    ap.add_argument("--config", dest="roi_cfg", default="configs/default.json", help="ROI/prompt config JSON; relative paths are tried under cwd then the repo root.")
    ap.add_argument("--val-manifest", default=None, help="fixed validation manifest from nanounet_build_valset; omit for the legacy per-epoch random val sampling")
    ap.add_argument("--epochs", type=int, default=1000, help="Supervised training epoch budget.")
    ap.add_argument("--lr", type=float, default=0.01, help="Supervised initial learning rate.")
    ap.add_argument("--wd", type=float, default=3e-5, help="Weight decay for the supervised optimizer.")
    ap.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd", help="Supervised optimizer.")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="Max gradient norm for supervised training; 0 disables clipping.")
    ap.add_argument("--batch-size", type=int, default=None, help="Supervised batch size; None takes it from plans 3d_fullres.batch_size.")
    ap.add_argument("--iters-per-epoch", type=int, default=250, help="Supervised training batches per epoch.")
    ap.add_argument("--val-iters", type=int, default=50, help="Supervised validation batches per epoch (ignored when --val-manifest is set).")
    ap.add_argument("--out", default=None, help="Run directory; default <results-env>/nanounet/<Dataset>_<plans>_f<fold>.")
    ap.add_argument("--lr-schedule", choices=("poly", "stretched_tail_poly"), default="poly", help="Supervised LR schedule.")
    ap.add_argument("--stretched-k", type=int, default=750, help="stretched_tail_poly: epoch where the poly curve transitions into the stretched tail.")
    ap.add_argument("--stretched-ref", type=int, default=1000, help="stretched_tail_poly: reference epoch count the poly portion is shaped against.")
    ap.add_argument("--stretched-exp", type=float, default=0.9, help="stretched_tail_poly: poly exponent.")
    ap.add_argument("--warmup-epochs", type=int, default=0, help="Linear LR warmup over the first N epochs, applied to either --lr-schedule. 0 disables it and reproduces the pre-warmup LR curve exactly.")
    ap.add_argument("--ema-decay", type=float, default=0.0, help="Weight EMA decay (e.g. 0.999); logs val_dice_ema next to val_dice. 0 disables EMA.")
    ap.add_argument("--monitor", default="val_dice", help="Metric ModelCheckpoint tracks for best-* checkpoints.")
    ap.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging (a CSVLogger under metrics/ is always on regardless).")
    ap.add_argument("--wandb-project", default="nanounet", help="W&B project name.")
    ap.add_argument("--wandb-name", default=None, help="W&B run name; default <Dataset>_f<fold>.")
    ap.add_argument("--loss", "-loss", choices=("dc_ce", "cc_dc_ce"), default="dc_ce", metavar="MODE", help="Supervised loss: dc_ce (Dice+CE) or cc_dc_ce (adds connected-component term, slower).")
    ap.add_argument("--resume", default=None, help="Resume supervised training from this Lightning ckpt; must sit in a checkpoints/ or finetune/ dir; its recorded num_epochs must match --epochs; omit for a fresh run.")
    ap.add_argument("--init-weights", default=None, help="Load full net weights from this supervised ckpt (fresh optimizer/epoch count); conflicts with --resume, --mae-ckpt, --mae-pretrain.")
    ap.add_argument("--only-prefix", default=None, help="Restrict train/val case keys to those starting with this prefix, e.g. d013_.")
    ap.add_argument("--precision", default="16-mixed", help="Precision passed to the Lightning Trainer.")
    ap.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "cuda", "gpu", "mps"), help="Training device passed to the Lightning Trainer; gpu maps to cuda.")
    ap.add_argument("--mae-ckpt", default=None, help="Load encoder weights from this MAE checkpoint; skips the integrated MAE run even with --mae-pretrain set.")
    ap.add_argument("--mae-pretrain", action="store_true", help="Run an integrated MAE stage under <run>/mae_pretrain/ before supervised training.")
    ap.add_argument("--mae-resume", default=None, help="Resume the integrated MAE stage from this Lightning ckpt; requires --mae-pretrain, conflicts with --mae-ckpt.")
    ap.add_argument("--mae-epochs", type=int, default=1000, help="Integrated MAE stage epoch budget (with --mae-pretrain).")
    ap.add_argument("--mae-lr", type=float, default=1e-2, help="Integrated MAE stage initial learning rate.")
    ap.add_argument("--mae-lr-schedule", choices=("cosine_warm_restarts", "poly"), default="cosine_warm_restarts", help="Integrated MAE stage LR schedule.")
    ap.add_argument("--mae-cosine-t0", type=int, default=250, help="Integrated MAE stage: cosine_warm_restarts restart period (epochs).")
    ap.add_argument("--mae-cosine-t-mult", type=int, default=1, help="Integrated MAE stage: cosine_warm_restarts period multiplier after each restart.")
    ap.add_argument("--mae-cosine-eta-min", type=float, default=0.0, help="Integrated MAE stage: cosine_warm_restarts minimum LR.")
    ap.add_argument("--mae-mask-ratio", type=float, default=0.75, help="Integrated MAE stage: fraction of voxels masked per patch.")
    ap.add_argument("--mae-iters-per-epoch", type=int, default=None, help="Integrated MAE stage training batches per epoch; None uses the same value as --iters-per-epoch.")
    ap.add_argument("--dl-bucket", choices=("s", "m", "l", "xl"), default="m", help="DataLoader worker preset (train/val workers): s=2/1 (0 on tmpfs), m=4/2, l=8/4, xl=16/8.")
    ap.add_argument("--dl-persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs.")
    ap.add_argument("--mem-diag", action="store_true", help="Log cgroup/process RAM to OUT/mem_diag.jsonl.")
    ap.add_argument("--prompts-per-patch", type=int, default=1, help="Independent click draws per patch, all sharing one crop and one augmentation pass. >1 pairs rows for the consistency term; batch_size must be divisible by it.")
    ap.add_argument("--consistency-weight", type=float, default=0.0, help="Lambda max for the two-prompt consistency term, ramped linearly over --consistency-warmup-epochs; 0 disables it. Requires --prompts-per-patch >1.")
    ap.add_argument("--consistency-warmup-epochs", type=int, default=50, help="Epochs to linearly ramp lambda from 0 to --consistency-weight.")
    ap.add_argument("--val-every-n-epochs", type=int, default=1, help="Run validation every N epochs. Dense validation exists to average out resampling noise; a fixed --val-manifest removes that noise at the source, so N=2 with a big manifest beats N=1 with a small random draw at a third of the cost.")
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
    if args.consistency_weight > 0 and args.prompts_per_patch < 2:
        raise ValueError(
            f"--consistency-weight {args.consistency_weight} requires --prompts-per-patch >= 2 "
            f"(got --prompts-per-patch {args.prompts_per_patch}).\n"
            f"Fix: nanounet_train … --prompts-per-patch 2 --consistency-weight {args.consistency_weight}"
        )
    if args.batch_size is not None and args.batch_size % args.prompts_per_patch != 0:
        raise ValueError(
            f"--batch-size {args.batch_size} is not divisible by --prompts-per-patch {args.prompts_per_patch}.\n"
            f"Fix: pick a --batch-size that is a multiple of --prompts-per-patch (e.g. "
            f"{(args.batch_size // args.prompts_per_patch) * args.prompts_per_patch or args.prompts_per_patch})."
        )
    if args.val_manifest and not os.path.isfile(args.val_manifest):
        raise FileNotFoundError(
            f"--val-manifest {args.val_manifest} does not exist.\n"
            f"Fix: nanounet_build_valset -d {args.dataset_id} --plans {args.plans_identifier} "
            f"--config {args.roi_cfg} --out {args.val_manifest}"
        )
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
        ("prompts_per_patch", args.prompts_per_patch, "cli/default"),
        ("consistency_weight", args.consistency_weight, "cli/default"),
        ("warmup_epochs", args.warmup_epochs, "cli/default"),
        ("ema_decay", args.ema_decay, "cli/default"),
        ("monitor", args.monitor, "cli/default"),
        ("val_manifest", args.val_manifest or "legacy random val", "cli/default"),
        ("val_every_n_epochs", args.val_every_n_epochs, "cli/default"),
        ("out", out, "derived"),
    ]
