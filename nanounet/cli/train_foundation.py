"""nnFoundationCNN defaults for nanounet_train: weights, optimizer, LR, deep supervision, test-patient guard."""

from __future__ import annotations

import json
import os

from nanounet.model.foundation import sha256_file
from nanounet.plan.dataset.holdout import assert_no_test_patients
from nanounet.plan.dataset.splits import fold_keys, load_splits
from core.ui import cprint

# flag -> (legacy default, foundation default)
DEFAULTS = {"optimizer": ("sgd", "sgd"), "lr": (0.01, 1e-3), "warmup_epochs": (0, 2)}


def check_foundation_flags(args) -> None:
    if args.no_foundation:
        return
    for flag, on in (("--mae-ckpt", args.mae_ckpt), ("--mae-pretrain", args.mae_pretrain)):
        if on:
            raise ValueError(
                f"{flag} was given while nnFoundationCNN weights are the default.\n"
                f"Expected one weight source: nnFoundationCNN, or an MAE backbone, not both.\n"
                f"Fix: add --no-foundation to use {flag}, or drop {flag} to use nnFoundationCNN"
            )


def resolve_foundation(args, plans_path: str) -> str | None:
    """Fill unset flags, record each value's source in args.sources; returns the verified foundation checkpoint or None."""
    with open(plans_path, encoding="utf-8") as f:
        info = json.load(f).get("pretrain_info")
    on = bool(info) and not args.no_foundation
    args.sources = {}
    for name, (legacy, found) in DEFAULTS.items():
        if getattr(args, name) is not None:
            args.sources[name] = "cli"
            continue
        setattr(args, name, found if on else legacy)
        args.sources[name] = "foundation default" if on else "default"
    ds_on = {"on": True, "off": False, "auto": not on}[args.deep_supervision]
    args.enable_ds = ds_on
    args.sources["deep_supervision"] = "cli" if args.deep_supervision != "auto" else ("foundation default" if on else "default")
    if on and args.init_weights:
        cprint("[dim]--init-weights given: nnFoundationCNN weights are not loaded[/dim]")
    if not on or args.init_weights:
        return None
    path = info["checkpoint_path"]
    if not os.path.isfile(path) or sha256_file(path) != info["sha256"]:
        raise SystemExit(
            f"nnFoundationCNN checkpoint {path} is missing or its sha256 differs from the plans' pretrain_info ({info['sha256'][:12]}).\n"
            f"Expected the file nanounet_preprocess downloaded and recorded in {plans_path}.\n"
            f"Fix: export NANOUNET_PRETRAINED=<cache dir> and re-run nanounet_preprocess -d {args.dataset_id}, or pass --no-foundation"
        )
    return path


def guard_test_patients(args, splits_path: str, dj_path: str) -> None:
    """Fail loudly if any train/val case of this fold belongs to a Longitudinal-CT test patient."""
    tr, va = fold_keys(load_splits(splits_path, args.dataset_id, args.plans_identifier), args.fold)
    with open(dj_path, encoding="utf-8") as f:
        n = assert_no_test_patients(sorted(set(tr) | set(va)), json.load(f))
    cprint(f"[green]test-patient guard: 0 of {n} Longitudinal-CT test patients found in {len(set(tr) | set(va))} train/val cases[/green]")
