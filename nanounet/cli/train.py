"""Train NanoUNet (Lightning)."""

from __future__ import annotations

from nanounet.dataloader_prefs import init_dataloader_ipc
from nanounet.runtime import set_safe_tmpdir

set_safe_tmpdir()
init_dataloader_ipc()

import os

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nanounet.cli.train_parser import build_train_parser, train_config_rows, validate_train_args
from nanounet.common import (
    cprint,
    config_table,
    nano_header,
    preprocessed_dir,
    quiet_lightning_runtime,
    raw_dir,
    results_dir,
    setup_logging,
)
from nanounet.diag import set_mem_diag
from nanounet.runtime import assert_mem_diag_cgroup, runtime_banner

quiet_lightning_runtime()

from pytorch_lightning.loggers import WandbLogger

from nanounet.dataloader_prefs import dataloader_bucket, init_dataloader_ipc
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.plans import Plans
from nanounet.train.fit import run_mae_pretrain, run_supervised


def main() -> None:
    args = build_train_parser().parse_args()
    validate_train_args(args)
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
    sup_resume = args.resume
    if sup_resume and not os.path.isfile(sup_resume):
        raise ValueError(sup_resume)

    config_table(train_config_rows(args, ds, out), title="nanoUNet train")

    mae_ckpt_arg = args.mae_ckpt
    if args.mae_pretrain and args.mae_ckpt is None:
        mae_ckpt_arg = run_mae_pretrain(args, ds, pp, plans_path, dj_path, out, accel, loggers, dl_b, pm0)

    run_supervised(
        args, ds, plans_path, dj_path, out, ckpt_dir, accel, loggers, dl_b,
        None if sup_resume or args.init_weights else mae_ckpt_arg, sup_resume,
    )
    cprint(f"[green]done — checkpoints in {join(out, ckpt_dir)}[/green]")


if __name__ == "__main__":
    main()
