"""Patient-level k-fold CV: train each fold, aggregate best EMA val_match_score."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from core.ui import arg_rows
from lesionglue.common import config_table, cprint, dump_json, nano_header, seed_all
from lesionglue.config import CKPT_MONITOR, load_config
from lesionglue.data.source.splits import aggregate_cv_folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to the lesionglue config JSON; sets n_folds and seed, and is passed to each fold's train run")
    ap.add_argument("--out", required=True, help="CV output root; writes fold_*/ subdirs")
    ap.add_argument("--start-fold", type=int, default=0, help="first fold index to run (inclusive); folds with an existing fold_metrics.json are reused")
    ap.add_argument("--end-fold", type=int, default=None, help="exclusive upper bound; default n_folds")
    ap.add_argument("--wandb", action="store_true", help="log each fold's training run to Weights & Biases")
    ap.add_argument("--wandb-project", default="lesion-tracking", help="W&B project name (used only with --wandb)")
    ap.add_argument("--wandb-run-name", default="", type=str, help="suffix; fold index appended")
    args = ap.parse_args()
    nano_header("lesionglue_cv")
    config_table(arg_rows(ap, args))

    cfg = load_config(args.config)
    end = cfg.n_folds if args.end_fold is None else args.end_fold
    assert 0 <= args.start_fold < end <= cfg.n_folds
    seed_all(cfg.seed)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    train_py = Path(__file__).with_name("train.py")
    fold_rows = []

    for fold in range(args.start_fold, end):
        fold_out = out_root / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        metrics_path = fold_out / "fold_metrics.json"
        if metrics_path.is_file():
            fold_rows.append({"fold": fold, "dir": str(fold_out), **__import__("json").loads(metrics_path.read_text())})
            continue
        cmd = [
            sys.executable, str(train_py),
            "--config", args.config,
            "--out", str(fold_out),
            "--fold", str(fold),
        ]
        if args.wandb:
            cmd.extend(["--wandb", "--wandb-project", args.wandb_project])
            base = args.wandb_run_name.strip() or "cv"
            cmd.extend(["--wandb-run-name", f"{base}_fold{fold}"])
        subprocess.run(cmd, check=True)
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"missing {metrics_path} after fold {fold}\n"
                "Expected lesionglue_train --fold to write fold_metrics.json into the fold dir; the train output above shows why it did not.\n"
                f"Fix: lesionglue_cv --config {args.config} --out {args.out} --start-fold {fold}"
            )
        fold_rows.append({"fold": fold, "dir": str(fold_out), **__import__("json").loads(metrics_path.read_text())})

    summary = aggregate_cv_folds(fold_rows)
    summary["monitor"] = CKPT_MONITOR
    summary["n_folds"] = end - args.start_fold
    dump_json(out_root / "cv_summary.json", summary)
    oofs = " && ".join(
        f"lesionglue_oof --ckpt {shlex.quote(str(out_root / f'fold_{f}' / 'best.ckpt'))} --fold {f} --config {shlex.quote(str(args.config))} --out {shlex.quote(str(out_root / f'fold_{f}' / 'oof_best'))}"
        for f in range(args.start_fold, end)
    )
    cprint(f"next: {oofs} && lesionglue_pool --runs {shlex.quote(str(out_root))} --out {shlex.quote(str(out_root / 'pool'))}", markup=False, soft_wrap=True)


if __name__ == "__main__":
    main()
