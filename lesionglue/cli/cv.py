"""Patient-level k-fold CV: train each fold, aggregate best EMA val_match_score."""

import argparse
import subprocess
import sys
from pathlib import Path

from tracking.common import dump_json, seed_all
from tracking.config import CKPT_MONITOR, load_config
from tracking.data.splits import aggregate_cv_folds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="CV output root; writes fold_*/ subdirs")
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--end-fold", type=int, default=None, help="exclusive upper bound; default n_folds")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lesion-tracking")
    ap.add_argument("--wandb-run-name", default="", type=str, help="suffix; fold index appended")
    args = ap.parse_args()

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
            raise FileNotFoundError(f"missing {metrics_path} after fold {fold}")
        fold_rows.append({"fold": fold, "dir": str(fold_out), **__import__("json").loads(metrics_path.read_text())})

    summary = aggregate_cv_folds(fold_rows)
    summary["monitor"] = CKPT_MONITOR
    summary["n_folds"] = end - args.start_fold
    dump_json(out_root / "cv_summary.json", summary)
