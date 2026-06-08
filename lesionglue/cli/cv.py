"""Patient-level k-fold CV: train each fold, aggregate best EMA val_match_score."""

import argparse
import importlib.util
import shlex
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from tracking.cli.train import ARG_FIELDS, CKPT_MONITOR, TrainConfig
from tracking.common import CACHE_ROOT, DATASET_ROOT, dump_json, seed_all
from tracking.data.features import add_feat_args, feat_from_args
from tracking.data.splits import aggregate_cv_folds


@dataclass
class CvConfig(TrainConfig):
    start_fold: int = 0
    end_fold: int | None = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--out", required=True, help="CV output root; writes fold_*/ subdirs")
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--end-fold", type=int, default=None, help="exclusive upper bound; default n_folds")
    for name, typ in ARG_FIELDS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=typ, default=getattr(TrainConfig, name))
    ap.add_argument("--nce-scope", choices=("graph", "batch"), default=TrainConfig.nce_scope)
    ap.add_argument("--single-pos-fu", action="store_true")
    ap.add_argument("--desc-norm", action="store_true")
    ap.add_argument("--no-desc-norm", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--dust-no-pair-summary", action="store_true")
    ap.add_argument("--dust-legacy-linear", action="store_true")
    ap.add_argument("--edge-cross-attn", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lesion-tracking")
    ap.add_argument("--wandb-run-name", default="", type=str, help="suffix; fold index appended")
    ap.add_argument("--extra-train-args", default="", type=str, help="extra flags forwarded to train.py per fold")
    add_feat_args(ap)
    args = ap.parse_args()

    feat = feat_from_args(args)
    cfg = CvConfig()
    for f in fields(TrainConfig):
        if hasattr(args, f.name):
            setattr(cfg, f.name, getattr(args, f.name))
    cfg.start_fold = args.start_fold
    cfg.end_fold = args.end_fold
    end = cfg.n_folds if cfg.end_fold is None else cfg.end_fold
    assert 0 <= cfg.start_fold < end <= cfg.n_folds
    seed_all(cfg.seed)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    train_py = Path(__file__).with_name("train.py")
    extra = shlex.split(args.extra_train_args) if args.extra_train_args.strip() else []
    fold_rows = []

    for fold in range(cfg.start_fold, end):
        fold_out = out_root / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        metrics_path = fold_out / "fold_metrics.json"
        if metrics_path.is_file():  # resume after wall-time kill: skip finished folds
            fold_rows.append({"fold": fold, "dir": str(fold_out), **__import__("json").loads(metrics_path.read_text())})
            continue
        cmd = [
            sys.executable, str(train_py),
            "--root", str(args.root), "--cache", str(args.cache), "--out", str(fold_out),
            "--fold", str(fold), "--n-folds", str(cfg.n_folds), "--cv-seed", str(cfg.cv_seed),
            "--seed", str(cfg.seed),
        ]
        cmd.extend(["--feat", feat.mode])
        if feat.mode == "mae":
            cmd.extend([
                "--mae-ckpt", feat.mae_ckpt, "--mae-plans", feat.mae_plans,
                "--mae-skip", str(feat.mae_skip), "--mae-batch", str(feat.mae_batch),
            ])
        for name, typ in ARG_FIELDS:
            val = getattr(cfg, name)
            if val is None:
                continue
            flag = f"--{name.replace('_', '-')}"
            if typ is bool:
                if val:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(val)])
        cmd.append(f"--nce-scope={cfg.nce_scope}")
        if args.single_pos_fu:
            cmd.append("--single-pos-fu")
        if args.desc_norm:
            cmd.append("--desc-norm")
        if args.no_desc_norm:
            cmd.append("--no-desc-norm")
        if args.no_early_stop:
            cmd.append("--no-early-stop")
        if args.dust_no_pair_summary:
            cmd.append("--dust-no-pair-summary")
        if args.dust_legacy_linear:
            cmd.append("--dust-legacy-linear")
        if args.edge_cross_attn:
            cmd.append("--edge-cross-attn")
        if args.wandb:
            cmd.extend(["--wandb", "--wandb-project", args.wandb_project])
            base = args.wandb_run_name.strip() or "cv"
            cmd.extend(["--wandb-run-name", f"{base}_fold{fold}"])
        cmd.extend(extra)
        subprocess.run(cmd, check=True)
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing {metrics_path} after fold {fold}")
        fold_rows.append({"fold": fold, "dir": str(fold_out), **__import__("json").loads(metrics_path.read_text())})

    summary = aggregate_cv_folds(fold_rows)
    summary["monitor"] = CKPT_MONITOR
    summary["n_folds"] = end - cfg.start_fold
    dump_json(out_root / "cv_summary.json", summary)
