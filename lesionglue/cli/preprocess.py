"""Materialize cached dense v5 PyG graphs for one data_split.json split.

Reads Longitudinal_CT_v2/{meta,inputs*,targets*}; writes processed/{split}_v5_{feat}.pt
and meta. MAE mode requires --jobs 1. --resume skips per-patient staging files.

Example (repo root, PYTHONPATH=.):

    python tracking/cli/preprocess.py --split train
    python tracking/cli/preprocess.py --split all --feat mae --jobs 1 --mae-batch 2
    python tracking/cli/preprocess.py --split train --feat l0 --jobs 4 --resume
"""

import argparse
from pathlib import Path

from tracking.common import CACHE_ROOT, DATASET_ROOT, print0
from tracking.data.dataset import LesionDataset
from tracking.data.features import add_feat_args, feat_from_args
from tracking.data.graph import GraphConfig

SPLITS = ("train", "val", "test")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        required=True,
        help="one split, or all for train+val+test",
    )
    ap.add_argument("--k-intra", type=int, default=8)
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel patients (ProcessPool); MAE requires 1 — use SLURM CPUs for intra-patient threads",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip patients already in processed/staging/{split}_v5_{feat}/; merge when split completes",
    )
    add_feat_args(ap)
    args = ap.parse_args()

    feat = feat_from_args(args)
    if feat.mode == "mae":
        assert args.jobs == 1, "MAE feature extraction requires --jobs 1"

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    cfg = GraphConfig(k_intra=args.k_intra, feat=feat)
    splits = SPLITS if args.split == "all" else (args.split,)

    counts = {}
    for split in splits:
        ds = LesionDataset(
            root=str(cache),
            split=split,
            dataset_root=root,
            cfg=cfg,
            feat=feat,
            num_workers=args.jobs,
            resume=args.resume,
        )
        counts[split] = len(ds)
        print0(f"built {counts[split]} graphs split={split} feat={feat.mode} -> {cache}")

    if len(splits) > 1:
        parts = ", ".join(f"{s}={counts[s]}" for s in splits)
        print0(f"done feat={feat.mode} -> {cache}: {parts}")
