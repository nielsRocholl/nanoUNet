"""Materialize cached dense v5_l0 PyG graphs for one data_split.json split."""

import argparse
from pathlib import Path

from tracking.common import CACHE_ROOT, DATASET_ROOT, print0
from tracking.data.dataset import LesionDataset
from tracking.data.graph import GraphConfig

SPLITS = ("train", "val", "test")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--split", choices=[*SPLITS, "all"], required=True)
    ap.add_argument("--k-intra", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1, help="parallel patients (ProcessPool)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    cfg = GraphConfig(k_intra=args.k_intra)
    splits = SPLITS if args.split == "all" else (args.split,)

    counts = {}
    for split in splits:
        ds = LesionDataset(
            root=str(cache), split=split, dataset_root=root, cfg=cfg,
            num_workers=args.jobs, resume=args.resume,
        )
        counts[split] = len(ds)
        print0(f"built {counts[split]} graphs split={split} -> {cache}")

    if len(splits) > 1:
        parts = ", ".join(f"{s}={counts[s]}" for s in splits)
        print0(f"done -> {cache}: {parts}")
