"""Materialize cached dense v7_native PyG graphs for one tracking split."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracking.common import CACHE_ROOT, DATASET_ROOT, cprint, nano_header
from tracking.data.dataset import LesionDataset
from tracking.data.features import CACHE_TAG
from tracking.data.graph import GraphConfig

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--split", choices=[*SPLITS, "all"], required=True)
    ap.add_argument("--k-intra", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1, help="parallel patients (ProcessPool)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    nano_header("lesion_track_preprocess")

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    stale = list((cache / "processed").glob("*_v5_l0.pt")) if (cache / "processed").is_dir() else []
    if stale:
        cprint(f"[yellow]stale v5_l0 cache ignored (using {CACHE_TAG}): {stale[0].parent}[/yellow]")
        cprint("[dim]delete those files if they confuse you; they are not loaded[/dim]")

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
        cprint(f"built {counts[split]} graphs split={split} -> {cache}/processed/{split}_{CACHE_TAG}.pt")
    if len(splits) > 1:
        parts = ", ".join(f"{s}={counts[s]}" for s in splits)
        cprint(f"done -> {cache}: {parts}")


if __name__ == "__main__":
    main()
