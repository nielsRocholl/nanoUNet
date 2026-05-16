"""Materialize cached dense v2 PyG graphs for one data_split.json split.

Reads Longitudinal_CT_v2/{meta,inputs*,targets*}; writes processed/{split}_v2.pt
and {split}_v2_meta.pt (pos_weight for BCE). Run train splits before Trainer fit.

--jobs > 1 uses spawn (macOS default): entry logic lives under ``if __name__ == "__main__"`` so
worker subprocesses (``__mp_main__``) do not recurse into ProcessPoolExecutor again.

Example (repo root, PYTHONPATH=.):

    python tracking/cli/preprocess.py --split train
    python tracking/cli/preprocess.py --split train --jobs 6   # parallel patients (~6× RAM spikes)
"""

import argparse
from pathlib import Path

from tracking.common import CACHE_ROOT, DATASET_ROOT, print0
from tracking.data.dataset import LesionDataset
from tracking.data.graph import GraphConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--k-intra", type=int, default=8)
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel patient builds (ProcessPoolExecutor); each loads full CTs — raise gradually on RAM",
    )
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    cfg = GraphConfig(k_intra=args.k_intra)
    ds = LesionDataset(
        root=str(cache),
        split=args.split,
        dataset_root=Path(args.root),
        cfg=cfg,
        num_workers=args.jobs,
    )
    print0(f"built {len(ds)} graphs split={args.split} -> {cache}")
