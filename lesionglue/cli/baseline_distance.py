"""Distance-only baseline on cached graphs: AP/AUROC using score = -dist_mm per cross edge."""

import argparse
import shlex
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from core.ui import arg_rows
from lesionglue.common import CACHE_ROOT, DATASET_ROOT, config_table, cprint, nano_header
from lesionglue.data.cache.dataset import LesionDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_ROOT), help="cached graph root (output of lesionglue_preprocess)")
    ap.add_argument("--root", default=str(DATASET_ROOT), help="Longitudinal-CT dataset root passed to the graph dataset")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="which cached split to score")
    ap.add_argument("--batch-size", type=int, default=8, help="graphs per batch while scoring")
    args = ap.parse_args()
    nano_header(f"LesionGlue baseline_distance  {args.split}")
    config_table(arg_rows(ap, args))
    ds = LesionDataset(root=args.cache, split=args.split, dataset_root=Path(args.root))
    loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False)
    ap_m = BinaryAveragePrecision()
    roc = BinaryAUROC()
    with torch.no_grad():
        for batch in loader:
            dist = batch["bl", "cross", "fu"].edge_attr[:, 3] * 100.0
            scores = -dist
            lab = batch["bl", "cross", "fu"].edge_label.int()
            ap_m.update(scores.cpu(), lab.cpu())
            roc.update(scores.cpu(), lab.cpu())
    cprint(f"distance_baseline split={args.split} AP={float(ap_m.compute()):.4f} AUROC={float(roc.compute()):.4f}", markup=False)
    cprint(f"next: lesionglue_eval --split {args.split} --cache {shlex.quote(args.cache)} --root {shlex.quote(args.root)}", markup=False, soft_wrap=True)


if __name__ == "__main__":
    main()
