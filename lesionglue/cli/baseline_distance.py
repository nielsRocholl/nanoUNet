"""Distance-only baseline on cached graphs: AP/AUROC using score = -dist_mm per cross edge."""

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from lesionglue.common import CACHE_ROOT, DATASET_ROOT
from lesionglue.data.dataset import LesionDataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
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
    print(f"distance_baseline split={args.split} AP={float(ap_m.compute()):.4f} AUROC={float(roc.compute()):.4f}")
