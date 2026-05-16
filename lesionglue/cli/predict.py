"""Batch inference: cross-edge probs per patient CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from tracking.common import CACHE_ROOT, DATASET_ROOT
from tracking.data.dataset import LesionDataset
from tracking.matcher import decode_hungarian
from tracking.train.module import MatcherModule

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", default="preds")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--dump-all", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    mod.eval()
    dev = mod.device
    ds = LesionDataset(root=args.cache, split=args.split, dataset_root=Path(args.root))
    loader = PyGDataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            outp = mod.matcher(batch)
            prob = torch.sigmoid(outp.pair).cpu().numpy()
            data = batch.to_data_list()[0]
            pid = data.pid
            ei = data["bl", "cross", "fu"].edge_index.cpu().numpy()
            bl_ids = data["bl"].lesion_id.cpu().numpy()
            fu_ids = data["fu"].lesion_id.cpu().numpy()
            n_bl = data["bl"].num_nodes
            n_fu = data["fu"].num_nodes
            pm = np.zeros((n_bl, n_fu), dtype=np.float64)
            for k in range(ei.shape[1]):
                pm[ei[0, k], ei[1, k]] = prob[k]
            hung = decode_hungarian(pm, args.thresh) if args.strict else None
            rows = []
            for k in range(ei.shape[1]):
                bi, fj = ei[:, k]
                p = float(prob[k])
                if not args.dump_all and p < args.thresh:
                    continue
                if args.strict and hung is not None:
                    dec = int(hung[bi] == fj)
                else:
                    dec = int(p >= args.thresh)
                rows.append((int(bl_ids[bi]), int(fu_ids[fj]), p, dec))
            path = out / f"{pid}.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bl_lesion_id", "fu_lesion_id", "prob", "decoded"])
                w.writerows(rows)
