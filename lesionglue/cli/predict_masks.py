"""CSV-free dense inference from CTs, instance masks, propagated BL centroids."""

import argparse
import csv
from pathlib import Path

import torch
from torch_geometric.data import Batch

from tracking.data.features import add_feat_args, feat_from_args
from tracking.data.graph import GraphConfig
from tracking.data.masks import build_mask_graph
from tracking.decode import decode_sinkhorn_hungarian
from tracking.train.module import MatcherModule

ap = argparse.ArgumentParser()
ap.add_argument("--bl-img", required=True)
ap.add_argument("--bl-mask", required=True)
ap.add_argument("--fu-img", required=True)
ap.add_argument("--fu-mask", required=True)
ap.add_argument("--propagated", required=True, help="CSV with lesion_id,z,y,x and optional lesion_type")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pairs-out", default="")
ap.add_argument("--default-lesion-type", default=None)
ap.add_argument("--k-intra", type=int, default=8)
ap.add_argument("--sinkhorn-iters", type=int, default=20)
ap.add_argument("--sinkhorn-tau", type=float, default=0.2)
ap.add_argument("--tta-n", type=int, default=5, help="TTA jitter passes; 0 disables")
ap.add_argument("--no-ema", action="store_true", help="use training weights instead of EMA shadow")
add_feat_args(ap)
args = ap.parse_args()

feat = feat_from_args(args)
mae = None
if feat.mode == "mae":
    from tracking.data.mae import MaeExtractor

    mae = MaeExtractor(feat)
data = build_mask_graph(
    Path(args.bl_img),
    Path(args.bl_mask),
    Path(args.fu_img),
    Path(args.fu_mask),
    Path(args.propagated),
    GraphConfig(k_intra=args.k_intra, feat=feat),
    args.default_lesion_type,
    mae=mae,
)
mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
mod.eval()
data = data.to(mod.device)
bat = Batch.from_data_list([data]).to(mod.device)

with torch.no_grad():
    outp = mod.predict_batch(bat, tta_n=args.tta_n, use_ema=not args.no_ema)
    pair_prob = torch.sigmoid(outp.pair).cpu()
    pr = outp.pair.cpu()
    db = outp.dust_bl.cpu()
    df = outp.dust_fu.cpu()

bl_ids = data["bl"].lesion_id.cpu().tolist()
fu_ids = data["fu"].lesion_id.cpu().tolist()
n_bl, n_fu = len(bl_ids), len(fu_ids)
pmat = pair_prob.reshape(n_bl, n_fu)
dec = decode_sinkhorn_hungarian(pr, db, df, n_bl, n_fu, iters=args.sinkhorn_iters, tau=args.sinkhorn_tau)

with Path(args.out).open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bl_lesion_id", "fu_lesion_id", "pair_prob", "decoded"])
    for i, lid in enumerate(bl_ids):
        j = int(dec[i])
        if j < 0:
            w.writerow([int(lid), -1, 0.0, 0])
        else:
            w.writerow([int(lid), int(fu_ids[j]), float(pmat[i, j]), 1])

if args.pairs_out:
    ei = data["bl", "cross", "fu"].edge_index.cpu()
    with Path(args.pairs_out).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bl_lesion_id", "fu_lesion_id", "prob"])
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            w.writerow([int(bl_ids[i]), int(fu_ids[j]), float(pair_prob[k])])
