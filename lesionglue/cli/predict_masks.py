"""CSV-free dense inference from CTs, instance masks, propagated BL centroids."""

import argparse
import csv
from pathlib import Path

import torch

from tracking.data.graph import GraphConfig
from tracking.data.masks import build_mask_graph
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
args = ap.parse_args()

data = build_mask_graph(
    Path(args.bl_img),
    Path(args.bl_mask),
    Path(args.fu_img),
    Path(args.fu_mask),
    Path(args.propagated),
    GraphConfig(k_intra=args.k_intra),
    args.default_lesion_type,
)
mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
mod.eval()
data = data.to(mod.device)

with torch.no_grad():
    outp = mod.matcher(data)
    pair_prob = torch.sigmoid(outp.pair).cpu()
    none_prob = torch.sigmoid(outp.bl_no_match).cpu()
    pair_log = outp.pair.cpu()
    none_log = outp.bl_no_match.cpu()

bl_ids = data["bl"].lesion_id.cpu().tolist()
fu_ids = data["fu"].lesion_id.cpu().tolist()
n_bl, n_fu = len(bl_ids), len(fu_ids)
pmat = pair_prob.reshape(n_bl, n_fu)
lmat = pair_log.reshape(n_bl, n_fu)

with Path(args.out).open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bl_lesion_id", "fu_lesion_id", "pair_prob", "no_match_prob", "decoded"])
    for i, lid in enumerate(bl_ids):
        j = int(torch.argmax(lmat[i]).item())
        is_none = bool(none_log[i] > lmat[i, j])
        w.writerow([int(lid), -1 if is_none else int(fu_ids[j]), float(pmat[i, j]), float(none_prob[i]), int(not is_none)])

if args.pairs_out:
    ei = data["bl", "cross", "fu"].edge_index.cpu()
    with Path(args.pairs_out).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bl_lesion_id", "fu_lesion_id", "prob"])
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            w.writerow([int(bl_ids[i]), int(fu_ids[j]), float(pair_prob[k])])
