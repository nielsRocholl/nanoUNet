"""Cached-graph benchmark inference. Deployment: lesionglue_track (lesionglue/cli/track.py)."""

import argparse
import csv
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from lesionglue.common import CACHE_ROOT, DATASET_ROOT, DEPLOYED_CKPT, DEPLOYED_DUST_TAU, require_ckpt
from lesionglue.data.cache.dataset import LesionDataset
from lesionglue.model.decode import decode_sinkhorn_hungarian
from lesionglue.infer import graph_cfg_from_ckpt
from lesionglue.train.module import MatcherModule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEPLOYED_CKPT), help="matcher Lightning checkpoint (default: DEPLOYED_CKPT in lesionglue/common.py)")
    ap.add_argument("--cache", default=str(CACHE_ROOT), help="root of the cached lesion graphs")
    ap.add_argument("--root", default=str(DATASET_ROOT), help="dataset root directory passed to LesionDataset")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="cached split to predict")
    ap.add_argument("--out", default="preds", help="output directory, one <patient>.csv per patient")
    ap.add_argument("--thresh", type=float, default=0.5, help="edge probability cutoff in [0, 1]: lower edges are not written (unless --dump-all); decoded = prob >= thresh (unless --strict)")
    ap.add_argument("--sinkhorn-iters", type=int, default=20, help="Sinkhorn normalisation iterations for --strict decoding")
    ap.add_argument("--sinkhorn-tau", type=float, default=DEPLOYED_DUST_TAU, help="min row-normalised transport mass to accept a match in --strict decoding (default: deployed dust tau)")
    ap.add_argument("--no-ema", action="store_true", help="use training weights instead of EMA shadow")
    ap.add_argument("--dump-all", action="store_true", help="write every candidate edge, ignoring --thresh")
    ap.add_argument("--strict", action="store_true", help="set decoded from a 1-to-1 Sinkhorn + Hungarian assignment instead of prob >= --thresh")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mod = MatcherModule.load_from_checkpoint(str(require_ckpt(args.ckpt)), map_location="cpu")
    mod.eval()
    dev = mod.device
    gcfg = graph_cfg_from_ckpt(mod, int(getattr(mod.hparams, "k_intra", 8)))
    ds = LesionDataset(root=args.cache, split=args.split, dataset_root=Path(args.root), cfg=gcfg)
    loader = PyGDataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            outp = mod.predict_batch(batch, use_ema=not args.no_ema)
            prob = torch.sigmoid(outp.pair).cpu().numpy()
            data = batch.to_data_list()[0]
            pid = data.pid
            ei = data["bl", "cross", "fu"].edge_index.cpu().numpy()
            bl_ids = data["bl"].lesion_id.cpu().numpy()
            fu_ids = data["fu"].lesion_id.cpu().numpy()
            n_bl = data["bl"].num_nodes
            n_fu = data["fu"].num_nodes
            dec_arr = None
            if args.strict:
                dec_arr = decode_sinkhorn_hungarian(
                    outp.pair.cpu(), outp.dust_bl.cpu(), outp.dust_fu.cpu(),
                    n_bl, n_fu, iters=args.sinkhorn_iters, tau=args.sinkhorn_tau,
                )
            rows = []
            for k in range(ei.shape[1]):
                bi, fj = ei[:, k]
                p = float(prob[k])
                if not args.dump_all and p < args.thresh:
                    continue
                if dec_arr is not None:
                    di = int(dec_arr[int(bi)])
                    edge_dec = int(di == int(fj) and di >= 0)
                else:
                    edge_dec = int(p >= args.thresh)
                rows.append((int(bl_ids[bi]), int(fu_ids[fj]), p, edge_dec))
            path = out / f"{pid}.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bl_lesion_id", "fu_lesion_id", "prob", "decoded"])
                w.writerows(rows)


if __name__ == "__main__":
    main()
