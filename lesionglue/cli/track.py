"""Deploy CLI: CT + instance masks + propagated centroids → match CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tracking.common import cprint, config_table, nano_header
from tracking.decode import DECODE_CHOICES, DECODE_HELP, resolve_decode
from tracking.infer import track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl-img", required=True)
    ap.add_argument("--bl-mask", required=True)
    ap.add_argument("--fu-img", required=True)
    ap.add_argument("--fu-mask", required=True)
    ap.add_argument("--propagated", required=True, help="CSV lesion_id,z,y,x (+ optional lesion_type)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", choices=DECODE_CHOICES, default=None, help=DECODE_HELP)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    ap.add_argument("--k-intra", type=int, default=8)
    ap.add_argument("--sinkhorn-iters", type=int, default=20)
    ap.add_argument("--sinkhorn-tau", type=float, default=0.2)
    ap.add_argument("--default-lesion-type", default="unclear")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--pairs-out", default="")
    args = ap.parse_args()

    nano_header("lesion_track")
    decode = resolve_decode(args.decode)
    config_table([
        ("bl-img", args.bl_img, "cli"),
        ("fu-img", args.fu_img, "cli"),
        ("ckpt", args.ckpt, "cli"),
        ("decode", decode, "cli" if args.decode else "prompt"),
        ("thresh", args.thresh, "cli"),
        ("device", args.device, "cli"),
    ])
    r = track(
        Path(args.bl_img), Path(args.bl_mask), Path(args.fu_img), Path(args.fu_mask),
        Path(args.propagated), Path(args.ckpt),
        decode=decode, device=args.device, default_lesion_type=args.default_lesion_type,
        k_intra=args.k_intra, thresh=args.thresh,
        sinkhorn_iters=args.sinkhorn_iters, sinkhorn_tau=args.sinkhorn_tau,
        use_ema=not args.no_ema,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bl_lesion_id", "fu_lesion_id", "pair_prob", "decode"])
        for i, j in r.pairs:
            w.writerow([int(r.bl_ids[i]), int(r.fu_ids[j]), float(r.pair_prob[i, j]), r.decode])
    if args.pairs_out:
        po = Path(args.pairs_out)
        po.parent.mkdir(parents=True, exist_ok=True)
        with po.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bl_lesion_id", "fu_lesion_id", "prob"])
            for i, lid in enumerate(r.bl_ids):
                for j, fid in enumerate(r.fu_ids):
                    w.writerow([int(lid), int(fid), float(r.pair_prob[i, j])])
    cprint(f"n_bl={len(r.bl_ids)} n_fu={len(r.fu_ids)} n_pairs={len(r.pairs)} decode={r.decode}")
    cprint(f"wrote {out}")


if __name__ == "__main__":
    main()
