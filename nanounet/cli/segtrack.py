"""Binary nanoUNet preds + clicks → instance masks → tracking CSV. Paths in; no nested trainer."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from nanounet.common import config_table, cprint, nano_header


def _tracking():
    try:
        from tracking.data.instances import instances_from_nifti
        from tracking.decode import DECODE_CHOICES, DECODE_HELP, resolve_decode
        from tracking.infer import track
    except ImportError:
        raise SystemExit(
            "tracking is not installed.\n"
            "Expected the lesion-tracking package on PYTHONPATH.\n"
            "Fix: pip install -e /lesion-tracking"
        )
    return instances_from_nifti, DECODE_CHOICES, DECODE_HELP, resolve_decode, track


def main() -> None:
    instances_from_nifti, DECODE_CHOICES, DECODE_HELP, resolve_decode, track = _tracking()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl-img", required=True)
    ap.add_argument("--bl-pred", required=True)
    ap.add_argument("--bl-clicks", required=True)
    ap.add_argument("--fu-img", required=True)
    ap.add_argument("--fu-pred", required=True)
    ap.add_argument("--fu-clicks", required=True)
    ap.add_argument("--propagated", required=True)
    ap.add_argument("--track-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", choices=DECODE_CHOICES, default=None, help=DECODE_HELP)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    args = ap.parse_args()

    nano_header("nanounet_segtrack")
    decode = resolve_decode(args.decode)
    config_table([
        ("bl-pred", args.bl_pred, "cli"),
        ("fu-pred", args.fu_pred, "cli"),
        ("track-ckpt", args.track_ckpt, "cli"),
        ("decode", decode, "cli" if args.decode else "prompt"),
        ("out", args.out, "cli"),
    ])
    tmp = Path(tempfile.mkdtemp())
    bl_inst = instances_from_nifti(Path(args.bl_pred), Path(args.bl_clicks), tmp / "bl.nii.gz")
    fu_inst = instances_from_nifti(Path(args.fu_pred), Path(args.fu_clicks), tmp / "fu.nii.gz")
    r = track(
        Path(args.bl_img), bl_inst, Path(args.fu_img), fu_inst,
        Path(args.propagated), Path(args.track_ckpt),
        decode=decode, device=args.device, thresh=args.thresh,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bl_lesion_id", "fu_lesion_id", "pair_prob", "decode"])
        for i, j in r.pairs:
            w.writerow([int(r.bl_ids[i]), int(r.fu_ids[j]), float(r.pair_prob[i, j]), r.decode])
    cprint(f"n_bl={len(r.bl_ids)} n_fu={len(r.fu_ids)} n_pairs={len(r.pairs)} decode={r.decode}")
    cprint(f"wrote {out}")
    cprint("next: inspect the CSV; oracle-mask tracking is lesion_track_eval --split test")


if __name__ == "__main__":
    main()
