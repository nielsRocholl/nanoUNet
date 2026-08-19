"""Deploy CLI: CT + instance masks + propagated centroids → match CSV.

Single case or Longitudinal-CT folder (--root + --split / --patients-csv).
--propagated: meta CSV, slim CSV, or FU-frame JSON (not inputsTrBL native clicks).
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from tracking.common import DATASET_ROOT, cprint, config_table, nano_header
from tracking.data.instances import instances_from_nifti
from tracking.data.meta import resolve_track_case
from tracking.data.splits import load_holdout, load_tracking_split
from tracking.decode import DECODE_CHOICES, DECODE_HELP, resolve_decode
from tracking.infer import load_matcher, mask_has_lesions, track, write_match_csv

_PROP_HELP = (
    "BL lesion_id → FU-frame centroid: meta CSV (cog_propagated), slim CSV (lesion_id,z,y,x), "
    "or nanoUNet JSON in the FU frame (not inputsTrBL native clicks)"
)


def _mask(path: str, clicks: str) -> Path:
    if not clicks:
        return Path(path)
    tmp = Path(tempfile.mkdtemp()) / "instances.nii.gz"
    return instances_from_nifti(Path(path), Path(clicks), tmp)


def _pids(split: str | None, patients_csv: str) -> list[str]:
    if bool(split) == bool(patients_csv):
        raise SystemExit(
            "Dataset mode needs exactly one of --split or --patients-csv.\n"
            "Expected configs/split.json key or a CSV with a patient column.\n"
            "Fix: --split test   or   --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv"
        )
    if split:
        return list(map(str, load_tracking_split()[split]))
    return load_holdout(patients_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl-img", default="")
    ap.add_argument("--bl-mask", default="")
    ap.add_argument("--fu-img", default="")
    ap.add_argument("--fu-mask", default="")
    ap.add_argument("--propagated", default="", help=_PROP_HELP)
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
    ap.add_argument("--bl-clicks", default="", help="instance JSON; treat --bl-mask as binary FG")
    ap.add_argument("--fu-clicks", default="", help="instance JSON; treat --fu-mask as binary FG")
    ap.add_argument("--root", default="", help=f"Longitudinal-CT root (default layout under {DATASET_ROOT})")
    ap.add_argument("--split", choices=("train", "val", "test"), default=None)
    ap.add_argument("--patients-csv", default="")
    ap.add_argument("--bl-mask-dir", default="")
    ap.add_argument("--fu-mask-dir", default="")
    ap.add_argument("--prop-dir", default="")
    args = ap.parse_args()

    nano_header("lesion_track")
    decode = resolve_decode(args.decode)
    root = args.root.strip()
    single = all(getattr(args, k).strip() for k in ("bl_img", "bl_mask", "fu_img", "fu_mask", "propagated"))
    if bool(root) == single:
        raise SystemExit(
            "Need either a single case (--bl-img --bl-mask --fu-img --fu-mask --propagated) or a dataset (--root).\n"
            "Expected one mode, not both or neither.\n"
            "Fix: lesion_track --root /nnunet_data/Longitudinal-CT --split test --ckpt ... --decode dense --out /tmp/track_test"
        )
    if not root and (args.split is not None or args.patients_csv.strip()):
        raise SystemExit(
            "--split / --patients-csv require --root.\n"
            "Expected Longitudinal-CT dataset mode.\n"
            "Fix: add --root /nnunet_data/Longitudinal-CT"
        )

    kw = dict(
        decode=decode, device=args.device, default_lesion_type=args.default_lesion_type,
        k_intra=args.k_intra, thresh=args.thresh, sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_tau=args.sinkhorn_tau, use_ema=not args.no_ema,
    )
    matcher = load_matcher(Path(args.ckpt), args.device)
    config_table([
        ("ckpt", args.ckpt, "cli"),
        ("decode", decode, "cli" if args.decode else "prompt"),
        ("mode", "dataset" if root else "single", "cli"),
        ("out", args.out, "cli"),
    ])
    if not root:
        r = track(
            Path(args.bl_img), _mask(args.bl_mask, args.bl_clicks),
            Path(args.fu_img), _mask(args.fu_mask, args.fu_clicks),
            Path(args.propagated), Path(args.ckpt), matcher=matcher, **kw,
        )
        out = Path(args.out)
        write_match_csv(out, r)
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
        return

    pids = _pids(args.split, args.patients_csv.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_pairs = 0
    cols = (SpinnerColumn(style="cyan"), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[dim]{task.completed}/{task.total}[/dim]"))
    with Progress(*cols) as prog:
        task = prog.add_task("track", total=len(pids))
        for pid in pids:
            case = resolve_track_case(
                Path(root), pid,
                prop_dir=Path(args.prop_dir) if args.prop_dir else None,
                bl_mask_dir=Path(args.bl_mask_dir) if args.bl_mask_dir else None,
                fu_mask_dir=Path(args.fu_mask_dir) if args.fu_mask_dir else None,
            )
            if case is None or not mask_has_lesions(case.bl_mask) or not mask_has_lesions(case.fu_mask):
                n_skip += 1
                cprint(f"skip {pid}")
                prog.advance(task)
                continue
            r = track(
                case.bl_img, case.bl_mask, case.fu_img, case.fu_mask, case.propagated,
                Path(args.ckpt), matcher=matcher, **kw,
            )
            write_match_csv(out_dir / f"{pid}.csv", r)
            n_ok += 1
            n_pairs += len(r.pairs)
            prog.advance(task)
    t = Table(title="lesion_track", box=None, padding=(0, 2))
    t.add_column("split", style="cyan")
    t.add_column("n", justify="right")
    t.add_row("ok", str(n_ok))
    t.add_row("skip", str(n_skip))
    t.add_row("pairs", str(n_pairs))
    cprint(t)
    cprint(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
