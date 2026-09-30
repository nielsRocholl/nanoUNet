"""Deploy CLI: CT + instance masks → match CSV. drop_dp ckpts omit --propagated.

Single case or Longitudinal-CT folder (--root + --split / --patients-csv).
Defaults: v7_complete last.ckpt, EMA, hungarian, dust_tau=0.125.
--propagated: meta CSV, slim CSV, or FU-frame JSON (not inputsTrBL native clicks).
Required unless the checkpoint was trained with drop_dp.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import tempfile
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from core.ui import console
from lesionglue.common import DATASET_ROOT, DEPLOYED_CKPT, DEPLOYED_DUST_TAU, cprint, config_table, nano_header, require_ckpt
from lesionglue.data.instances.build import instances_from_nifti
from lesionglue.data.source.meta import resolve_track_case
from lesionglue.data.source.splits import load_holdout, load_tracking_split
from lesionglue.model.decode import DECODE_CHOICES, DECODE_HELP
from lesionglue.infer import graph_cfg_from_ckpt, load_matcher, mask_has_lesions, track, write_match_csv

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
            "Expected lesionglue/configs/split.json key or a CSV with a patient column.\n"
            "Fix: --split test   or   --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv"
        )
    if split:
        return list(map(str, load_tracking_split()[split]))
    return load_holdout(patients_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl-img", default="", help="single case: baseline CT NIfTI")
    ap.add_argument("--bl-mask", default="", help="single case: baseline lesion instance mask NIfTI (binary FG with --bl-clicks)")
    ap.add_argument("--fu-img", default="", help="single case: follow-up CT NIfTI")
    ap.add_argument("--fu-mask", default="", help="single case: follow-up lesion instance mask NIfTI (binary FG with --fu-clicks)")
    ap.add_argument(
        "--propagated", default="",
        help=_PROP_HELP + ". Required unless the checkpoint was trained with drop_dp.",
    )
    ap.add_argument("--ckpt", default=str(DEPLOYED_CKPT), help="matcher Lightning checkpoint; default: DEPLOYED_CKPT in lesionglue/common.py")
    ap.add_argument("--out", required=True, help="single case: match CSV to write; dataset mode: dir for one <pid>.csv per patient")
    ap.add_argument("--decode", choices=DECODE_CHOICES, default="hungarian", help=DECODE_HELP)
    ap.add_argument("--thresh", type=float, default=0.5, help="pair probability cutoff (0-1) for --decode dense; unused by sinkhorn and hungarian")
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda", help="device the matcher runs on")
    ap.add_argument("--k-intra", type=int, default=8, help="kNN neighbours per lesion in the intra-scan graph; must equal the checkpoint value")
    ap.add_argument("--sinkhorn-iters", type=int, default=20, help="Sinkhorn normalisation iterations for --decode sinkhorn and hungarian")
    ap.add_argument("--sinkhorn-tau", type=float, default=DEPLOYED_DUST_TAU, help="min row-normalised Sinkhorn mass to keep a pair (sinkhorn, hungarian); default: DEPLOYED_DUST_TAU in lesionglue/common.py")
    ap.add_argument("--default-lesion-type", default="unclear", help="lesion type given to lesions absent from --types-csv (single case); 'unclear' = no real type")
    ap.add_argument("--types-csv", default="", help="lesion_id,lesion_type CSV; required for type_mask unless --default-lesion-type is not unclear")
    ap.add_argument("--no-ema", action="store_true", help="run the raw weights instead of the EMA weights")
    ap.add_argument("--pairs-out", default="", help="single case: also write every BL x FU pair probability to this CSV; empty = skip")
    ap.add_argument("--bl-clicks", default="", help="instance JSON; treat --bl-mask as binary FG")
    ap.add_argument("--fu-clicks", default="", help="instance JSON; treat --fu-mask as binary FG")
    ap.add_argument("--root", default="", help=f"Longitudinal-CT root (default layout under {DATASET_ROOT})")
    ap.add_argument("--split", choices=("train", "val", "test"), default=None, help="dataset mode: patients from this lesionglue/configs/split.json split (or use --patients-csv)")
    ap.add_argument("--patients-csv", default="", help="dataset mode: CSV with a patient column listing the patients to track (or use --split)")
    ap.add_argument("--bl-mask-dir", default="", help="dataset mode: dir of <pid>_<idx>.nii.gz baseline masks; empty = root/targetsTrBL")
    ap.add_argument("--fu-mask-dir", default="", help="dataset mode: dir of <pid>_<idx>.nii.gz follow-up masks; empty = root/targetsTrFU")
    ap.add_argument("--prop-dir", default="", help="dataset mode: dir of per-patient propagated coords (<pid>.csv or <pid>_<idx>.json); empty = root/meta")
    args = ap.parse_args()

    nano_header("lesionglue_track")
    ckpt = require_ckpt(args.ckpt)
    decode = args.decode
    matcher = load_matcher(ckpt, args.device)
    gcfg = graph_cfg_from_ckpt(matcher, args.k_intra)
    root = args.root.strip()
    need = ("bl_img", "bl_mask", "fu_img", "fu_mask")
    if not gcfg.drop_dp:
        need = (*need, "propagated")
    single = all(getattr(args, k).strip() for k in need)
    if bool(root) == single:
        geo = "--bl-img --bl-mask --fu-img --fu-mask" + ("" if gcfg.drop_dp else " --propagated")
        raise SystemExit(
            f"Need either a single case ({geo}) or a dataset (--root).\n"
            "Expected one mode, not both or neither.\n"
            "Fix: lesionglue_track --root /nnunet_data/Longitudinal-CT --split test --out /tmp/track_test"
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
    config_table([
        ("ckpt", str(ckpt), "default" if ckpt == DEPLOYED_CKPT else "cli"),
        ("decode", decode, "default" if decode == "hungarian" else "cli"),
        ("sinkhorn-tau", args.sinkhorn_tau, "default" if args.sinkhorn_tau == DEPLOYED_DUST_TAU else "cli"),
        ("ema", "off" if args.no_ema else "on", "cli" if args.no_ema else "default"),
        ("drop_dp", str(gcfg.drop_dp), "ckpt"),
        ("intra", gcfg.intra, "ckpt"),
        ("type_mask", str(gcfg.type_mask), "ckpt"),
        ("mode", "dataset" if root else "single", "cli"),
        ("out", args.out, "cli"),
    ])
    if not root:
        r = track(
            Path(args.bl_img), _mask(args.bl_mask, args.bl_clicks),
            Path(args.fu_img), _mask(args.fu_mask, args.fu_clicks),
            None if gcfg.drop_dp else Path(args.propagated), ckpt, matcher=matcher,
            types_csv=Path(args.types_csv) if args.types_csv.strip() else None, **kw,
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
        cprint(f"next: head -n 20 {shlex.quote(str(out))}", markup=False, soft_wrap=True)
        return

    pids = _pids(args.split, args.patients_csv.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_pairs = 0
    cols = (SpinnerColumn(style="cyan"), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[dim]{task.completed}/{task.total}[/dim]"))
    with Progress(*cols, console=console()) as prog:
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
                case.bl_img, case.bl_mask, case.fu_img, case.fu_mask,
                None if gcfg.drop_dp else case.propagated,
                ckpt, matcher=matcher,
                types_csv=case.propagated if gcfg.type_mask else None, **kw,
            )
            write_match_csv(out_dir / f"{pid}.csv", r)
            n_ok += 1
            n_pairs += len(r.pairs)
            prog.advance(task)
    t = Table(title="lesionglue_track", box=None, padding=(0, 2))
    t.add_column("split", style="cyan")
    t.add_column("n", justify="right")
    t.add_row("ok", str(n_ok))
    t.add_row("skip", str(n_skip))
    t.add_row("pairs", str(n_pairs))
    cprint(t)
    cprint(f"wrote {out_dir}")
    cprint(f"next: ls {shlex.quote(str(out_dir))}", markup=False, soft_wrap=True)


if __name__ == "__main__":
    main()
