"""One-shot scans + clicks → linked instance masks. Argparse + UI only."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from nanounet.common import config_table, console, cprint, nano_banner, quiet_lightning_runtime
from nanounet.config import load_config
from nanounet.data.resampling import set_resample_device
from nanounet.infer.predictor import load_net_from_ckpt, pick_checkpoint
from nanounet.infer.predict_io import patient_ids_from_csv
from nanounet.infer.segtrack import DEFAULT_MODEL, DEFAULT_TRACK, SegTrackCase, pair_folder, resolve_ckpt_path, resolve_out, run_case
from nanounet.plan.plans import Plans


def _require_tracking():
    try:
        from tracking.decode import DECODE_CHOICES
        from tracking.infer import load_matcher
    except ImportError:
        raise SystemExit(
            "tracking is not installed.\n"
            "Expected the lesion-tracking package on PYTHONPATH.\n"
            "Fix: pip install -e /lesion-tracking"
        )
    return DECODE_CHOICES, load_matcher


def _mode(ap: argparse.ArgumentParser, choices: tuple[str, ...]) -> argparse.Namespace:
    ap.add_argument("--bl-dir"), ap.add_argument("--fu-dir")
    ap.add_argument("--bl-img"), ap.add_argument("--bl-clicks")
    ap.add_argument("--fu-img"), ap.add_argument("--fu-clicks")
    ap.add_argument("--meta"), ap.add_argument("--meta-dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("-m", "--model-dir")
    ap.add_argument("--ckpt", default="last.ckpt")
    ap.add_argument("--track-ckpt")
    ap.add_argument("--decode", choices=choices, default="hungarian")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    ap.add_argument("--patients-csv")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-pred", action="store_true")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--inference-mode", choices=("clustered", "centered"), default="clustered")
    ap.add_argument("--disable-tta", dest="tta_flag", action="store_false", default=None)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args()


def _cases(args) -> tuple[list[SegTrackCase], bool]:
    folder = bool(args.bl_dir) or bool(args.fu_dir)
    single = any((args.bl_img, args.bl_clicks, args.fu_img, args.fu_clicks))
    if folder == single:
        raise SystemExit(
            "Need either folder mode (--bl-dir --fu-dir) or single case (--bl-img --bl-clicks --fu-img --fu-clicks).\n"
            "Expected one mode, not both or neither.\n"
            "Fix: see docs/steps/track.md"
        )
    if folder:
        if not (args.bl_dir and args.fu_dir):
            raise SystemExit("--bl-dir requires --fu-dir.\nExpected both folders.\nFix: see docs/steps/track.md")
        if args.meta:
            raise SystemExit("--meta is for single mode.\nExpected --meta-dir in folder mode.\nFix: --meta-dir /path/to/meta")
        cases = pair_folder(Path(args.bl_dir), Path(args.fu_dir))
        if args.patients_csv:
            pids = patient_ids_from_csv(args.patients_csv)
            cases = [c for c in cases if c.stem.split("_", 1)[0] in pids]
            if not cases:
                raise SystemExit(
                    f"no cases match --patients-csv '{args.patients_csv}'.\n"
                    f"Expected CSV column 'patient' matching stem prefixes.\n"
                    f"Fix: --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv  (see docs/steps/track.md)"
                )
        if args.meta_dir:
            md, miss = Path(args.meta_dir), []
            for c in cases:
                p = md / f"{c.stem.split('_', 1)[0]}.csv"
                if not p.is_file():
                    miss.append(str(p))
                else:
                    c.types_csv = p
            if miss:
                raise SystemExit(
                    f"No types CSV at {miss[0]} ({len(miss)} missing).\n"
                    f"Expected {{pid}}.csv under --meta-dir.\n"
                    f"Fix: --meta-dir /nnunet_data/Longitudinal-CT/meta  (see docs/steps/track.md)"
                )
        return cases, False
    if not all((args.bl_img, args.bl_clicks, args.fu_img, args.fu_clicks)):
        raise SystemExit(
            "Single mode needs --bl-img --bl-clicks --fu-img --fu-clicks.\n"
            "Expected four paths.\nFix: see docs/steps/track.md"
        )
    if args.meta_dir or args.patients_csv:
        raise SystemExit("--meta-dir / --patients-csv are folder mode.\nExpected --meta for one case.\nFix: see docs/steps/track.md")
    types = Path(args.meta) if args.meta else None
    if types is not None and not types.is_file():
        raise SystemExit(
            f"No types CSV at {types}.\nExpected lesion_id, lesion_type.\nFix: --meta <pid>.csv or omit it  (see docs/steps/track.md)"
        )
    stem = Path(args.fu_img).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    return [SegTrackCase(stem, Path(args.bl_img), Path(args.bl_clicks), Path(args.fu_img), Path(args.fu_clicks), types)], True


def main() -> None:
    quiet_lightning_runtime()
    choices, load_matcher = _require_tracking()
    args = _mode(argparse.ArgumentParser(), choices)
    cases, single = _cases(args)
    model_dir, msrc = resolve_ckpt_path(args.model_dir, "NANOUNET_SEGTRACK_MODEL", DEFAULT_MODEL)
    track_ckpt, tsrc = resolve_ckpt_path(args.track_ckpt, "NANOUNET_SEGTRACK_TRACK", DEFAULT_TRACK)
    if not (model_dir / "plans.json").is_file():
        raise SystemExit(
            f"No seg model at {model_dir}.\n"
            f"Expected a nanoUNet run dir with plans.json and checkpoints/last.ckpt.\n"
            f"Fix: nanounet_segtrack -m $NANOUNET_RESULTS/nanounet/<run>   or export NANOUNET_SEGTRACK_MODEL=...\n"
            f"(see docs/steps/track.md)"
        )
    if not track_ckpt.is_file():
        raise SystemExit(
            f"No checkpoint at {track_ckpt}.\n"
            f"Expected a Lightning .ckpt from lesion_track_train.\n"
            f"Fix: --track-ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt  (see docs/steps/track.md)"
        )
    d = args.device
    if (d == "cuda" and not torch.cuda.is_available()) or (d == "mps" and not torch.backends.mps.is_available()):
        raise SystemExit(
            f"--device {d} is not available.\nExpected a working {d} device.\nFix: --device cpu  (see docs/steps/track.md)"
        )
    pl = Plans(join(str(model_dir), "plans.json"))
    cm, dj = pl.get_configuration("3d_fullres"), load_json(join(str(model_dir), "dataset.json"))
    cfg = load_config(join(str(model_dir), "nano_config.json"))
    set_resample_device(dev := torch.device(d))
    net, lm = load_net_from_ckpt(pick_checkpoint(str(model_dir), args.ckpt), cm, dj, dev, longi=False, ema=args.ema)
    matcher = load_matcher(track_ckpt, d)
    use_tta = (not cfg.inference.disable_tta_default) if args.tta_flag is None else args.tta_flag
    out_cli = Path(args.out) if args.out else None
    fu_name = Path(args.fu_dir).name if args.fu_dir else None
    parent = resolve_out(cases[0].stem, fu_dir_name=fu_name, out=out_cli, single=single).parent
    nano_banner("nanoUNet  seg × track", "scans + clicks → linked instance masks")
    config_table([
        ("model-dir", model_dir, msrc), ("ckpt", args.ckpt, "cli" if args.ckpt != "last.ckpt" else "default"),
        ("track-ckpt", track_ckpt, tsrc), ("decode", args.decode, "cli" if args.decode != "hungarian" else "default"),
        ("device", d, "cli"), ("n_cases", len(cases), "folder" if not single else "single"),
        ("out", parent, "cli" if args.out else "default"),
    ])
    seg_kw = dict(
        use_tta=use_tta, batch_size=args.batch_size, use_amp=not args.no_amp,
        inference_mode=args.inference_mode, cluster_margin_frac=0.1, border_expand=True,
    )
    n, n_ok, n_empty, n_skip, n_pairs = len(cases), 0, 0, 0, 0
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console(), transient=False,
    ) as prog:
        tid = prog.add_task("seg × track", total=n)
        for i, case in enumerate(cases, 1):
            cdir = resolve_out(case.stem, fu_dir_name=fu_name, out=out_cli, single=single)

            def on_step(s: str, i=i, stem=case.stem) -> None:
                prog.update(tid, description=f"{i}/{n}  {stem}  ·  {s}")

            r = run_case(
                case, cdir, net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dj=dj, dev=dev, matcher=matcher,
                decode=args.decode, overwrite=args.overwrite, keep_pred=args.keep_pred,
                track_ckpt=track_ckpt, thresh=args.thresh, device=d, seg_kw=seg_kw, on_step=on_step,
            )
            n_ok += r["status"] == "ok"
            n_empty += r["status"] == "empty"
            n_skip += r["status"] == "skip"
            n_pairs += r["n_pairs"]
            if r["status"] == "skip":
                cprint(f"[dim]skip {case.stem}[/dim]")
            prog.advance(tid)
    elapsed = time.perf_counter() - t0
    mins, secs = divmod(int(elapsed), 60)
    console().print(Panel(
        f"{n} cases  ·  {n_ok} linked  ·  {n_empty} empty  ·  {n_skip} skip\n"
        f"{n_pairs} pairs  ·  {mins}m {secs:02d}s\n"
        f"wrote  {parent}\n"
        f"next   open fu.nii.gz — same integer = same lesion\n"
        f"       docs/reference/track_ids.md",
        border_style="green",
    ))

if __name__ == "__main__":
    main()
