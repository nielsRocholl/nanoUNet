"""One-shot scans + clicks → linked instance masks. Argparse + UI only."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from nanounet.common import config_table, console, cprint, nano_banner, quiet_lightning_runtime
from nanounet.config import load_config
from nanounet.data.resampling import set_resample_device
from nanounet.cli.segtrack_cases import collect_cases
from nanounet.infer.predictor import load_net_from_ckpt, pick_checkpoint
from nanounet.infer.segtrack import DEFAULT_MODEL, load_case_io, run_case
from nanounet.infer.segtrack_case import resolve_ckpt_path, resolve_out
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
    ap.add_argument("--bl-mask"), ap.add_argument("--bl-mask-dir")
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
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--inference-mode", choices=("clustered", "centered"), default="clustered")
    ap.add_argument("--disable-tta", dest="tta_flag", action="store_false", default=None)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args()


def main() -> None:
    quiet_lightning_runtime()
    choices, load_matcher = _require_tracking()
    from tracking.common import DEPLOYED_CKPT, DEPLOYED_DUST_TAU
    from tracking.infer import graph_cfg_from_ckpt

    args = _mode(argparse.ArgumentParser(), choices)
    cases, single, skipped, (meta_dir, meta_src) = collect_cases(args)
    model_dir, msrc = resolve_ckpt_path(args.model_dir, "NANOUNET_SEGTRACK_MODEL", DEFAULT_MODEL)
    track_ckpt, tsrc = resolve_ckpt_path(args.track_ckpt, "NANOUNET_SEGTRACK_TRACK", DEPLOYED_CKPT)
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
            f"Fix: --track-ckpt {DEPLOYED_CKPT}  (see docs/steps/track.md)"
        )
    d = args.device
    if (d == "cuda" and not torch.cuda.is_available()) or (d == "mps" and not torch.backends.mps.is_available()):
        raise SystemExit(
            f"--device {d} is not available.\nExpected a working {d} device.\nFix: --device cpu  (see docs/steps/track.md)"
        )
    matcher = load_matcher(track_ckpt, d)
    gcfg = graph_cfg_from_ckpt(matcher, int(getattr(matcher.hparams, "k_intra", 8)))
    out_cli = Path(args.out) if args.out else None
    fu_name = Path(args.fu_dir).name if args.fu_dir else None
    parent = resolve_out(cases[0].stem if cases else "_", fu_dir_name=fu_name, out=out_cli, single=single).parent
    nano_banner("nanoUNet  seg × track", "scans + clicks → linked instance masks")
    rows = [
        ("model-dir", model_dir, msrc), ("ckpt", args.ckpt, "cli" if args.ckpt != "last.ckpt" else "default"),
        ("track-ckpt", track_ckpt, tsrc),
        ("decode", args.decode, "cli" if args.decode != "hungarian" else "default"),
        ("sinkhorn-tau", DEPLOYED_DUST_TAU, "default"),
        ("track-ema", "on", "default"),
        ("seg-ema", "on" if args.ema else "off", "default" if args.ema else "cli"),
        ("intra", gcfg.intra, "ckpt"),
        ("drop_dp", gcfg.drop_dp, "ckpt"),
        ("device", d, "cli"), ("n_cases", len(cases), "folder" if not single else "single"),
        ("out", parent, "cli" if args.out else "default"),
    ]
    if args.bl_mask:
        rows.append(("bl-mask", args.bl_mask, "cli"))
    elif args.bl_mask_dir:
        rows.append(("bl-mask-dir", args.bl_mask_dir, "cli"))
    rows.append(("meta-dir", meta_dir if meta_dir is not None else "none", meta_src))
    config_table(rows)
    for stem, why in skipped:
        cprint(f"[dim]skip {stem}  ({why})[/dim]")
    n_all, n_ok, n_empty, n_skip, n_pairs = len(cases) + len(skipped), 0, 0, len(skipped), 0
    if not cases:
        console().print(Panel(
            f"{n_all} cases  ·  0 linked  ·  0 empty  ·  {n_skip} skip\n"
            f"0 pairs  ·  0m 00s\n"
            f"wrote  {parent}\n"
            f"next   docs/steps/track.md",
            border_style="green",
        ))
        return
    pl = Plans(join(str(model_dir), "plans.json"))
    cm, dj = pl.get_configuration("3d_fullres"), load_json(join(str(model_dir), "dataset.json"))
    cfg = load_config(join(str(model_dir), "nano_config.json"))
    set_resample_device(dev := torch.device(d))
    net, lm = load_net_from_ckpt(pick_checkpoint(str(model_dir), args.ckpt), cm, dj, dev, longi=False, ema=args.ema)
    use_tta = (not cfg.inference.disable_tta_default) if args.tta_flag is None else args.tta_flag
    seg_kw = dict(
        use_tta=use_tta, batch_size=args.batch_size, use_amp=not args.no_amp,
        inference_mode=args.inference_mode, cluster_margin_frac=0.1, border_expand=True,
    )
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console(), transient=False,
    ) as prog:
        tid = prog.add_task("seg × track", total=len(cases))
        with ThreadPoolExecutor(max_workers=1) as io_pool:
            next_fut = io_pool.submit(load_case_io, cases[0])
            for i, case in enumerate(cases, 1):
                preloaded = next_fut.result()
                next_fut = io_pool.submit(load_case_io, cases[i]) if i < len(cases) else None
                cdir = resolve_out(case.stem, fu_dir_name=fu_name, out=out_cli, single=single)

                def on_step(s: str, i=i, stem=case.stem) -> None:
                    prog.update(tid, description=f"{i}/{len(cases)}  {stem}  ·  {s}")

                try:
                    r = run_case(
                        case, cdir, net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dj=dj, dev=dev, matcher=matcher,
                        decode=args.decode, overwrite=args.overwrite, keep_pred=args.keep_pred,
                        track_ckpt=track_ckpt, thresh=args.thresh, device=d, seg_kw=seg_kw, on_step=on_step,
                        preloaded=preloaded,
                    )
                except SystemExit as e:
                    if single:
                        raise
                    cprint(f"[dim]skip {case.stem}  ({str(e).splitlines()[0]})[/dim]")
                    n_skip += 1
                    prog.advance(tid)
                    continue
                n_ok += r["status"] == "ok"
                n_empty += r["status"] == "empty"
                n_skip += r["status"] == "skip"
                n_pairs += r["n_pairs"]
                if r["status"] == "skip":
                    why = r.get("why")
                    cprint(f"[dim]skip {case.stem}  ({why})[/dim]" if why else f"[dim]skip {case.stem}[/dim]")
                else:
                    cprint(f"[dim]{case.stem}  {r['sec']:.0f}s  seg={r.get('t_seg', 0):.0f}s track={r.get('t_track', 0):.0f}s[/dim]")
                prog.advance(tid)
    elapsed = time.perf_counter() - t0
    mins, secs = divmod(int(elapsed), 60)
    console().print(Panel(
        f"{n_all} cases  ·  {n_ok} linked  ·  {n_empty} empty  ·  {n_skip} skip\n"
        f"{n_pairs} pairs  ·  {mins}m {secs:02d}s\n"
        f"wrote  {parent}\n"
        f"next   open fu.mha — same integer = same lesion\n"
        f"       docs/reference/track_ids.md",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
