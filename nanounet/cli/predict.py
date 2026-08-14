"""Dataset / single-case prompt-driven inference: CPU prefetch, depth-1 export overlap."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p

from nanounet.common import config_table, cprint, nano_header
from nanounet.config import load_config
from nanounet.infer.predict_case import MAX_BORDER_EXTRA, predict_case_logits
from nanounet.infer.tta import cat_status
from nanounet.infer.export import export_prediction_from_logits
from nanounet.infer.predict_io import baseline_resolver, check_baseline_files, patient_ids_from_csv, preprocess_case
from nanounet.infer.predictor import load_net_from_ckpt, pick_checkpoint
from nanounet.model.dwb import LongiResEncUNet
from nanounet.plan.labels import labels_from_dataset_json
from nanounet.plan.plans import Plans
from nanounet.score import check_gt_dir, report, score_case, write


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="folder or single .nii.gz")
    ap.add_argument("-o", "--output", required=True, help="output folder or single .nii.gz")
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--points", default=None, help="points JSON (single mode)")
    ap.add_argument("--baseline-image", default=None, help="sibling BL .nii.gz for two-stream longi inference")
    ap.add_argument("--baseline-points", default=None, help="BL click JSON (single mode); native voxel x,y,z")
    ap.add_argument("--baseline-dir", default=None, help="dataset mode: per-case BL <cid>.nii.gz + <cid>.json")
    ap.add_argument("--longi", action="store_true", help="force two-stream net build (else auto-detect from ckpt)")
    ap.add_argument("--no-prompt-encode", action="store_true")
    ap.add_argument("--no-border-expand", dest="border_expand", action="store_false")
    ap.set_defaults(border_expand=True)
    ap.add_argument("--max-border-extra", type=int, default=MAX_BORDER_EXTRA)
    tta_g = ap.add_mutually_exclusive_group()
    tta_g.add_argument("--disable-tta", dest="tta_flag", action="store_false", default=None)
    tta_g.add_argument("--tta", dest="tta_flag", action="store_true", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cluster-margin-frac", type=float, default=0.1)
    ap.add_argument("--inference-mode", choices=("clustered", "centered"), default="clustered")
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--patients-csv", default=None, help="CSV with patient column; keep cases whose id prefix matches")
    ap.add_argument("--gt-dir", default=None, help="instance-labeled native GT folder (same stems as -i)")
    ap.add_argument("--metrics-out", default=None, help="write {stem}.json and {stem}.csv; requires --gt-dir")
    args = ap.parse_args()
    if args.metrics_out and not args.gt_dir:
        raise SystemExit(
            "--metrics-out was set without --gt-dir.\n"
            "Scoring needs instance-labeled native GT with the same stems as -i.\n"
            "Fix: nanounet_predict ... --gt-dir <targetsTrFU> --metrics-out <stem>   (see docs/steps/predict.md)"
        )

    nano_header("nanoUNet predict", color="blue")
    md = args.model_dir
    pl = Plans(join(md, "plans.json"))
    cm = pl.get_configuration("3d_fullres")
    dj = load_json(join(md, "dataset.json"))
    cfg = load_config(join(md, "nano_config.json"))
    labels_from_dataset_json(dj)
    if bool(args.baseline_points) != bool(args.baseline_image):
        raise SystemExit("--baseline-points requires --baseline-image")

    d = args.device
    if d == "cuda" and not torch.cuda.is_available():
        d = "cpu"
    if d == "mps" and not torch.backends.mps.is_available():
        d = "cpu"
    dev = torch.device(d)
    net, lm = load_net_from_ckpt(pick_checkpoint(md, args.ckpt), cm, dj, dev, longi=args.longi)
    use_tta = (not cfg.inference.disable_tta_default) if args.tta_flag is None else args.tta_flag
    end = dj["file_ending"]
    single_mode = not os.path.isdir(args.input)
    if not single_mode:
        case_files = sorted(f for f in os.listdir(args.input) if f.endswith(end))
        cases = [(f[:-len(end)], join(args.input, f), join(args.input, f[:-len(end)] + ".json"), None) for f in case_files]
        if args.patients_csv:
            pids = patient_ids_from_csv(args.patients_csv)
            cases = [(cid, scan, jp, ot) for cid, scan, jp, ot in cases if cid.split("_", 1)[0] in pids]
            if not cases:
                raise SystemExit(f"no cases match --patients-csv {args.patients_csv}")
        missing = [cid for cid, _, jp, _ in cases if not os.path.isfile(jp)]
        if missing:
            raise FileNotFoundError(f"missing points JSON for: {', '.join(missing)}")
        out_dir = args.output
        maybe_mkdir_p(out_dir)
    else:
        if not args.points:
            raise SystemExit("single mode requires --points")
        scan = args.input
        case_id = os.path.basename(scan)
        if case_id.endswith(end):
            case_id = case_id[: -len(end)]
        out_trunc = args.output[: -len(end)] if args.output.endswith(end) else args.output
        out_dir = os.path.dirname(out_trunc) or "."
        maybe_mkdir_p(out_dir)
        cases = [(case_id, scan, args.points, out_trunc)]

    is_longi = isinstance(net, LongiResEncUNet)
    if single_mode and args.baseline_dir:
        raise SystemExit("--baseline-dir is for dataset mode; single mode uses --baseline-image/--baseline-points")
    if not single_mode and (args.baseline_image or args.baseline_points):
        raise SystemExit("dataset mode uses --baseline-dir (per-case BL); not --baseline-image/--baseline-points")
    resolve_bl, bl_present = baseline_resolver(args.baseline_image, args.baseline_points, args.baseline_dir, end)
    if bl_present and not is_longi:
        raise SystemExit("baseline given but checkpoint is not longi (no dwb.* keys). Drop --baseline-* or pass a longi ckpt.")
    if is_longi and not bl_present:
        cprint("[yellow]longi checkpoint without a baseline: running null-baseline (single-timepoint identity)[/yellow]")
    if args.baseline_dir: check_baseline_files(cases, resolve_bl, args.baseline_dir, end)
    if args.gt_dir: check_gt_dir(args.gt_dir, cases, end)
    config_table(
        [("model_dir", args.model_dir, "cli"), ("ckpt", args.ckpt or "auto", "cli/default"),
         ("device", args.device, "cli/default"), ("inference_mode", args.inference_mode, "cli/default"),
         ("border_expand", args.border_expand, "cli/default"), ("batch_size", args.batch_size, "cli/default"),
         ("tta", "auto" if args.tta_flag is None else args.tta_flag, "cli/config"),
         ("longi", "on" if bl_present else ("null-baseline" if is_longi else "off"), "cli/ckpt"),
         ("gt_dir", args.gt_dir or "off", "cli" if args.gt_dir else "default"),
         ("metrics_out", args.metrics_out or "off", "cli" if args.metrics_out else "default")],
        title="nanoUNet predict",
    )
    n = len(cases)
    scored = [(c, (o if o is not None else join(out_dir, c)) + end, j) for c, _, j, o in cases]
    ex, prev, logged = ThreadPoolExecutor(max_workers=1), None, False

    def push(fn, *a):
        nonlocal prev
        if prev is not None:
            prev.result()
        prev = ex.submit(fn, *a)

    def gpu(case_id, idx, out_trunc, pack, bl_case):
        nonlocal logged
        t0 = time.perf_counter()
        pad_cpu, slicer_revert, props, points_xyz, bl_points = pack
        pad = pad_cpu.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else pad_cpu.to(dev)
        logits, tiles = predict_case_logits(
            net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dev=dev,
            pad=pad, slicer_revert=slicer_revert, props=props, points_xyz=points_xyz,
            encode_prompt=not args.no_prompt_encode, use_tta=use_tta,
            border_expand=args.border_expand, max_border_expand_extra=args.max_border_extra,
            batch_size=args.batch_size, use_amp=not args.no_amp,
            cluster_margin_frac=args.cluster_margin_frac, mode=args.inference_mode,
            is_longi=is_longi, bl_present=bl_case, bl_points_xyz=bl_points,
        )
        if not logged:
            logged, s = True, cat_status()
            if s:
                cprint(f"[dim]{s}[/dim]")
        push(export_prediction_from_logits, logits, props, cm, pl, dj, out_trunc, tiles)
        cprint(f"[bold green][{idx}/{n}] {case_id} ({time.perf_counter() - t0:.1f}s)[/bold green]")

    if n == 1 or args.num_workers <= 0:
        for i, (cid, scan, jp, ot) in enumerate(cases, 1):
            out = ot if ot is not None else join(out_dir, cid)
            if not args.overwrite and os.path.isfile(out + end):
                cprint(f"[dim][{i}/{n}] skip {cid} (exists)[/dim]")
                continue
            bs, bj = resolve_bl(cid)
            gpu(cid, i, out, preprocess_case(scan, jp, pl, cm, dj, bs, bj), bs is not None)
    else:
        pool = ThreadPoolExecutor(max_workers=args.num_workers)
        inflight: deque = deque()
        for i, (cid, scan, jp, ot) in enumerate(cases, 1):
            out = ot if ot is not None else join(out_dir, cid)
            if not args.overwrite and os.path.isfile(out + end):
                cprint(f"[dim][{i}/{n}] skip {cid} (exists)[/dim]")
                continue
            bs, bj = resolve_bl(cid)
            inflight.append((i, cid, out, bs is not None,
                             pool.submit(preprocess_case, scan, jp, pl, cm, dj, bs, bj)))
            if len(inflight) > args.num_workers:
                idx, case_id, ot, bl_case, fut = inflight.popleft()
                gpu(case_id, idx, ot, fut.result(), bl_case)
        while inflight:
            idx, case_id, ot, bl_case, fut = inflight.popleft()
            gpu(case_id, idx, ot, fut.result(), bl_case)
        pool.shutdown(wait=True)
    if prev is not None:
        prev.result()
    ex.shutdown(wait=True)
    if args.gt_dir:
        rows = [score_case(cid, pred, join(args.gt_dir, cid + end), jp) for cid, pred, jp in scored]
        report(rows)
    cprint(f"[green]done — {n} case(s) → {out_dir}[/green]")
    if args.metrics_out: write(rows, args.metrics_out)


if __name__ == "__main__":
    main()
