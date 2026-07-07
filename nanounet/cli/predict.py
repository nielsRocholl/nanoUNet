"""Dataset / single-case prompt-driven inference with CPU prefetch."""

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
from nanounet.infer.border_expand import DEFAULT_MAX_BORDER_EXPAND_EXTRA
from nanounet.infer.export import export_prediction_from_logits
from nanounet.infer.predict_case import predict_case_logits
from nanounet.infer.predict_io import patient_ids_from_csv, preprocess_case
from nanounet.infer.predictor import load_net_from_ckpt, pick_checkpoint
from nanounet.plan.labels import labels_from_dataset_json
from nanounet.plan.plans import Plans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="folder or single .nii.gz")
    ap.add_argument("-o", "--output", required=True, help="output folder or single .nii.gz")
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--points", default=None, help="points JSON (single mode)")
    ap.add_argument("--baseline-image", default=None, help="sibling BL .nii.gz for two-stream longi inference")
    ap.add_argument("--baseline-points", default=None, help="BL partner points JSON, parallel to --points")
    ap.add_argument("--baseline-image-dir", default=None,
                    help="folder mode: per-case BL .nii.gz, looked up by case id (same stem as -i)")
    ap.add_argument("--baseline-points-dir", default=None,
                    help="folder mode: per-case BL partner points JSON, looked up by case id")
    ap.add_argument("--longi", action="store_true", help="force two-stream net build (else auto-detect from ckpt)")
    ap.add_argument("--no-prompt-encode", action="store_true")
    ap.add_argument("--border-expand", action="store_true")
    ap.add_argument("--max-border-extra", type=int, default=DEFAULT_MAX_BORDER_EXPAND_EXTRA)
    tta_g = ap.add_mutually_exclusive_group()
    tta_g.add_argument("--disable-tta", dest="tta_flag", action="store_false", default=None)
    tta_g.add_argument("--tta", dest="tta_flag", action="store_true", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cluster-margin-frac", type=float, default=0.1)
    ap.add_argument("--inference-mode", choices=("clustered", "centered"), default="clustered",
                    help="patch placement: 'clustered' packs clicks, 'centered' = one patch per click")
    ap.add_argument("--merge", choices=("max", "average"), default="max",
                    help="cross-patch merge: 'max' = union (per-voxel most-foreground patch wins; "
                         "avoids washout), 'average' = legacy gaussian-weighted mean")
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--patients-csv", default=None, help="CSV with patient column; keep cases whose id prefix matches")
    args = ap.parse_args()

    nano_header("nanoUNet predict", color="blue")
    config_table(
        [("model_dir", args.model_dir, "cli"), ("ckpt", args.ckpt or "auto", "cli/default"),
         ("device", args.device, "cli/default"), ("inference_mode", args.inference_mode, "cli/default"),
         ("merge", args.merge, "cli/default"), ("batch_size", args.batch_size, "cli/default"),
         ("tta", "auto" if args.tta_flag is None else args.tta_flag, "cli/config")],
        title="nanoUNet predict",
    )
    md = args.model_dir
    pl = Plans(join(md, "plans.json"))
    cm = pl.get_configuration("3d_fullres")
    dj = load_json(join(md, "dataset.json"))
    cfg = load_config(join(md, "nano_config.json"))
    labels_from_dataset_json(dj)

    if args.baseline_points and not args.baseline_image:
        raise SystemExit("--baseline-points requires --baseline-image")
    if args.baseline_image and not args.baseline_points:
        raise SystemExit("--baseline-image requires --baseline-points")
    if args.baseline_points_dir and not args.baseline_image_dir:
        raise SystemExit("--baseline-points-dir requires --baseline-image-dir")
    if args.baseline_image_dir and not args.baseline_points_dir:
        raise SystemExit("--baseline-image-dir requires --baseline-points-dir")
    if args.baseline_image_dir and args.baseline_image:
        raise SystemExit("use either --baseline-image or --baseline-image-dir, not both")

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

    n = len(cases)

    def out_trunc_for(ot: str | None, case_id: str) -> str:
        return ot if ot is not None else join(out_dir, case_id)

    def skip_case(case_id: str, idx: int, out_trunc: str) -> bool:
        if args.overwrite or not os.path.isfile(out_trunc + end):
            return False
        cprint(f"[dim][{idx}/{n}] skip {case_id} (exists)[/dim]")
        return True

    def gpu_export(case_id: str, idx: int, out_trunc: str, pack) -> None:
        t0 = time.perf_counter()
        pad_cpu, slicer_revert, props, points_xyz, bl_pack = pack
        kw = dict(
            net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dj=dj, dev=dev,
            pad=pad_cpu.to(dev), slicer_revert=slicer_revert, props=props, points_xyz=points_xyz,
            encode_prompt=not args.no_prompt_encode, use_tta=use_tta,
            border_expand=args.border_expand, max_border_expand_extra=args.max_border_extra,
            batch_size=args.batch_size, use_amp=not args.no_amp,
            cluster_margin_frac=args.cluster_margin_frac,
            mode=args.inference_mode,
            merge=args.merge,
        )
        if bl_pack is not None:
            pad_bl, bl_slicer, bl_props, bl_pts = bl_pack
            kw.update(pad_bl=pad_bl.to(dev), bl_points_xyz=bl_pts, bl_props=bl_props, bl_slicer_revert=bl_slicer)
        logits = predict_case_logits(**kw)
        export_prediction_from_logits(logits, props, cm, pl, dj, out_trunc)
        cprint(f"[bold green][{idx}/{n}] {case_id} ({time.perf_counter() - t0:.1f}s)[/bold green]")

    def consume(idx: int, case_id: str, out_trunc: str, pack) -> None:
        gpu_export(case_id, idx, out_trunc, pack)

    def bl_paths_for(cid: str) -> tuple[str | None, str | None]:
        if args.baseline_image_dir:
            bl_scan = join(args.baseline_image_dir, cid + end)
            bl_json = join(args.baseline_points_dir, cid + ".json")
            if not os.path.isfile(bl_scan):
                raise FileNotFoundError(f"--baseline-image-dir missing case: {bl_scan}")
            if not os.path.isfile(bl_json):
                raise FileNotFoundError(f"--baseline-points-dir missing case: {bl_json}")
            return bl_scan, bl_json
        return args.baseline_image, args.baseline_points

    if n == 1 or args.num_workers <= 0:
        for i, (cid, scan, jp, ot) in enumerate(cases, 1):
            out = out_trunc_for(ot, cid)
            if skip_case(cid, i, out):
                continue
            bl_scan, bl_json = bl_paths_for(cid)
            consume(i, cid, out, preprocess_case(scan, jp, pl, cm, dj, bl_scan, bl_json))
        cprint(f"[green]done — {n} case(s) → {out_dir}[/green]")
        return

    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    inflight: deque = deque()
    for i, (cid, scan, jp, ot) in enumerate(cases, 1):
        out = out_trunc_for(ot, cid)
        if skip_case(cid, i, out):
            continue
        bl_scan, bl_json = bl_paths_for(cid)
        inflight.append((i, cid, out, pool.submit(preprocess_case, scan, jp, pl, cm, dj, bl_scan, bl_json)))
        if len(inflight) > args.num_workers:
            idx, case_id, ot, fut = inflight.popleft()
            consume(idx, case_id, ot, fut.result())
    while inflight:
        idx, case_id, ot, fut = inflight.popleft()
        consume(idx, case_id, ot, fut.result())
    pool.shutdown(wait=True)
    cprint(f"[green]done — {n} case(s) → {out_dir}[/green]")


if __name__ == "__main__":
    main()
