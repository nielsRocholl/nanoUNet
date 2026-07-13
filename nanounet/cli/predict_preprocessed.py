"""Longi inference on preprocessed b2nd cases: CPU prefetch, GPU batched, no raw pre/post."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p

from nanounet.common import config_table, cprint, nano_header, nano_progress
from nanounet.config import load_config
from nanounet.infer.border_expand import DEFAULT_MAX_BORDER_EXPAND_EXTRA
from nanounet.infer.export import save_preprocessed_seg
from nanounet.infer.predict_case import predict_case_logits
from nanounet.infer.predict_io import check_preprocessed_folder, preprocess_preprocessed_case
from nanounet.infer.predictor import load_net_from_ckpt, pick_checkpoint
from nanounet.plan.labels import labels_from_dataset_json
from nanounet.plan.plans import Plans

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("-i", "--input", required=True, help="preprocessed data_identifier folder")
    ap.add_argument("-o", "--output", required=True, help="output preds folder")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--border-expand", action="store_true")
    ap.add_argument("--max-border-extra", type=int, default=DEFAULT_MAX_BORDER_EXPAND_EXTRA)
    tta_g = ap.add_mutually_exclusive_group()
    tta_g.add_argument("--disable-tta", dest="tta_flag", action="store_false", default=None)
    tta_g.add_argument("--tta", dest="tta_flag", action="store_true", default=None)
    ap.add_argument("--inference-mode", choices=("clustered", "centered"), default="clustered")
    ap.add_argument("--merge", choices=("max", "average"), default="max")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                    help="CPU blosc2+pad prefetch threads (keep GPU fed)")
    ap.add_argument("--cluster-margin-frac", type=float, default=0.1)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    nano_header("nanoUNet predict (preprocessed)", color="blue")
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"Model dir not found: {args.model_dir}\n"
            f"Expected a training run with plans.json, dataset.json, nano_config.json, and a checkpoint.\n"
            f"Fix: pass -m <NANOUNET_RESULTS>/nanounet/<run_dir>   (see docs/steps/predict.md)"
        )
    case_ids = check_preprocessed_folder(args.input)
    md = args.model_dir
    pl = Plans(join(md, "plans.json"))
    cm = pl.get_configuration("3d_fullres")
    dj = load_json(join(md, "dataset.json"))
    cfg = load_config(join(md, "nano_config.json"))
    ckpt_path = pick_checkpoint(md, args.ckpt)
    d = args.device
    if d == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA requested but torch.cuda.is_available() is False.\n"
            "This command is GPU-bound; CPU fallback would be impractically slow.\n"
            "Fix: run on a CUDA node, or pass --device cpu explicitly."
        )
    dev = torch.device(d)
    use_tta = (not cfg.inference.disable_tta_default) if args.tta_flag is None else args.tta_flag
    config_table(
        [("model_dir", md, "cli"), ("ckpt", args.ckpt or ckpt_path, "cli/auto"),
         ("input", args.input, "cli"), ("output", args.output, "cli"),
         ("cases", len(case_ids), "input"), ("device", dev, "cli"),
         ("inference_mode", args.inference_mode, "cli/default"), ("merge", args.merge, "cli/default"),
         ("batch_size", args.batch_size, "cli/default"), ("num_workers", args.num_workers, "cli/default"),
         ("tta", use_tta, "cli/config"), ("border_expand", args.border_expand, "cli")],
        title="nanoUNet predict (preprocessed)",
    )

    net, lm = load_net_from_ckpt(ckpt_path, cm, dj, dev, longi=True)
    labels_from_dataset_json(dj)
    maybe_mkdir_p(args.output)
    end = dj["file_ending"]
    spacing = tuple(float(s) for s in cm.spacing)
    patch_size = tuple(int(x) for x in cm.patch_size)
    todo = []
    for cid in case_ids:
        out_trunc = join(args.output, cid)
        if args.overwrite or not os.path.isfile(out_trunc + end):
            todo.append((cid, out_trunc))
    n_skip = len(case_ids) - len(todo)
    if n_skip:
        cprint(f"[dim]skip {n_skip} case(s) with existing output (pass --overwrite to rerun)[/dim]")
    if not todo:
        cprint(f"[green]done — all {len(case_ids)} case(s) already in {args.output}[/green]")
        return

    timings: list[tuple[str, float]] = []
    t_all = time.perf_counter()

    def gpu_case(idx: int, cid: str, out_trunc: str, pack) -> None:
        t0 = time.perf_counter()
        pad_cpu, slicer_revert, props, fu_xyz, bl_xyz, has_bl = pack
        logits = predict_case_logits(
            net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dj=dj, dev=dev,
            pad=pad_cpu.to(dev, non_blocking=True), slicer_revert=slicer_revert, props=props,
            points_xyz=fu_xyz, encode_prompt=True, use_tta=use_tta,
            border_expand=args.border_expand, max_border_expand_extra=args.max_border_extra,
            batch_size=args.batch_size, use_amp=not args.no_amp,
            cluster_margin_frac=args.cluster_margin_frac, mode=args.inference_mode, merge=args.merge,
            is_longi=True, bl_present=has_bl, bl_points_xyz=bl_xyz if has_bl else None,
        )
        seg = lm.convert_logits_to_segmentation(logits).numpy().astype(np.uint8)
        save_preprocessed_seg(seg, spacing, out_trunc + end)
        dt = time.perf_counter() - t0
        timings.append((cid, dt))

    with nano_progress(len(todo), "predict preprocessed") as advance:
        if len(todo) == 1 or args.num_workers <= 0:
            for i, (cid, out_trunc) in enumerate(todo, 1):
                gpu_case(i, cid, out_trunc, preprocess_preprocessed_case(args.input, cid, patch_size))
                advance()
        else:
            pool = ThreadPoolExecutor(max_workers=args.num_workers)
            inflight: deque = deque()
            for i, (cid, out_trunc) in enumerate(todo, 1):
                inflight.append((i, cid, out_trunc,
                                 pool.submit(preprocess_preprocessed_case, args.input, cid, patch_size)))
                if len(inflight) > args.num_workers:
                    idx, case_id, ot, fut = inflight.popleft()
                    gpu_case(idx, case_id, ot, fut.result())
                    advance()
            while inflight:
                idx, case_id, ot, fut = inflight.popleft()
                gpu_case(idx, case_id, ot, fut.result())
                advance()
            pool.shutdown(wait=True)

    total_s = time.perf_counter() - t_all
    mean_s = sum(t for _, t in timings) / len(timings)
    cprint(
        f"[green]done — {len(todo)} case(s) → {args.output} "
        f"({total_s:.0f}s wall, {mean_s:.1f}s/case mean)[/green]"
    )
    cprint(
        "[dim]next: python3 eval/eval_longi_fu_dice.py --pred-dir <preds> --gt-dir <gt> --meta-dir <meta>[/dim]"
    )


if __name__ == "__main__":
    main()
