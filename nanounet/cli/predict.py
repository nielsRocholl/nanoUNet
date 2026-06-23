"""Dataset / single-case prompt-driven inference with CPU prefetch."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p

from nanounet.common import cprint, nano_header
from nanounet.config import load_config
from nanounet.infer.border_expand import DEFAULT_MAX_BORDER_EXPAND_EXTRA
from nanounet.infer.export import export_prediction_from_logits
from nanounet.infer.predict_case import predict_case_logits
from nanounet.infer.predictor import load_net_from_ckpt, pick_checkpoint
from nanounet.plan.case_pp import run_case
from nanounet.plan.labels import labels_from_dataset_json
from nanounet.plan.plans import Plans
from nanounet.prompt.coords import load_points_xyz


def _preprocess_case(scan: str, json_path: str, pl, cm, dj):
    data, _seg, props = run_case([scan], None, pl, cm, dj, verbose=False)
    data_t = torch.from_numpy(data).float()
    pad, slicer_revert = pad_nd_image(data_t, tuple(cm.patch_size), "constant", {"value": 0}, True, None)
    return pad, slicer_revert, props, load_points_xyz(json_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="folder or single .nii.gz")
    ap.add_argument("-o", "--output", required=True, help="output folder or single .nii.gz")
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--points", default=None, help="points JSON (single mode)")
    ap.add_argument("--no-prompt-encode", action="store_true")
    ap.add_argument("--border-expand", action="store_true")
    ap.add_argument("--max-border-extra", type=int, default=DEFAULT_MAX_BORDER_EXPAND_EXTRA)
    tta_g = ap.add_mutually_exclusive_group()
    tta_g.add_argument("--disable-tta", dest="tta_flag", action="store_false", default=None)
    tta_g.add_argument("--tta", dest="tta_flag", action="store_true", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cluster-margin-frac", type=float, default=0.1)
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    nano_header("nanoUNet predict", color="blue")
    md = args.model_dir
    pl = Plans(join(md, "plans.json"))
    cm = pl.get_configuration("3d_fullres")
    dj = load_json(join(md, "dataset.json"))
    cfg = load_config(join(md, "nano_config.json"))
    labels_from_dataset_json(dj)

    d = args.device
    if d == "cuda" and not torch.cuda.is_available():
        d = "cpu"
    if d == "mps" and not torch.backends.mps.is_available():
        d = "cpu"
    dev = torch.device(d)
    net, lm = load_net_from_ckpt(pick_checkpoint(md, args.ckpt), cm, dj, dev)
    use_tta = (not cfg.inference.disable_tta_default) if args.tta_flag is None else args.tta_flag

    end = dj["file_ending"]
    single_mode = not os.path.isdir(args.input)
    if not single_mode:
        case_files = sorted(f for f in os.listdir(args.input) if f.endswith(end))
        cases = [(f[:-len(end)], join(args.input, f), join(args.input, f[:-len(end)] + ".json"), None) for f in case_files]
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

    def gpu_export(case_id: str, idx: int, out_trunc: str, pad_cpu, slicer_revert, props, points_xyz) -> None:
        if not args.overwrite and os.path.isfile(out_trunc + end):
            cprint(f"[dim][{idx}/{n}] skip {case_id} (exists)[/dim]")
            return
        t0 = time.perf_counter()
        logits = predict_case_logits(
            net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dj=dj, dev=dev,
            pad=pad_cpu.to(dev), slicer_revert=slicer_revert, props=props, points_xyz=points_xyz,
            encode_prompt=not args.no_prompt_encode, use_tta=use_tta,
            border_expand=args.border_expand, max_border_expand_extra=args.max_border_extra,
            batch_size=args.batch_size, use_amp=not args.no_amp,
            cluster_margin_frac=args.cluster_margin_frac,
        )
        export_prediction_from_logits(logits, props, cm, pl, dj, out_trunc)
        cprint(f"[bold green][{idx}/{n}] {case_id} ({time.perf_counter() - t0:.1f}s)[/bold green]")

    def consume(idx: int, case_id: str, out_trunc: str, pack) -> None:
        gpu_export(case_id, idx, out_trunc, *pack)

    if n == 1 or args.num_workers <= 0:
        for i, (cid, scan, jp, ot) in enumerate(cases, 1):
            consume(i, cid, ot or join(out_dir, cid), _preprocess_case(scan, jp, pl, cm, dj))
        return

    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    inflight: deque = deque()
    for i, (cid, scan, jp, _) in enumerate(cases, 1):
        inflight.append((i, cid, join(out_dir, cid), pool.submit(_preprocess_case, scan, jp, pl, cm, dj)))
        if len(inflight) > args.num_workers:
            idx, case_id, ot, fut = inflight.popleft()
            consume(idx, case_id, ot, fut.result())
    while inflight:
        idx, case_id, ot, fut = inflight.popleft()
        consume(idx, case_id, ot, fut.result())
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
