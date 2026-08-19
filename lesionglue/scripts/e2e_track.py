"""Holdout instance conversion + timed tracking. Outside product CLIs.

Writes match CSVs and timing.json. Graph vs GPU split via wrapping build_mask_graph.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from tracking.common import cprint, nano_header
from tracking.data.instances import instances_from_nifti
from tracking.data.meta import resolve_track_case
from tracking.data.splits import load_tracking_split
from tracking.infer import load_matcher, mask_has_lesions, track, write_match_csv
import tracking.infer as infer_mod


def _convert(pred: Path, clicks: Path, out: Path) -> None:
    if not pred.is_file():
        raise FileNotFoundError(
            f"No pred at {pred}.\nExpected nanounet_predict output for this stem.\n"
            f"Fix: run nanounet_predict -i inputsTrXX -o preds_xx --patients-csv test_patients.csv"
        )
    if not clicks.is_file():
        raise FileNotFoundError(
            f"No clicks at {clicks}.\nExpected sibling JSON next to the native CT.\n"
            f"Fix: use Longitudinal-CT/inputsTrBL or inputsTrFU"
        )
    instances_from_nifti(pred, clicks, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/nnunet_data/Longitudinal-CT")
    ap.add_argument("--preds-bl", default="")
    ap.add_argument("--preds-fu", default="")
    ap.add_argument("--inst-bl", required=True)
    ap.add_argument("--inst-fu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="/nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt")
    ap.add_argument("--decode", default="dense")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-convert", action="store_true", help="inst dirs already instance-labeled (GT oracle)")
    args = ap.parse_args()
    if not args.skip_convert and not (args.preds_bl and args.preds_fu):
        raise SystemExit(
            "--preds-bl and --preds-fu are required unless --skip-convert.\n"
            "Expected binary nanoUNet folders.\n"
            "Fix: --preds-bl .../preds_bl --preds-fu .../preds_fu"
        )
    nano_header("e2e_track")
    root, pids = Path(args.root), list(map(str, load_tracking_split()["test"]))
    inst_bl, inst_fu = Path(args.inst_bl), Path(args.inst_fu)
    inst_bl.mkdir(parents=True, exist_ok=True)
    inst_fu.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph_s, fwd_s = [], []
    _orig = infer_mod.build_mask_graph

    def _timed(*a, **k):
        t0 = time.perf_counter()
        g = _orig(*a, **k)
        graph_s.append(time.perf_counter() - t0)
        return g

    infer_mod.build_mask_graph = _timed
    matcher = load_matcher(Path(args.ckpt), args.device)
    rows, n_skip = [], 0
    t_all = time.perf_counter()
    for pid in pids:
        gt = resolve_track_case(root, pid)
        if gt is None:
            n_skip += 1
            cprint(f"skip {pid} (missing GT paths)")
            continue
        t_c = time.perf_counter()
        if not args.skip_convert:
            _convert(Path(args.preds_bl) / gt.bl_mask.name, gt.bl_img.parent / (gt.bl_img.name[: -len(".nii.gz")] + ".json"), inst_bl / gt.bl_mask.name)
            _convert(Path(args.preds_fu) / gt.fu_mask.name, gt.fu_img.parent / (gt.fu_img.name[: -len(".nii.gz")] + ".json"), inst_fu / gt.fu_mask.name)
        conv = time.perf_counter() - t_c
        case = resolve_track_case(root, pid, bl_mask_dir=inst_bl, fu_mask_dir=inst_fu)
        if case is None or not mask_has_lesions(case.bl_mask) or not mask_has_lesions(case.fu_mask):
            n_skip += 1
            cprint(f"skip {pid} (empty pred instances)")
            continue
        n_g = len(graph_s)
        t0 = time.perf_counter()
        r = track(case.bl_img, case.bl_mask, case.fu_img, case.fu_mask, case.propagated, Path(args.ckpt), matcher=matcher, decode=args.decode, device=args.device)
        tr = time.perf_counter() - t0
        write_match_csv(out / f"{pid}.csv", r)
        (out / f"{pid}.json").write_text(json.dumps({"bl_ids": [int(x) for x in r.bl_ids], "fu_ids": [int(x) for x in r.fu_ids]}))
        g = graph_s[-1] if len(graph_s) > n_g else 0.0
        fwd = tr - g
        fwd_s.append(fwd)
        rows.append({"pid": pid, "n_bl": int(len(r.bl_ids)), "n_fu": int(len(r.fu_ids)), "n_pairs": int(len(r.pairs)), "convert_s": conv, "graph_s": g, "fwd_s": fwd, "track_s": tr})
        cprint(f"{pid}  bl={len(r.bl_ids)} fu={len(r.fu_ids)} pairs={len(r.pairs)}  conv={conv:.1f}s graph={g:.1f}s fwd={fwd:.3f}s")
    wall = time.perf_counter() - t_all
    summary = {
        "n_ok": len(rows), "n_skip": n_skip, "wall_s": wall,
        "convert_s_mean": float(sum(r["convert_s"] for r in rows) / max(len(rows), 1)),
        "graph_s_mean": float(sum(graph_s) / max(len(graph_s), 1)),
        "fwd_s_mean": float(sum(fwd_s) / max(len(fwd_s), 1)),
        "cases": rows,
    }
    (out.parent / "eval" / "timing_track.json").parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "eval" / "timing_track.json").write_text(json.dumps(summary, indent=2))
    cprint(f"ok={len(rows)} skip={n_skip} wall={wall:.1f}s  graph_mean={summary['graph_s_mean']:.2f}s fwd_mean={summary['fwd_s_mean']:.3f}s")
    cprint(f"wrote {out}  {out.parent / 'eval' / 'timing_track.json'}")


if __name__ == "__main__":
    main()
