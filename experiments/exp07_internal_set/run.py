# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
"""exp07 - Internal private set: our pipeline against Kirchhoff et al.  (paper: Experiments > Private sets > "Same centre"; Table 'nine experiments' row 7)

QUESTION   Does the system generalise to unseen patients of the same centre? Masks (Dice, NSD, detection) and lesion identity (recall per
           class, edge F1) for our pipeline against the prompted longitudinal segmenter of Kirchhoff et al. (LongiSeg), both driven by the
           same propagated points on the same scan pairs. exp08 runs the very same protocol on a new centre (site tag only).
WHY        Fills the private-set rows of the paper. The data never leaves the partner lab: this file, the two model folders and the owner's
           container are what travels; results.json (anonymised) is what comes back.
DATA       Any folder in the Longitudinal-CT layout (`--data-root`): inputsTrBL|FU/<pid>_<idx>.nii.gz + .json (BL = true lesion points, FU =
           PROPAGATED points, made with the same propagation the public dataset shipped with: this code never registers), targetsTrBL|FU/
           (instance masks, voxel value = lesion id), meta/<pid>.csv, and a patient list CSV with a `patient` column (`--patients-csv`). One
           scan pair per patient = the dominant follow-up region (most meta rows), as everywhere in the paper. Default = the Longitudinal-CT
           held-out 60 (used to validate the protocol before it leaves the machine). Headline excludes lesions flagged `linking_unclear`.
METHOD     1. Ours (`experiments/pipeline.py`, setting B): BL = annotated instance masks, FU = segmented by our promptable model from the
              propagated points, matcher = `--matcher-ckpt` (EMA weights), segmenter = `--seg-ckpt` (EMA), decoders hungarian and sinkhorn at
              the matcher checkpoint's own dust_tau. Must equal exp05 setting B on the same patients.
           2. Kirchhoff (`kirchhoff.py`): LongiSeg's `predict_case` in-process, as in /nnunet_data/LongiSeg/scripts/predict_nanounet_testset.py
              (BL picked by click-id overlap, folds 0-4 of `--longiseg-model`). Identity is inherited from the prompt: prompt i -> the FU lesion
              segmented from it; `new` is undefined for it (null): a lesion no prompt reaches is not a node, one a prompt reaches is linked.
           3. Masks: per annotated FU lesion that carries a prompt (same lesion set for both methods): Dice, NSD at 1 mm, detection = IoU > 0.1,
              prediction credited = the connected component with the largest overlap (`scoring.score_lesions`, the nanoUNet rule); lesions
              pooled over patients, patient-level bootstrap (B=10000).
           4. Identity: `scoring.score_pair` per class + identity ceiling + edge P/R/F1, patient-level bootstrap CIs and paired deltas
              (ours - Kirchhoff) on the same patients. Decoding is separate from prediction: `--rescore` reruns only scoring.
           5. Container rules: every path is a flag, no network, nothing written outside `--out-root`; `--anonymize` (default on) replaces
              patient ids by a salted hash everywhere (files, tables, log) and keeps no coordinates; the hash -> id table and the salt stay in
              `artifacts/` at the lab; `--validate-only` checks layout, models and environment at t=0 and lists every problem with a Fix.
OUTPUT     results.json tables: per_patient (patient, method, decoder, variant, status, error, timings, class counts ok/tot/ceil, tp/fp/fn,
           node ids, predicted links in annotated-id space; Kirchhoff also the prompt -> lesion assignment), per_lesion (patient, method,
           lesion id, topology, Dice, NSD, IoU, hit), metrics (every CI), deltas. artifacts/: pipeline.json, salt.txt, patient_map.csv,
           records/, scores/ (raw matcher scores), masks/ (our FU instances), kirchhoff/ (LongiSeg instance predictions).
COMMAND    python -m experiments.exp07_internal_set.run --tag paper_v1
DEPENDS ON experiments/common.py, scoring.py, segment.py (via pipeline.py), pipeline.py; kirchhoff.py (this folder); the LongiSeg source and
           model; the matcher and segmenter checkpoints. exp08 and exp09 import this folder.
RUNTIME    About 1 min per patient for ours (segmentation dominates) plus about 2.2 s per lesion and ~15 s model loading for Kirchhoff: one to
           two hours for 60 patients on one A100. Resumable (--resume skips finished units and retries failed ones); --rescore takes a minute.
CAVEATS    NUMBERS ARE NOT MEANINGFUL until the matcher is retrained on the fixed graph cache (experiments plan Sec. 2) and `MATCHER_FINAL` is
           repointed: the current checkpoint never saw a merge-target node, merge recall is 0 by construction. A patient either method cannot
           process stays as status failed and counts as fully missed in identity (its mask lesions are unknown and absent from the mask
           table). The LongiSeg model is research-only (Longitudinal-CT, KiTS23, LiTS, PanTS licences). `predict_case` needs a CUDA device.
           Kirchhoff's `new` class is null by construction, its macro recall averages the three defined classes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import secrets
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from core.ui import cprint
from experiments import pipeline as P
from experiments import segment as SEG
from experiments.common import LONGISEG_DIR, LONGISEG_FIX, LONGI_ROOT, LONGISEG_MODEL, MATCHER_FINAL, SEG_CKPT, abort_if, add_common_args, limited, missing_paths, problem, start_run
from experiments.exp07_internal_set import kirchhoff as K
from segtrack.case import load_instance_zyx
from experiments.scoring import COUNT_KEYS, DEFINITIONS, META_COLUMNS, METRICS, bootstrap, bootstrap_stat, load_pairs, paired_delta, paired_delta_stat, score_lesions, score_pair

EXP = "exp07_internal_set"
PAPER = {"section": "Experiments > Private sets > Same centre", "table_row": 7, "supports": "internal private set: masks and identity, ours vs Kirchhoff et al."}
METHODS = ("ours", "kirchhoff")
LAYOUT = ("inputsTrBL", "inputsTrFU", "targetsTrBL", "targetsTrFU", "meta")
HEAD = ("recall_unchanged", "recall_disappeared", "recall_new", "recall_merged", "recall_macro", "edge_f1")
CEIL = ("ceiling_unchanged", "ceiling_disappeared", "ceiling_new", "ceiling_merged")
MASK_STATS = {"dsc": 0, "nsd": 1, "detection": 2}
NODE_SUPPLY = "Lstar BL (annotated masks), Lhat FU (segmented from the propagated points)"


def patient_key(pid: str, salt: str, anonymize: bool) -> str:
    """Id used in every output: the patient id, or the first 10 hex digits of sha256(salt + id) under --anonymize."""
    return hashlib.sha256((salt + pid).encode()).hexdigest()[:10] if anonymize else pid


def scrub(text: str, pid: str, key: str) -> str:
    """Error text with the real patient id replaced by its output key (paths in messages would otherwise leak it)."""
    return text.replace(pid, key)


def header_info(path: Path) -> tuple[tuple, tuple]:
    """(size xyz, spacing xyz) of an image from its header only."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return r.GetSize(), r.GetSpacing()


def layout_problems(root: Path, pid: str) -> list[str]:
    """Everything wrong with one patient's files in the Longitudinal-CT layout (empty list = ok); reads headers only."""
    meta = root / "meta" / f"{pid}.csv"
    if not meta.is_file():
        return [f"meta/{pid}.csv is missing"]
    header = pd.read_csv(meta, nrows=0).columns
    bad = [f"meta/{pid}.csv lacks column {c}" for c in META_COLUMNS if c not in header]
    if bad:
        return bad
    pair = P.longitudinal_pair(root, pid)
    out = [f"{p.relative_to(root)} is missing" for p in (pair.bl_img, pair.fu_img, pair.bl_clicks, pair.fu_clicks, pair.bl_mask, pair.fu_mask) if not p.is_file()]
    for c in (pair.bl_clicks, pair.fu_clicks):
        try:
            K.load_clicks(c) if c.is_file() else None
        except (ValueError, KeyError, TypeError) as e:
            out.append(f"{c.relative_to(root)} is not a click file ({type(e).__name__}: {e})")
    for img, mask in ((pair.bl_img, pair.bl_mask), (pair.fu_img, pair.fu_mask)):
        if img.is_file() and mask.is_file() and header_info(img)[0] != header_info(mask)[0]:
            out.append(f"{mask.relative_to(root)} and {img.relative_to(root)} differ in size (masks must be on the scan's own grid)")
    return out


def startup_problems(args: argparse.Namespace, pids: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """(global problems, per-patient layout problems); every problem carries its Fix (E6). Global ones abort a run, per-patient ones do not."""
    probs = missing_paths({"data root": args.data_root}, "pass --data-root <folder in the Longitudinal-CT layout>")
    probs += [problem(f"{args.data_root / d}/ is missing", "the Longitudinal-CT layout " + "|".join(LAYOUT), f"create {d}/ with the files described in the exp07 docstring") for d in LAYOUT if args.data_root.is_dir() and not (args.data_root / d).is_dir()]
    if args.methods and "ours" in args.methods:
        probs += missing_paths({"segmenter checkpoint (--seg-ckpt)": args.seg_ckpt, "matcher checkpoint (--matcher-ckpt)": args.matcher_ckpt}, "pass the checkpoint file from the model package")
        seg_dir = args.seg_ckpt.parent.parent
        probs += [problem(f"segmenter model file {seg_dir / f} is missing", "<model dir>/{plans.json,dataset.json,nano_config.json} two levels above the checkpoint (<model dir>/finetune/x.ckpt)", "keep the model folder structure intact or pass --seg-ckpt <model dir>/finetune/<file>.ckpt")
                  for f in ("plans.json", "dataset.json", "nano_config.json") if args.seg_ckpt.is_file() and not (seg_dir / f).is_file()]
    if "kirchhoff" in args.methods:
        probs += missing_paths({"LongiSeg model folder (--longiseg-model)": args.longiseg_model}, "pass the folder containing plans.json, dataset.json and fold_0..fold_4")
        probs += [problem(f"{what} is missing: {path}", "a complete LongiSeg model folder (plans.json, dataset.json, fold_0..4/checkpoint_final.pth)", "copy the whole model folder again") for what, path in (K.model_problems(args.longiseg_model) if args.longiseg_model.is_dir() else [])]
        if importlib.util.find_spec("longiseg") is None and not (LONGISEG_DIR / "longiseg").is_dir():
            probs.append(problem("python package longiseg is not importable", "LongiSeg source on PYTHONPATH (same Python environment as nanoUNet)", f"export PYTHONPATH=<LongiSeg checkout>:$PYTHONPATH"))
        if importlib.util.find_spec("difference_weighting") is None:
            probs.append(problem("python package difference_weighting is not installed", "difference_weighting 0.1.0 (LongiSeg dependency)", LONGISEG_FIX))
    if not args.patients_csv.is_file():
        probs.append(problem(f"patient list not found: {args.patients_csv}", "a CSV with a `patient` column", "pass --patients-csv <file> or --patients <ids...>"))
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            probs.append(problem(f"--device {args.device} is not available", "a CUDA GPU (segmenter and LongiSeg need one)", "run on the GPU node (docker run --gpus all ...)"))
    parent = next(p for p in (args.out_root, *args.out_root.parents) if p.exists())
    if not os.access(parent, os.W_OK):
        probs.append(problem(f"--out-root {args.out_root} is not writable (nearest existing folder {parent})", "a writable folder", "pass --out-root <writable folder>"))
    per_patient = {pid: layout_problems(args.data_root, pid) for pid in pids} if args.data_root.is_dir() and all((args.data_root / d).is_dir() for d in LAYOUT) else {}
    return probs, {pid: p for pid, p in per_patient.items() if p}


def predict_unit(pl: P.Pipeline | None, args: argparse.Namespace, pid: str, key: str, method: str, art: Path) -> dict:
    """Run one (patient, method), write scores / masks / predictions into artifacts; a failure is recorded, never dropped (plan Sec. 4, rule 3)."""
    rec_path = art / "records" / f"{key}_{method}.json"
    if rec_path.is_file() and json.loads(rec_path.read_text())["status"] == "ok":
        return json.loads(rec_path.read_text())  # resume: finished units are skipped, failed ones retried
    rec = {"patient": key, "method": method, "status": "ok", "error": None}
    try:
        pair = P.longitudinal_pair(args.data_root, pid)
        rec["fu_click_ids"] = sorted(K.load_clicks(pair.fu_clicks))
        if method == "ours":
            s, scans = P.run_setting(pl, args.data_root, pair, "B")
            P.save_scores(art / "scores" / f"{key}.npz", s)
            P.write_instances(art / "masks" / f"{key}_ours_fu.mha", scans["fu"]["inst"], scans["fu"]["props"])
            rec.update(t_seg=s.t_seg, t_track=s.t_track)
        else:
            rec.update(K.predict(args.longiseg_model, pid, pair.fu_img, pair.fu_clicks, pair.bl_img.parent, pair.bl_mask.parent, art / "kirchhoff" / f"{key}.nii.gz"))
            rec["bl_stem"] = "same" if rec["bl_stem"] == pair.stem else "other"  # the stem itself would leak the patient id
    except (OSError, ValueError, KeyError, IndexError, AssertionError, RuntimeError, SystemExit) as e:  # a patient a method cannot process stays as status=failed, counted as missed
        rec.update(status="failed", error=scrub(f"{type(e).__name__}: {e}", pid, key))
        cprint(f"[red]failed[/red] {key} {method}: {rec['error']}")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec))
    return rec


def read_record(art: Path, key: str, method: str) -> dict:
    p = art / "records" / f"{key}_{method}.json"
    return json.loads(p.read_text()) if p.is_file() else {"patient": key, "method": method, "status": "missing", "error": "no record: the run never reached this patient"}


def lesion_rows(key: str, method: str, case, pred_fg: np.ndarray, gt: np.ndarray, spacing: tuple, ids: list[int]) -> list[dict]:
    """Per-lesion mask metrics of one prediction: one row per annotated FU lesion that carries a prompt (same lesion set for both methods)."""
    if pred_fg.shape != gt.shape:
        raise ValueError(f"{key} {method}: prediction {pred_fg.shape} and annotated FU mask {gt.shape} differ in shape\n     Fix: targetsTrFU masks must be on the scan's native grid")
    return [{"patient": key, "method": method, "lesion_id": r["id"], "topology": case.topology.get(r["id"]), "unclear": r["id"] in case.unclear, "dsc": r["dsc"], "nsd": r["nsd"], "iou": r["iou"], "hit": r["hit"]}
            for r in score_lesions(pred_fg, gt, spacing, ids)]


def kirchhoff_counts(case, pred: np.ndarray | None, gt: np.ndarray, rec: dict, excl: bool) -> tuple[dict, dict]:
    """Identity counts of one Kirchhoff unit. Class `new` is undefined for it: its ok/tot/ceil are zeroed (the real total is kept as new_tot_unreported)."""
    if rec["status"] != "ok":
        c, assign = score_pair(replace(case, found_bl=set(), found_fu=set(), pred_links=set()), exclude_unclear=excl), None
    else:
        filled, assign = K.identity(case, pred, gt, rec["prompt_ids"])
        c = score_pair(filled, exclude_unclear=excl)
    return {**c, "new_ok": 0, "new_ceil": 0, "new_tot": 0, "new_tot_unreported": c["new_tot"]}, assign


def score(art: Path, pids: list[str], keys: dict[str, str], methods: list[str], root: Path, tau: float) -> tuple[list[dict], list[dict], dict, dict]:
    """Pure CPU: identity counts (both decoders for ours, prompt inheritance for Kirchhoff) and per-lesion mask metrics from the stored artifacts."""
    cases, rows, lesions, counts, masks = load_pairs(root, pids, include_unclear=True), [], [], {}, {m: {} for m in methods}
    for pid in pids:
        key, case, pair = keys[pid], cases[pid], P.longitudinal_pair(root, pid)
        recs = {m: read_record(art, key, m) for m in methods}
        gt = spacing = None
        if any(r["status"] == "ok" for r in recs.values()):
            gt, sp = load_instance_zyx(pair.fu_mask)[0], header_info(pair.fu_img)[1]
            spacing = (sp[2], sp[1], sp[0])
        for m in methods:
            rec, ok = recs[m], recs[m]["status"] == "ok"
            base = {"patient": key, "method": m, "status": rec["status"], "error": rec["error"], "t_seg": rec.get("t_seg"), "t_track": rec.get("t_track"), "t_kirchhoff": rec.get("t_kirchhoff")}
            masks[m][key] = []
            pred = None
            if m == "ours":
                s = P.load_scores(art / "scores" / f"{key}.npz") if ok else None
                pred = read_instances(art / "masks" / f"{key}_ours_fu.mha") if ok else None
                for dec in P.DECODERS:
                    for variant, excl in (("headline", True), ("with_unclear", False)):
                        c = P.score_scores(case, s, dec, tau, exclude_unclear=excl)
                        counts.setdefault((m, dec, variant), {})[key] = c
                        row = {**base, "decoder": dec, "variant": variant, **c}
                        if variant == "headline" and s is not None:
                            row.update(n_bl_nodes=len(s.bl_ids), n_fu_nodes=len(s.fu_ids), n_fu_extra=int((s.fu_ann == 0).sum()), bl_nodes=[[int(i), int(a)] for i, a in zip(s.bl_ids, s.bl_ann)],
                                       fu_nodes=[[int(i), int(a)] for i, a in zip(s.fu_ids, s.fu_ann)], pred_links=sorted([int(b), int(f)] for b, f in P.decode_links(s, dec, tau)))
                        rows.append(row)
            else:
                kpath = art / "kirchhoff" / f"{key}.nii.gz"
                if ok and rec["predict_status"] == "ok" and not kpath.is_file():
                    raise FileNotFoundError(f"Kirchhoff prediction of {key} is missing: {kpath}\n     Fix: rerun with --resume to regenerate it")
                pred = K.read_prediction(kpath) if ok and rec["predict_status"] == "ok" else None
                for variant, excl in (("headline", True), ("with_unclear", False)):
                    c, assign = kirchhoff_counts(case, pred, gt, rec, excl)
                    counts.setdefault((m, "prompt", variant), {})[key] = c
                    rows.append({**base, "decoder": "prompt", "variant": variant, "predict_status": rec.get("predict_status"), **c, **({"assignment": assign} if variant == "headline" else {})})
            if ok:
                fg = pred > 0 if pred is not None else np.zeros(gt.shape, bool)
                les = lesion_rows(key, m, case, fg, gt, spacing, rec["fu_click_ids"])
                lesions += les
                masks[m][key] = [(r["dsc"], r["nsd"], r["hit"]) for r in les if not r["unclear"]]
    return rows, lesions, counts, masks


def read_instances(path: Path) -> np.ndarray:
    """Our FU instance prediction (Z, Y, X) int32 back from the .mha written at predict time."""
    img = sitk.ReadImage(str(path))  # hold the image while copying the array (never inline a view of a temporary)
    return np.rint(sitk.GetArrayFromImage(img)).astype(np.int32)


def mean_of(col: int):
    def stat(items: list) -> float:
        v = [x[col] for x in items if x[col] == x[col]]
        return float(np.mean(v)) if v else float("nan")
    return stat


def summarise(counts: dict, masks: dict, methods: list[str]) -> tuple[list[dict], list[dict], dict]:
    """Bootstrap CIs (patient level) of every identity and mask metric per method, paired deltas ours - Kirchhoff on the same patients."""
    metrics, deltas, summary = [], [], {"identity": {}, "mask": {}, "delta": {}}
    for (m, dec, variant), by_pid in counts.items():
        ci = bootstrap(by_pid)
        metrics += [{"method": m, "decoder": dec, "variant": variant, "metric": name, "point": p, "lo": lo, "hi": hi, "n_patients": len(by_pid)} for name, (p, lo, hi) in ci.items()]
        if variant == "headline":
            summary["identity"][f"{m}/{dec}"] = {name: list(ci[name]) for name in HEAD + CEIL}
    for m in methods:
        ci = {s: bootstrap_stat(masks[m], mean_of(i)) for s, i in MASK_STATS.items()}
        metrics += [{"method": m, "decoder": None, "variant": "headline", "metric": s, "point": p, "lo": lo, "hi": hi, "n_patients": len(masks[m])} for s, (p, lo, hi) in ci.items()]
        summary["mask"][m] = {s: list(v) for s, v in ci.items()}
    if set(methods) == set(METHODS):
        for dec in P.DECODERS:
            d = paired_delta(counts[("ours", dec, "headline")], counts[("kirchhoff", "prompt", "headline")])
            deltas += [{"a": f"ours/{dec}", "b": "kirchhoff/prompt", "metric": name, "delta": p, "lo": lo, "hi": hi} for name, (p, lo, hi) in d.items()]
            summary["delta"][f"ours_{dec}_minus_kirchhoff"] = {name: list(d[name]) for name in ("recall_macro", "edge_f1")}
        for s, i in MASK_STATS.items():
            p, lo, hi = paired_delta_stat(masks["ours"], masks["kirchhoff"], mean_of(i))
            deltas.append({"a": "ours", "b": "kirchhoff", "metric": s, "delta": p, "lo": lo, "hi": hi})
            summary["delta"].setdefault("mask_ours_minus_kirchhoff", {})[s] = [p, lo, hi]
    return metrics, deltas, summary


def fmt(v) -> str:
    return "n/a" if v is None or v[0] is None or v[0] != v[0] else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def markdown(exp: str, site: str, summary: dict, notes: list[str]) -> str:
    out = [f"# {exp} ({site})", "", "NUMBERS ARE NOT MEANINGFUL until the matcher is retrained on the fixed graph cache (see CAVEATS in run.py).", "",
           "## Masks (per annotated FU lesion that carries a prompt; lesions pooled, patient-level bootstrap 95 % CI)", "", "| method | Dice | NSD@1mm | detection (IoU > 0.1) |", "|---|---|---|---|"]
    out += [f"| {m} | {fmt(v['dsc'])} | {fmt(v['nsd'])} | {fmt(v['detection'])} |" for m, v in summary["mask"].items()]
    out += ["", "## Identity (recall per class, edge F1 and identity ceiling; Kirchhoff `new` is undefined)", "",
            "| method / decoder | unchanged | disappeared | new | merged | macro | edge F1 | ceil unch. | ceil disapp. | ceil new | ceil merged |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {k} | " + " | ".join(fmt(v[m]) for m in HEAD + CEIL) + " |" for k, v in summary["identity"].items()]
    out += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(out)


def main(exp: str = EXP, paper: dict = PAPER, doc: str | None = __doc__, site: str = "internal set, same centre") -> None:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, rescore=True)
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="dataset root in the Longitudinal-CT layout (inputsTrBL|FU, targetsTrBL|FU, meta)")
    ap.add_argument("--patients-csv", type=Path, default=None, help="CSV with a `patient` column (default: <data root>/test_patients.csv)")
    ap.add_argument("--patients", nargs="+", default=None, help="explicit patient ids instead of --patients-csv (debugging; ids then appear in command.txt)")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS), help="systems to run: ours (nanoUNet + LesionGlue) and/or kirchhoff (LongiSeg)")
    ap.add_argument("--seg-ckpt", type=Path, default=SEG_CKPT, help="our segmenter checkpoint <model dir>/finetune/<file>.ckpt (EMA weights; plans.json etc. are read from <model dir>)")
    ap.add_argument("--matcher-ckpt", type=Path, default=MATCHER_FINAL, help="our matcher checkpoint (EMA weights); the owner repoints the default after the retrain")
    ap.add_argument("--longiseg-model", type=Path, default=LONGISEG_MODEL, help="LongiSeg model folder (plans.json, dataset.json, fold_0..fold_4); research-only weights")
    ap.add_argument("--anonymize", action=argparse.BooleanOptionalAction, default=True, help="replace patient ids by a salted hash in every output (files, tables, log); coordinates are never stored")
    ap.add_argument("--salt", default=None, help="salt of the patient hash (default: random, kept in artifacts/salt.txt so --resume and --rescore reproduce the ids)")
    ap.add_argument("--validate-only", action="store_true", help="check layout, models and environment, report every problem with its Fix, run nothing")
    ap.add_argument("--tau", type=float, default=None, help="decoder cut-off; default: the matcher checkpoint's own dust_tau (only change it to rescore a sensitivity row)")
    args = ap.parse_args()
    rescore = args.rescore is not None
    args.patients_csv = args.patients_csv or args.data_root / "test_patients.csv"
    src_art = args.rescore / "artifacts" if rescore else None
    if rescore:  # the source run fixes the methods, the anonymisation and the tau; the patient list comes from its patient_map.csv
        info = json.loads((src_art / "pipeline.json").read_text())
        args.anonymize, args.methods = info["anonymize"], info["methods"]
        pids = [r["pid"] for r in csv.DictReader(open(src_art / "patient_map.csv", newline=""))]
    else:
        pids = args.patients if args.patients is not None else (sorted(pd.read_csv(args.patients_csv)["patient"].astype(str)) if args.patients_csv.is_file() else [])
    pids = limited(pids, args)
    problems, by_patient = ([], {}) if rescore else startup_problems(args, pids)
    if rescore:
        problems = missing_paths({"data root": args.data_root}, "pass --data-root")
    if args.validate_only:
        all_p = problems + [problem(f"patient {pid}: {what}", "a complete scan pair in the Longitudinal-CT layout", "restore the file or drop the patient from the list") for pid, ws in by_patient.items() for what in ws]
        abort_if(all_p)
        cprint(f"[green]layout ok[/green]: {len(pids)} patients, methods {args.methods}, nothing was run")
        sys.stdout.write(json.dumps({"exp": exp, "status": "validated", "n_patients": len(pids)}) + "\n")
        return
    abort_if(problems + ([problem(f"every one of the {len(pids)} patients has layout problems, e.g. {next(iter(by_patient.values()))[:2]}", "files in the Longitudinal-CT layout under --data-root", "run with --validate-only and fix every listed problem")] if pids and len(by_patient) == len(pids) else [])
             + ([] if pids else [problem("no patients selected", "at least one patient", "check --patients-csv / --limit-patients")]))
    inputs = {"patients": args.patients_csv} if not rescore else {}
    if not rescore:
        inputs.update({"matcher": args.matcher_ckpt, "segmenter": args.seg_ckpt} if "ours" in args.methods else {})
        inputs.update({"longiseg_model": args.longiseg_model} if "kirchhoff" in args.methods else {})
    run = start_run(exp, ap, args, inputs=inputs, paper=paper)
    art, salt_file = run.artifacts, run.artifacts / "salt.txt"
    if not rescore and not salt_file.is_file():
        salt_file.write_text(args.salt or secrets.token_hex(8))
    keys = {p: patient_key(p, (src_art / "salt.txt" if rescore else salt_file).read_text(), args.anonymize) for p in pids}
    if not rescore:
        with open(art / "patient_map.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pid", "key"])
            w.writerows((p, keys[p]) for p in pids)
        for pid, ws in by_patient.items():
            cprint(f"[yellow]layout[/yellow] {keys[pid]}: {'; '.join(scrub(w, pid, keys[pid]) for w in ws)}  (kept: it will be recorded as failed and counted as missed)")
    if rescore:
        art = src_art
        tau = args.tau if args.tau is not None else info["tau"]
    else:
        SEG.SEG_CKPT, SEG.SEG_MODEL_DIR = args.seg_ckpt, args.seg_ckpt.parent.parent  # load_segmenter reads these module constants; --seg-ckpt repoints them
        pl = P.load_pipeline(args.device, matcher_ckpt=args.matcher_ckpt, segmenter=True) if "ours" in args.methods else None
        tau = args.tau if args.tau is not None else (pl.tau if pl else 0.0)
        (art / "pipeline.json").write_text(json.dumps({"tau": tau, "matcher": str(args.matcher_ckpt), "segmenter": str(args.seg_ckpt), "longiseg_model": str(args.longiseg_model), "methods": args.methods,
                                                      "anonymize": args.anonymize, "prompts": "BL inputsTrBL/*.json (true), FU inputsTrFU/*.json (propagated, backend original)"}))
        for i, pid in enumerate(pids, 1):
            for m in args.methods:
                rec = predict_unit(pl, args, pid, keys[pid], m, art)
                cprint(f"[{i}/{len(pids)}] {keys[pid]} {m}: {rec['status']}" + (f"  {', '.join(f'{k} {rec[k]:.0f}s' for k in ('t_seg', 't_track', 't_kirchhoff') if rec.get(k) is not None)}" if rec["status"] == "ok" else ""))
        del pl
    rows, lesions, counts, masks = score(art, pids, keys, args.methods, args.data_root, tau)
    metrics, deltas, summary = summarise(counts, masks, args.methods)
    failed = sorted({(r["patient"], r["method"], r["status"], r["error"]) for r in rows if r["status"] != "ok"})
    for key, m, status, err in failed:
        cprint(f"[red]missed[/red] {key} {m} ({status}): {err}")
    notes = [f"{len(failed)} (patient, method) units failed or missing and were scored as fully missed in identity: {[f[:3] for f in failed]}",
             f"site: {site}; decoders {list(P.DECODERS)} at tau={tau} for ours; Kirchhoff identity is inherited from the prompt and its class `new` is undefined (null)",
             "NUMBERS ARE NOT MEANINGFUL until the matcher is retrained on the fixed graph cache and MATCHER_FINAL is repointed (merge recall is 0 by construction with the current one)",
             "prompts: BL = true BL centroids (inputsTrBL/*.json), FU = propagated points shipped with the data (inputsTrFU/*.json, backend original), the same for both methods",
             "masks: per annotated FU lesion that carries a prompt, lesions pooled over patients, patient-level bootstrap; a failed patient has no lesion rows (its annotated lesions are unknown)",
             f"anonymised: {args.anonymize} (patient ids are salted hashes; no coordinates stored; hash -> id table in artifacts/patient_map.csv stays at the lab)"]
    if rescore:
        notes.append(f"rescored from {args.rescore} (no prediction run)")
    run.finish(summary, {"per_patient": rows, "per_lesion": lesions, "metrics": metrics, "deltas": deltas}, table_md=markdown(exp, site, summary, notes),
               definitions={**DEFINITIONS, "node_supply": NODE_SUPPLY, "decoders": list(P.DECODERS), "tau": tau, "count_keys": list(COUNT_KEYS), "metrics": list(METRICS), "mask_unit": "annotated FU lesion with a prompt"},
               notes=notes, next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
