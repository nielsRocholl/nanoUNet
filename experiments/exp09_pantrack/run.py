"""exp09 - PanTrack, transfer to another disease  (paper: Experiments > Identity > "Another disease"; Table 'nine experiments' row 9)

QUESTION   Does the identity matcher transfer to a disease and a scan protocol it never saw? PanTrack is pancreatic cancer with hepatic metastases
           (portal-venous CT, one centre); our pipeline is scored at Lstar and Lhat beside Di Veroli, Qahqaie and (when available) Kirchhoff.
WHY        Fills the PanTrack row of the paper: the same identity metrics as the Longitudinal-CT experiments on a held-out cohort from a different disease.
DATA       /nnunet_data/raw/PanTrack: 45 patients, 161 CT, 116 consecutive scan pairs (all pairs are scored, one pseudo-patient per pair), 289 measured
           lesion instances. Identity ground truth = the same label id in BL and FU; vanishing = id absent in FU (fu_point NaN), new = id only in FU; no merges.
           Prompts: BL = the true BL points (`bl_point`), FU = `fu_point_prop`, the uniGradICON-propagated BL point (the only propagation PanTrack ships: this is
           the one place uniGradICON is the prompt source). Lesion type for the matcher = organ (liver -> Liver, lymph node -> Lymph node, pancreas -> Others).
METHOD     1. Adapter (pantrack.py) validates the dataset (coordinate frame of the points against mask centroids, modulo-256 label ids, organ counts from
              organ_annotations.json against the TotalSeg-overlap derivation, every problem at once with Fix:) and synthesises a Longitudinal-CT layout.
           2. Ours through experiments/pipeline.py: setting A = Lstar (annotated nodes), B = annotated BL + segmented FU, C = Lhat (both scans segmented from
              points); matcher common.MATCHER_FINAL (EMA), segmenter common.SEG_CKPT, decoders hungarian and sinkhorn at the matcher's own dust_tau.
           3. Di Veroli (iterative overlap) and Qahqaie (unbalanced OT) at Lstar on point-propagated masks (BL mask translated by fu_point_prop - bl_point in mm,
              the exp04 rule). Hyperparameters: read from an exp04 run (--baselines-run, table chosen_params: most frequent choice over the folds, ties to the
              last fold); without it Di Veroli runs at the published d=1 p=0.10 r=7 and Qahqaie is not run.
           4. Kirchhoff et al.: not implemented here, it waits for experiments/exp07_internal_set/kirchhoff.py (column stays empty, stated in notes).
           5. Scoring = experiments.scoring (recall per class, identity ceiling, edge P/R/F1) with a patient-level bootstrap over the 45 patients (the
              pairs of one patient are pooled, never resampled separately).
OUTPUT     results.json tables: validation (one row per scan), lesions (per instance: id, organ, type, class), per_pair (method x pair counts, status),
           metrics (CIs per method), chosen_params. artifacts/: layout/ (synthesised dataset), stats/, inputs/*.pkl (baseline inputs), records/ and scores/*.npz
           (raw matcher scores per pair and setting), masks/ (matches.csv and predicted instances of settings B/C), pipeline.json.
COMMAND    python -m experiments.exp09_pantrack.run --tag paper_v1 --baselines-run /nnunet_data/experiments/exp04_baselines/<run_id>
DEPENDS ON experiments/common.py, scoring.py, pipeline.py, segment.py, exp04_baselines/{diveroli,qahqaie}.py, pantrack.py; an exp04 run for the tuned
           baseline parameters (optional); exp07 kirchhoff.py (not yet).
RUNTIME    Validation and baselines are CPU (reading 161 labels + TotalSeg masks and 322 CT reads: about 10-15 min with 4 workers). Ours: a few minutes per
           pair and setting on one A100 (scans are up to 972 slices); 116 pairs with B and C take hours: resumable with --resume, rescore with --rescore.
CAVEATS    NUMBERS ARE NOT MEANINGFUL until MATCHER_FINAL is repointed to the model retrained on the fixed graph cache (graph-builder defects, plan Sec. 2).
           Liver annotations are partial by design: an unannotated detected lesion is a false positive for every method, depressing precision and the
           ceiling. The vocabulary has no pancreas, so pancreatic lesions enter as `Others`, a type the matcher saw rarely. Lesions new in FU cannot be
           prompted (no propagated point exists for them), so under B and C they are never found. Di Veroli's dilation is in voxels on an anisotropic
           grid (0.4 mm slices). Patient 3988c7f88e is a Longitudinal-CT case and does not occur here.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.ui import cprint, nano_progress
from experiments import pipeline as P
from experiments import scoring
from experiments.common import MATCHER_FINAL, SEG_CKPT, abort_if, add_common_args, limited, missing_paths, problem, start_run
from experiments.exp04_baselines import diveroli, qahqaie
from experiments.exp09_pantrack import pantrack as T

EXP = "exp09_pantrack"
PAPER = {"section": "Experiments > Identity > Another disease", "table_row": 9, "supports": "PanTrack row: identity recall per class and edge F1 against the baselines"}
NODES = {"A": "lstar", "B": "lstar_bl_lhat_fu", "C": "lhat"}
HEAD = ("recall_unchanged", "recall_disappeared", "recall_new", "recall_macro", "edge_precision", "edge_recall", "edge_f1")
CEIL = ("ceiling_unchanged", "ceiling_disappeared", "ceiling_new")
FAILS = (OSError, ValueError, KeyError, IndexError, AssertionError, RuntimeError, MemoryError, SystemExit)


def chosen_params(run_dir: Path) -> dict[str, tuple[dict, str]]:
    """Baseline parameters of an exp04 run: per method the most frequent `chosen_params` entry over the folds, ties to the last fold."""
    rows = json.loads((run_dir / "results.json").read_text())["tables"]["chosen_params"]
    out = {}
    for m in ("diveroli", "qahqaie"):
        mine = sorted((r for r in rows if r["method"] == m), key=lambda r: r["fold"])
        assert mine, f"exp04 run {run_dir} has no chosen_params rows for {m}"
        n = Counter(json.dumps(r["params"], sort_keys=True) for r in mine)
        top = max(n.values())
        last = [r for r in mine if n[json.dumps(r["params"], sort_keys=True)] == top][-1]
        out[m] = (last["params"], f"most frequent over {len(mine)} folds ({top}x), ties to the last fold")
    return out


def prepare_inputs(args: tuple) -> str:
    root, p, cache = args
    f = Path(cache) / f"{p['pid']}.pkl"
    if not f.is_file():
        try:
            data = T.baseline_inputs(Path(root), p)
        except FAILS as e:
            data = {"pid": p["pid"], "failed": f"{type(e).__name__}: {e}"}
        with open(f, "wb") as fh:
            pickle.dump(data, fh)
    return p["pid"]


def edges_diveroli(data: dict, prm: dict) -> list[tuple[int, int]]:
    return diveroli.match(data["overlap"], prm["d"], prm["p"], prm["r"])


def edges_qahqaie(data: dict, prm: dict) -> list[tuple[int, int]]:
    bl, fu = list(range(len(data["bl"]))), list(range(len(data["fu"])))
    if not bl or not fu:
        return []
    g = lambda side, key: np.stack([n[key] for n in data[side]])
    sim = None
    if prm["w_s"] > 0:
        z = lambda p: (p - p.mean()) / (p.std() + 1e-6)
        pb, pf = [z(n["patch"]) for n in data["bl"]], [z(n["patch"]) for n in data["fu"]]
        sim = (np.array([[float(np.mean(a * b)) for b in pf] for a in pb]) + 1.0) / 2.0
    return qahqaie.match(g("bl", "x_mm"), g("fu", "x_mm"), np.array([n["vol_mm3"] for n in data["bl"]]), np.array([n["vol_mm3"] for n in data["fu"]]), sim, prm)


def score_edges(case: scoring.PairCase, data: dict, edges: list[tuple[int, int]]) -> dict:
    """Identity counts of one pair from baseline edges (indices into data['bl'] / data['fu']); a failed input scores as fully missed."""
    if "failed" in data:
        return scoring.score_pair(replace(case, found_bl=set(), found_fu=set(), pred_links=set()))
    links = {(data["bl"][i]["id"], data["fu"][j]["id"]) for i, j in edges}
    return scoring.score_pair(replace(case, found_bl={n["id"] for n in data["bl"]}, found_fu={n["id"] for n in data["fu"]}, pred_links=links))


def predict_unit(pl: P.Pipeline, layout: Path, pair: P.Pair, setting: str, art: Path) -> dict:
    """Run ours on one (pair, setting); a failure is recorded and counted as missed, never dropped."""
    rec_path = art / "records" / f"{pair.pid}_{setting}.json"
    if rec_path.is_file() and json.loads(rec_path.read_text())["status"] == "ok":
        return json.loads(rec_path.read_text())
    rec = {"pid": pair.pid, "setting": setting, "status": "ok", "error": None}
    try:
        s, scans = P.run_setting(pl, layout, pair, setting)
        P.save_scores(art / "scores" / f"{pair.pid}_{setting}.npz", s)
        P.write_links_csv(art / "masks" / f"{pair.pid}_{setting}" / "matches.csv", s, pl.tau)
        if scans is not None:
            P.write_instances(art / "masks" / f"{pair.pid}_{setting}" / "pred_fu.mha", scans["fu"]["inst"], scans["fu"]["props"])
        rec.update(t_seg=s.t_seg, t_track=s.t_track)
    except FAILS as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}")
        cprint(f"[red]failed[/red] {pair.pid} setting {setting}: {rec['error']}")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec))
    return rec


def fmt(v: list | tuple | None) -> str:
    return "n/a" if v is None or v[0] is None or v[0] != v[0] else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, rescore=True)
    ap.add_argument("--data-root", type=Path, default=T.ROOT, help="PanTrack root (images/, labels/, totalseg/, patients.json, tracking.json, organ_annotations.json)")
    ap.add_argument("--patients", nargs="+", default=None, help="explicit PanTrack patient ids (e.g. PanTrack_001) instead of all 45 (smoke runs)")
    ap.add_argument("--settings", nargs="*", choices=P.SETTINGS, default=["A", "C"], help="ours: A = Lstar, C = Lhat (both scans segmented), B = annotated BL + segmented FU; none = baselines only")
    ap.add_argument("--matcher-ckpt", type=Path, default=MATCHER_FINAL, help="matcher checkpoint (EMA weights); the owner repoints the default after the retrain")
    ap.add_argument("--tau", type=float, default=None, help="decoder cut-off; default: the matcher checkpoint's own dust_tau")
    ap.add_argument("--baselines-run", type=Path, default=None, help="exp04 RUN_DIR whose chosen_params give the Di Veroli / Qahqaie hyperparameters (default: Di Veroli published d=1 p=0.10 r=7, no Qahqaie)")
    ap.add_argument("--workers", type=int, default=3, help="processes for the scan statistics and the baseline inputs (each holds two CTs in memory)")
    args = ap.parse_args()
    rescore = args.rescore is not None
    problems = missing_paths({"PanTrack root": args.data_root, "tracking.json": args.data_root / "tracking.json", "organ_annotations.json": args.data_root / "organ_annotations.json"}, "mount /nnunet_data or pass --data-root <PanTrack root>")
    if args.settings and not rescore:
        problems += missing_paths({"matcher checkpoint": args.matcher_ckpt}, "pass --matcher-ckpt <file> or pass --settings with no values for a baselines-only run")
    if args.baselines_run is not None and not (args.baselines_run / "results.json").is_file():
        problems.append(problem(f"--baselines-run {args.baselines_run} has no results.json", "a finished exp04 run directory", "ls /nnunet_data/experiments/exp04_baselines/"))
    abort_if(problems)
    idx = T.load_index(args.data_root)
    unknown = [p for p in (args.patients or []) if p not in idx["patients"]]
    abort_if([problem(f"unknown patient(s) {unknown}", "ids of patients.json (PanTrack_001 ...)", "see `python3 -c \"import json;print(list(json.load(open('/nnunet_data/raw/PanTrack/patients.json')))[:5])\"`")] if unknown else [])
    patients = limited(sorted(args.patients or idx["patients"]), args)
    pairs = T.pair_list(idx, patients)
    if not rescore and args.settings:
        import torch
        abort_if([problem(f"--device {args.device} is not available", "a CUDA GPU (the segmenter needs one)", "run on the GPU node or pass --settings A with --device cpu for a tiny check")]
                 if args.device.startswith("cuda") and not torch.cuda.is_available() else [])
    run = start_run(EXP, ap, args, inputs={"tracking": args.data_root / "tracking.json", "organ annotations": args.data_root / "organ_annotations.json", "matcher": args.matcher_ckpt, "segmenter": SEG_CKPT}, paper=PAPER)
    art, layout = run.artifacts, run.artifacts / "layout"
    (art / "stats").mkdir(exist_ok=True)
    scans = sorted({s for p in pairs for s in (p["bl"], p["fu"])})
    todo = [s for s in scans if not (art / "stats" / f"{s}.json").is_file()]
    cprint(f"status: scan statistics | patients: {len(patients)} | pairs: {len(pairs)} | scans to read: {len(todo)}")
    with ProcessPoolExecutor(args.workers) as ex, nano_progress(max(len(todo), 1), "scans") as adv:
        for s, st in zip(todo, ex.map(T.scan_stats, [args.data_root] * len(todo), todo)):
            (art / "stats" / f"{s}.json").write_text(json.dumps(st))
            adv(1)
    stats = {s: json.loads((art / "stats" / f"{s}.json").read_text()) for s in scans}
    # JSON turns the int instance keys into strings
    stats = {s: {**v, "instances": {int(k): i for k, i in v["instances"].items()}} for s, v in stats.items()}
    val_rows, val_problems = T.validate(idx, pairs, stats)
    abort_if(val_problems)
    differ = [r["scan"] for r in val_rows if not r["counts_agree"]]
    cprint(f"status: validation ok | scans: {len(val_rows)} | organ counts differ between annotation and TotalSeg in {len(differ)} scans: {differ}")
    by_pid = {p["pid"]: p for p in pairs}
    layout_pairs = T.build_layout(args.data_root, idx, pairs, stats, layout) if not rescore else {}
    cases = scoring.load_pairs(layout, list(by_pid))
    patient_of = {p["pid"]: p["patient"] for p in pairs}
    lesions = [{"pid": p["pid"], "scan_pair": f"{p['bl']} -> {p['fu']}", "lesion_id": r["lesion_id"], "organ": r["organ"], "type": r["lesion_type"], "class": r["topology_class"]}
               for p in pairs for r in T.pair_rows(idx, p, stats)]
    per_pair: list[dict] = []
    methods: dict[str, dict[str, dict]] = {}

    def add(name: str, pid: str, counts: dict, **extra: object) -> None:
        methods.setdefault(name, {})[pid] = counts
        per_pair.append({"method": name, "pid": pid, "patient": patient_of[pid], "n_bl": len(cases[pid].bl_ids), "n_fu": len(cases[pid].fu_ids), **extra, **counts})

    used_params = []
    if not rescore:
        (art / "inputs").mkdir(exist_ok=True)
        cprint(f"status: baseline inputs | pairs: {len(pairs)} | workers: {args.workers}")
        with ProcessPoolExecutor(args.workers) as ex, nano_progress(len(pairs), "inputs") as adv:
            for _ in ex.map(prepare_inputs, [(str(args.data_root), p, str(art / "inputs")) for p in pairs]):
                adv(1)
    data = {}
    for p in pairs:
        with open(art / "inputs" / f"{p['pid']}.pkl", "rb") as fh:
            data[p["pid"]] = pickle.load(fh)
    params = chosen_params(args.baselines_run) if args.baselines_run is not None else {"diveroli": (diveroli.PUBLISHED, "published d=1 p=0.10 r=7")}
    rows_by = {"diveroli_lstar": ("diveroli", edges_diveroli), "qahqaie_lstar": ("qahqaie", edges_qahqaie)}
    for name, (m, fn) in rows_by.items():
        if m not in params:
            continue
        prm, why = params[m]
        used_params.append({"method": name, "params": prm, "source": why, "baselines_run": str(args.baselines_run) if args.baselines_run else None})
        for p in pairs:
            d = data[p["pid"]]
            add(name, p["pid"], score_edges(cases[p["pid"]], d, [] if "failed" in d else fn(d, prm)), status="failed" if "failed" in d else "ok", error=d.get("failed"))
    if not rescore and args.settings:
        pl = P.load_pipeline(args.device, matcher_ckpt=args.matcher_ckpt, segmenter=any(s in ("B", "C") for s in args.settings))
        (art / "pipeline.json").write_text(json.dumps({"tau": pl.tau, "matcher": str(args.matcher_ckpt), "segmenter": str(SEG_CKPT), "prompts": "BL bl_point (true), FU fu_point_prop (uniGradICON-propagated)"}))
        for i, p in enumerate(pairs, 1):
            for setting in args.settings:
                rec = predict_unit(pl, layout, layout_pairs[p["pid"]], setting, art)
                cprint(f"[{i}/{len(pairs)}] {p['pid']} {setting}: {rec['status']}" + (f"  seg {rec['t_seg']:.0f}s track {rec['t_track']:.1f}s" if rec["status"] == "ok" else ""))
        del pl
    tau = args.tau
    if (art / "pipeline.json").is_file():
        tau = tau if tau is not None else json.loads((art / "pipeline.json").read_text())["tau"]
    done = [s for s in P.SETTINGS if any((art / "records").glob(f"*_{s}.json"))]
    for setting in done:
        for p in pairs:
            rp = art / "records" / f"{p['pid']}_{setting}.json"
            rec = json.loads(rp.read_text()) if rp.is_file() else {"status": "missing", "error": "no record: the run never reached this pair"}
            s = P.load_scores(art / "scores" / f"{p['pid']}_{setting}.npz") if rec["status"] == "ok" else None
            for dec in P.DECODERS:
                add(f"ours_{NODES[setting]}_{dec}", p["pid"], P.score_scores(cases[p["pid"]], s, dec, tau, exclude_unclear=True), status=rec["status"], error=rec["error"])
    summary, rows_md = {}, []
    for name, by_pair in methods.items():
        by_patient: dict[str, dict] = defaultdict(lambda: dict.fromkeys(scoring.COUNT_KEYS, 0))
        for pid, c in by_pair.items():
            for k, v in c.items():
                by_patient[patient_of[pid]][k] += v
        ci = scoring.bootstrap(dict(by_patient))
        summary[name] = {m: list(ci[m]) for m in HEAD + CEIL}
        rows_md.append(f"| {name} | " + " | ".join(fmt(summary[name][m]) for m in HEAD + CEIL) + " |")
    metrics = [{"method": n, "metric": m, "point": v[0], "lo": v[1], "hi": v[2], "n_patients": len(patients)} for n, s in summary.items() for m, v in s.items()]
    failed = sorted({(r["method"], r["pid"], r["status"], r["error"]) for r in per_pair if r["status"] != "ok"})
    for m, pid, st, err in failed:
        cprint(f"[red]missed[/red] {pid} {m} ({st}): {err}")
    notes = [f"{len(patients)} patients, {len(pairs)} consecutive pairs, {sum(r['n_instances'] for r in val_rows)} lesion-instance rows over scans (each scan counted once per appearance in a pair)",
             "FU prompts (and the baselines' point propagation) = fu_point_prop, the uniGradICON-propagated BL point: PanTrack ships no other propagation, so this is the one place uniGradICON is the prompt source",
             "frame verified by validation: bl_point / fu_point are mask centroids in voxel x,y,z of the scan's own grid; label ids wrap modulo 256 (annotation 307 = label 51)",
             "lesion type for the matcher = organ: liver -> Liver, lymph node -> Lymph node, pancreas -> Others (the LESION_TYPES vocabulary has no pancreas)",
             f"organ counts (organ_annotations.json vs TotalSeg-overlap derivation) differ in {len(differ)} scans: {differ}",
             "annotation caveat: liver annotations are partial by design; an unannotated detected lesion counts as a false positive for every method (lower precision and ceiling)",
             "Kirchhoff column: not implemented, waits for exp07_internal_set/kirchhoff.py",
             "NUMBERS ARE NOT MEANINGFUL until MATCHER_FINAL is repointed to the retrained model (graph-builder fix)",
             f"baseline hyperparameters: {[(u['method'], u['params'], u['source']) for u in used_params]}",
             f"{len(failed)} (method, pair) units failed or are missing and were scored as fully missed" + (f" (decoders {list(P.DECODERS)} at tau={tau})" if done else "")]
    md = (f"# {EXP}\n\nNUMBERS ARE NOT MEANINGFUL until MATCHER_FINAL is repointed to the retrained model.\n\n| method | " + " | ".join(HEAD + CEIL) + " |\n|---|" + "---|" * len(HEAD + CEIL) + "\n"
          + "\n".join(rows_md) + "\n\n## Notes\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    run.finish(summary, {"validation": val_rows, "lesions": lesions, "per_pair": per_pair, "metrics": metrics, "chosen_params": used_params}, table_md=md, notes=notes,
               definitions={**scoring.DEFINITIONS, "node_supply": {"ours": NODES, "baselines": "lstar"}, "bootstrap_unit": "patient (pairs of one patient pooled)", "count_keys": list(scoring.COUNT_KEYS)},
               next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
