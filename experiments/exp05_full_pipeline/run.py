"""exp05 - Full pipeline, the segmenter supplies the lesions  (paper: Experiments > Identity > "The full pipeline"; Table 'nine experiments' row 5)

QUESTION   What does it cost when the segmenter, not the annotation, supplies the lesions the matcher links? Identity recall per class and edge F1
           for the complete deployment pipeline (prompted segmentation, graph, matcher, decoder), beside the ceiling set by lesions that never got a node.
WHY        Fills the full-pipeline row of the paper: the same metrics as the matcher-alone row (exp03) with annotated nodes replaced by predicted ones;
           the drop from setting A to B to C is the price of each segmentation step, and (ceiling - recall) separates the matcher's share of the error
           from the node supply's.
DATA       The Longitudinal-CT held-out 60 patients (test_patients.csv), one scan pair per patient = the dominant follow-up region (most meta rows), files
           `<pid>_<region>` in inputsTrBL/inputsTrFU/targetsTrBL/targetsTrFU. Prompts (no flag, recorded in run.json): BL = the true BL centroids
           inputsTrBL/*.json, FU = the propagated points shipped with the dataset inputsTrFU/*.json (registration backend `original`). Headline
           excludes lesions flagged `linking_unclear`; the with-unclear row is stored beside it. Patient 3988c7f88e is never dropped: if the pipeline
           fails on a patient the record keeps `status`, the error, and the patient counts as fully missed (printed at the end and stored in notes).
METHOD     Matcher = common.MATCHER_FINAL (trained on the 240-patient pool, no selection, EMA weights), segmenter = common.SEG_CKPT (EMA), decoders
           hungarian (strict 1-to-1) and sinkhorn (every cell with row mass >= tau, merges expressible) at tau = the matcher checkpoint's own dust_tau.
           1. Setting A (control, Lstar both sides): nodes from the annotation through lesionglue's graph builder, so identity is isolated.
           2. Setting B (protocol): BL = annotated instance masks with ids, FU = segmented from the propagated points. FU nodes are tied to annotated FU
              lesions by IoU > 0.1 (nanounet's hit rule); unmatched predicted nodes are extra nodes (negative ids: they can steal links, never count as lesions).
           3. Setting C (points only): both scans segmented, BL from the true BL points, FU from the propagated points; both sides tied by IoU > 0.1.
           4. Raw pair logits and dustbin scores of every (patient, setting) are stored, decoding and scoring are separate steps (`--rescore` reruns only those).
           5. Score with scoring.score_pair: recall per class (unchanged, disappeared, new, merged), identity ceiling per class, matcher error = ceiling -
              recall, edge P/R/F1; patient-level bootstrap CIs (B=10000) and paired deltas A-B and B-C on the same patients.
OUTPUT     results.json tables: per_patient (per patient x setting x decoder x variant: status, error, node counts, timings, counts per class ok/tot/ceil,
           tp/fp/fn, node -> annotated ids, predicted links in annotated-id space), metrics (every CI), deltas. artifacts/: pipeline.json (tau, matcher),
           records/<pid>_<setting>.json, scores/<pid>_<setting>.npz (raw scores, A also _A_unclear), masks/<pid>_<setting>/{matches.csv, pred_fu.mha, pred_bl.mha}.
COMMAND    python -m experiments.exp05_full_pipeline.run --tag paper_v1
DEPENDS ON experiments/common.py, scoring.py (score_pair, load_pairs, bootstrap, paired_delta), segment.py (load_segmenter), pipeline.py (the pipeline); the
           final matcher checkpoint (owner repoints MATCHER_FINAL after the retrain) and the graph-builder fix for setting A.
RUNTIME    About 1 min per patient and setting on one A100 (segmentation dominates): B+C about 2-3 h for 60 patients, A minutes. Resumable (--resume skips
           finished patients and retries failed ones); scoring alone (--rescore) takes about a minute.
CAVEATS    NUMBERS ARE NOT MEANINGFUL until the matcher is retrained on the fixed graph cache (graph-builder defects in the experiments plan, Sec. 2): the
           current MATCHER_FINAL never saw a merge-target node, so merge recall is 0 by construction. Setting A must be re-verified after that fix (it uses
           the cache builder `build_hetero_data`, merge-target nodes and the opt-in prop-fill follow it). Lesion types and propagated BL positions come from the
           meta CSV as in segtrack; `load_propagated` falls back to cog_fu for a BL lesion without cog_propagated (B and C only). An FU component that no
           prompt claims gets a fresh id and is an extra node; a BL lesion without a propagated point gets no node (ceiling < 1). The headline row treats
           flagged-unclear lesions as absent in the scoring, the matcher still sees them as nodes in B and C.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ui import cprint
from experiments import pipeline as P
from experiments.common import HOLDOUT_CSV, LONGI_ROOT, MATCHER_FINAL, SEG_CKPT, abort_if, add_common_args, limited, missing_paths, problem, start_run
from experiments.scoring import COUNT_KEYS, DEFINITIONS, METRICS, bootstrap, load_pairs, paired_delta, pooled

EXP = "exp05_full_pipeline"
PAPER = {"section": "Experiments > Identity > The full pipeline", "table_row": 5, "supports": "full-pipeline row: recall per class and edge F1 under Lstar / Lstar+Lhat / Lhat nodes, with ceilings"}
NODE_SUPPLY = {"A": "Lstar both sides (annotation)", "B": "Lstar BL, Lhat FU (segmented from propagated points)", "C": "Lhat both sides (segmented from points)"}
HEAD = ("recall_unchanged", "recall_disappeared", "recall_new", "recall_merged", "recall_macro", "edge_f1")
CEIL = ("ceiling_unchanged", "ceiling_disappeared", "ceiling_new", "ceiling_merged")


def predict_unit(pl: P.Pipeline, root: Path, pid: str, setting: str, art: Path) -> dict:
    """Run one (patient, setting), write scores/records/masks into artifacts; a failure is recorded, never dropped (plan Sec. 4, rule 3)."""
    rec_path = art / "records" / f"{pid}_{setting}.json"
    if rec_path.is_file() and json.loads(rec_path.read_text())["status"] == "ok":
        return json.loads(rec_path.read_text())  # resume: finished units are skipped, failed ones retried
    rec = {"pid": pid, "setting": setting, "status": "ok", "error": None}
    try:
        pair = P.longitudinal_pair(root, pid)
        rec["stem"] = pair.stem
        absent = [str(p) for p in (pair.bl_img, pair.fu_img, pair.fu_clicks, pair.bl_mask, pair.fu_mask, pair.meta) + ((pair.bl_clicks,) if setting == "C" else ()) if not p.is_file()]
        if absent:
            raise FileNotFoundError(f"missing input files {absent}. Expected the Longitudinal-CT layout. Fix: restore the files or drop the patient from --patients-csv")
        s, scans = P.run_setting(pl, root, pair, setting)
        P.save_scores(art / "scores" / f"{pid}_{setting}.npz", s)
        P.write_links_csv(art / "masks" / f"{pid}_{setting}" / "matches.csv", s, pl.tau)
        if scans is not None:
            P.write_instances(art / "masks" / f"{pid}_{setting}" / "pred_fu.mha", scans["fu"]["inst"], scans["fu"]["props"])
            if setting == "C":
                P.write_instances(art / "masks" / f"{pid}_{setting}" / "pred_bl.mha", scans["bl"]["inst"], scans["bl"]["props"])
        if setting == "A":  # the with-unclear row needs a graph that keeps the flagged lesions
            P.save_scores(art / "scores" / f"{pid}_A_unclear.npz", P.run_setting(pl, root, pair, "A", keep_unclear=True)[0])
        rec.update(t_seg=s.t_seg, t_track=s.t_track)
    except (OSError, ValueError, KeyError, IndexError, AssertionError, RuntimeError, SystemExit) as e:  # a patient the pipeline cannot process stays as status=failed, counted as missed
        rec.update(status="failed", error=f"{type(e).__name__}: {e}")
        cprint(f"[red]failed[/red] {pid} setting {setting}: {rec['error']}")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec))
    return rec


def read_unit(art: Path, pid: str, setting: str) -> tuple[dict, P.Scores | None, P.Scores | None]:
    """(record, headline scores, with-unclear scores) of one unit; scores are None when the unit failed or never ran."""
    rp = art / "records" / f"{pid}_{setting}.json"
    rec = json.loads(rp.read_text()) if rp.is_file() else {"pid": pid, "setting": setting, "status": "missing", "error": "no record: the run never reached this patient"}
    if rec["status"] != "ok":
        return rec, None, None
    s = P.load_scores(art / "scores" / f"{pid}_{setting}.npz")
    unc = art / "scores" / f"{pid}_A_unclear.npz"
    return rec, s, (P.load_scores(unc) if setting == "A" and unc.is_file() else s)


def score(art: Path, pids: list[str], settings: list[str], root: Path, tau: float) -> tuple[list[dict], dict]:
    """Pure CPU: decode the stored raw scores with both decoders, score every (patient, setting, decoder) headline and with-unclear."""
    cases, rows, counts = load_pairs(root, pids, include_unclear=True), [], {}
    for setting in settings:
        for pid in pids:
            rec, s_head, s_unc = read_unit(art, pid, setting)
            for dec in P.DECODERS:
                for variant, s, excl in (("headline", s_head, True), ("with_unclear", s_unc, False)):
                    c = P.score_scores(cases[pid], s, dec, tau, exclude_unclear=excl)
                    counts.setdefault((setting, dec, variant), {})[pid] = c
                    row = {"pid": pid, "setting": setting, "decoder": dec, "variant": variant, "status": rec["status"], "error": rec["error"], "stem": rec.get("stem"),
                           "t_seg": rec.get("t_seg"), "t_track": rec.get("t_track"), "n_bl_nodes": None if s is None else len(s.bl_ids), "n_fu_nodes": None if s is None else len(s.fu_ids),
                           "n_bl_extra": None if s is None else int((s.bl_ann == 0).sum()), "n_fu_extra": None if s is None else int((s.fu_ann == 0).sum()), **c}
                    if variant == "headline" and s is not None:
                        row.update(bl_nodes=[[int(i), int(a)] for i, a in zip(s.bl_ids, s.bl_ann)], fu_nodes=[[int(i), int(a)] for i, a in zip(s.fu_ids, s.fu_ann)],
                                   pred_links=sorted([int(b), int(f)] for b, f in P.decode_links(s, dec, tau)))
                    rows.append(row)
    return rows, counts


def summarise(counts: dict, settings: list[str]) -> tuple[list[dict], list[dict], dict]:
    """Bootstrap CIs of every metric per (setting, decoder, variant), paired deltas on the same patients, and the compact summary dict."""
    metrics, deltas, summary = [], [], {}
    for (setting, dec, variant), by_pid in counts.items():
        ci, n = bootstrap(by_pid), len(by_pid)
        metrics += [{"setting": setting, "decoder": dec, "variant": variant, "metric": m, "point": p, "lo": lo, "hi": hi, "n_patients": n} for m, (p, lo, hi) in ci.items()]
        if variant == "headline":
            summary.setdefault(setting, {})[dec] = {m: list(ci[m]) for m in HEAD + CEIL}
    for dec in P.DECODERS:
        for a, b in zip(settings, settings[1:]):
            d = paired_delta(counts[(a, dec, "headline")], counts[(b, dec, "headline")])
            deltas += [{"a": a, "b": b, "decoder": dec, "metric": m, "delta": p, "lo": lo, "hi": hi} for m, (p, lo, hi) in d.items()]
            summary.setdefault("delta", {})[f"{a}_minus_{b}_{dec}"] = {m: list(d[m]) for m in ("recall_macro", "edge_f1")}
    return metrics, deltas, summary


def fmt(v: list[float] | tuple) -> str:
    return "n/a" if v is None or v[0] is None or v[0] != v[0] else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def markdown(metrics: list[dict], settings: list[str], notes: list[str]) -> str:
    look = {(r["setting"], r["decoder"], r["variant"], r["metric"]): [r["point"], r["lo"], r["hi"]] for r in metrics}
    out = [f"# {EXP}", "", "NUMBERS ARE NOT MEANINGFUL until the matcher is retrained on the fixed graph cache (see CAVEATS in run.py).", ""]
    for dec in P.DECODERS:
        for variant in ("headline", "with_unclear"):
            out += [f"## {dec}, {variant.replace('_', ' ')} (recall per class, edge F1, identity ceiling; patient-bootstrap 95 % CI)", "",
                    "| setting | unchanged | disappeared | new | merged | macro | edge F1 | ceil unch. | ceil disapp. | ceil new | ceil merged |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            out += [f"| {s} {NODE_SUPPLY[s]} | " + " | ".join(fmt(look.get((s, dec, variant, m))) for m in HEAD + CEIL) + " |" for s in settings]
            out.append("")
    return "\n".join(out + ["## Notes", ""] + [f"- {n}" for n in notes])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, rescore=True)
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="Longitudinal-CT layout root (inputsTrBL|FU, targetsTrBL|FU, meta)")
    ap.add_argument("--patients-csv", type=Path, default=HOLDOUT_CSV, help="CSV with a `patient` column: the held-out 60 by default")
    ap.add_argument("--patients", nargs="+", default=None, help="explicit patient ids instead of --patients-csv (debugging, e.g. the known-bad 3988c7f88e)")
    ap.add_argument("--settings", nargs="+", choices=P.SETTINGS, default=list(P.SETTINGS), help="node supplies to run: A annotation, B annotated BL + segmented FU, C segmented both")
    ap.add_argument("--matcher-ckpt", type=Path, default=MATCHER_FINAL, help="matcher checkpoint (EMA weights are used); the owner repoints the default after the retrain")
    ap.add_argument("--tau", type=float, default=None, help="decoder cut-off; default: the matcher checkpoint's own dust_tau (only change it to rescore a sensitivity row)")
    args = ap.parse_args()
    rescore = args.rescore is not None
    problems = missing_paths({"data root": args.data_root, "matcher checkpoint": args.matcher_ckpt}, "mount /nnunet_data or pass the path") if not rescore else missing_paths({"data root": args.data_root}, "mount /nnunet_data")
    if args.patients is None and not args.patients_csv.is_file():
        problems.append(problem(f"patient list not found: {args.patients_csv}", "a CSV with a `patient` column", "pass --patients-csv <file> or --patients <ids...>"))
    abort_if(problems)
    if rescore:
        pids = sorted({p.name.rsplit("_", 1)[0] for p in (args.rescore / "artifacts" / "records").glob("*_?.json")})
        pids = pids if args.patients is None else sorted(args.patients)
    else:
        pids = args.patients if args.patients is not None else sorted(pd.read_csv(args.patients_csv)["patient"].astype(str))
    pids = limited(pids, args)
    abort_if(missing_paths({f"meta CSV of {p}": args.data_root / "meta" / f"{p}.csv" for p in pids}, "restore the meta CSVs or drop the patient") + ([] if pids else [problem("no patients selected", "at least one patient", "check --patients-csv / --limit-patients")]))
    if not rescore:
        import torch
        abort_if([problem(f"--device {args.device} is not available", "a CUDA GPU (the segmenter needs one)", "run on the GPU node, or --device cpu for a tiny smoke (slow)")] if args.device.startswith("cuda") and not torch.cuda.is_available() else [])
    run = start_run(EXP, ap, args, inputs={"patients": args.patients_csv, "matcher": args.matcher_ckpt, "segmenter": SEG_CKPT} if not rescore else {"patients": args.patients_csv},
                    paper=PAPER)
    art = run.artifacts
    if rescore:
        tau = args.tau if args.tau is not None else json.loads((art / "pipeline.json").read_text())["tau"]
    else:
        pl = P.load_pipeline(args.device, matcher_ckpt=args.matcher_ckpt, segmenter=any(s in ("B", "C") for s in args.settings))
        tau = args.tau if args.tau is not None else pl.tau
        (art / "pipeline.json").write_text(json.dumps({"tau": pl.tau, "matcher": str(args.matcher_ckpt), "segmenter": str(SEG_CKPT), "prompts": "BL inputsTrBL/*.json (true), FU inputsTrFU/*.json (propagated, backend original)"}))
        for i, pid in enumerate(pids, 1):
            for setting in args.settings:
                rec = predict_unit(pl, args.data_root, pid, setting, art)
                cprint(f"[{i}/{len(pids)}] {pid} {setting}: {rec['status']}" + (f"  seg {rec['t_seg']:.0f}s track {rec['t_track']:.1f}s" if rec["status"] == "ok" else ""))
        del pl
    settings = list(args.settings) if not rescore else [s for s in P.SETTINGS if any((art / "records").glob(f"*_{s}.json"))]
    rows, counts = score(art, pids, settings, args.data_root, tau)
    metrics, deltas, summary = summarise(counts, settings)
    failed = sorted({(r["pid"], r["setting"], r["status"], r["error"]) for r in rows if r["status"] != "ok"})
    for pid, setting, status, err in failed:
        cprint(f"[red]missed[/red] {pid} setting {setting} ({status}): {err}")
    notes = [f"{len(failed)} (patient, setting) units failed or missing and were scored as fully missed: {[f[:3] for f in failed]}",
             f"decoders {list(P.DECODERS)} at tau={tau} (matcher dust_tau unless --tau); matcher {args.matcher_ckpt if not rescore else json.loads((art / 'pipeline.json').read_text())['matcher']}",
             "NUMBERS ARE NOT MEANINGFUL until the matcher is retrained on the fixed graph cache: the current MATCHER_FINAL never saw merge-target nodes (merge recall 0 by construction); setting A must be re-verified after the graph-builder fix",
             "prompts: BL = true BL centroids (inputsTrBL/*.json), FU = propagated points shipped with the dataset (inputsTrFU/*.json, backend original)",
             "headline excludes linking_unclear lesions from the scoring; with_unclear scores everything (setting A uses a second graph that keeps the flagged lesions)"]
    if rescore:
        notes.append(f"rescored from {args.rescore} (no prediction run)")
    run.finish(summary, {"per_patient": rows, "metrics": metrics, "deltas": deltas}, table_md=markdown(metrics, settings, notes),
               definitions={**DEFINITIONS, "node_supply": NODE_SUPPLY, "decoders": list(P.DECODERS), "tau": tau, "count_keys": list(COUNT_KEYS), "metrics": list(METRICS)}, notes=notes,
               next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
