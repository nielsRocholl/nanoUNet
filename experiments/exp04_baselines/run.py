"""exp04 - Baselines  (paper: Sec. "Identity" > "Baselines"; Table 'nine experiments' row 4)

QUESTION   Does an identity-fitted correspondence beat hand-set costs? Two published matchers, reimplemented (neither released code nor
           hyperparameters), are scored exactly like our matcher on the same patients, folds and node supply.
WHY        Fills the baseline rows of the identity table: Di Veroli et al. (iterative greedy overlap, published d/p/r and tuned) and
           Qahqaie et al. (unbalanced entropic optimal transport, registration-trust term omitted because no deformation field exists).
DATA       All 300 Longitudinal-CT patients, one scan pair per patient (dominant img_id_fu), node supply Lstar (annotated lesions), the same
           5 patient folds as exp03 (`experiments/exp03_matcher_alone/folds.py`). Headline excludes linking_unclear lesions.
METHOD     1. Per patient (cached in artifacts/inputs/): lesion centroids, volumes, point sets. BL lesions are moved into FU space by the
              shipped point propagation (`cog_propagated`, backend original); lesions without one get the uniGradICON point only where
              its sanity check passed (`--prop-fill unigradicon`, the same opt-in as the graph builder), otherwise they own no node.
           2. Every hyperparameter grid point is scored on every patient (Di Veroli: d in {1,2}, p in {.05,.10,.20,.30}, r in
              {3,5,7,10,15}; Qahqaie: see qahqaie.GRID). Nested tuning: to score fold k, pick the grid point with the best macro average
              of the four class recalls on the OTHER four folds' patients, apply it to fold k; chosen values are stored per fold.
           3. Di Veroli is also reported at its published values d=1, p=0.10 with r=5, 7 (headline) and 10.
           4. Scoring = `experiments.scoring` (recall per class, identity ceiling, edge P/R/F1, patient bootstrap); `--ours-run` adds paired
              deltas against the per-patient counts of an exp03 run (table `per_patient`, columns pid, decoder, counts).
OUTPUT     results.json tables: per_patient (method x patient counts), chosen_params (per fold), tuning_scores (grid x fold), missing
           (lesions without a node and why); summary: per method all four class recalls, ceilings, edge F1 with CIs. artifacts/inputs/*.pkl
           (resumable), edges_<method>.json (the chosen (BL id, FU id) edges per patient).
COMMAND    python -m experiments.exp04_baselines.run --tag paper_v1 --ours-run /nnunet_data/experiments/exp03_matcher_alone/<run_id>
DEPENDS ON experiments.common, experiments.scoring, experiments/exp03_matcher_alone/folds.py (fold assignment), side modules diveroli.py,
           qahqaie.py, inputs.py in this folder; an exp03 run for the paired deltas (optional).
RUNTIME    CPU only. Preparing 300 patients (mask/CT reads + overlap tables) about 15-30 min with 8 workers; tuning and scoring a few minutes.
CAVEATS    Di Veroli's pseudo-code appendix is unavailable: dilation is cumulative (see diveroli.py). Qahqaie's hyperparameters, pruning and
           completion terms are our reading (see qahqaie.py); the paper's registration-trust term is omitted (w_J = 0). The secondary
           Di Veroli registration input (uniGradICON-warped BL masks) is NOT implemented in this version. Patients or lesions without
           a node are kept and counted as missed (see `missing`). Appearance uses 9x9x9 voxel patches at the centroids, no resampling.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.ui import cprint, nano_progress
from experiments import scoring
from experiments.common import LONGI_ROOT, abort_if, add_common_args, limited, missing_paths, problem, start_run
from experiments.exp03_matcher_alone.folds import all_patients, assign_folds
from experiments.exp04_baselines import diveroli, inputs, qahqaie

EXP = "exp04_baselines"
PAPER = {"section": "Identity > Baselines", "table_row": 4, "supports": "baseline rows of the identity table"}
PUBLISHED_R = (5, 7, 10)  # r = 7 is the headline published row
CLASSES = scoring.CLASSES


def _prepare(args: tuple) -> str:
    root, pid, case, prop_fill, cache, write = args
    inputs.get(Path(root), pid, case, prop_fill, Path(cache), write=write)
    return pid


def edges_diveroli(data: dict, prm: dict) -> list[tuple[int, int]]:
    return diveroli.match(data["overlap"], prm["d"], prm["p"], prm["r"])


def edges_qahqaie(data: dict, prm: dict) -> list[tuple[int, int]]:
    bl, fu = [i for i, b in enumerate(data["bl"]) if b["node"]], [j for j, f in enumerate(data["fu"]) if f["node"]]
    if not bl or not fu:
        return []
    g = lambda side, idx, key: np.stack([data[side][i][key] for i in idx])
    sim = None
    if prm["w_s"] > 0:
        z = lambda p: (p - p.mean()) / (p.std() + 1e-6)
        pb, pf = [z(data["bl"][i]["patch"]) for i in bl], [z(data["fu"][j]["patch"]) for j in fu]
        sim = (np.array([[float(np.mean(a * b)) for b in pf] for a in pb]) + 1.0) / 2.0
    e = qahqaie.match(g("bl", bl, "x_mm"), g("fu", fu, "x_mm"), np.array([data["bl"][i]["vol_mm3"] for i in bl]),
                      np.array([data["fu"][j]["vol_mm3"] for j in fu]), sim, prm)
    return [(bl[i], fu[j]) for i, j in e]


def score(case: scoring.PairCase, data: dict, edges: list[tuple[int, int]]) -> dict:
    bl_ids, fu_ids = [b["id"] for b in data["bl"]], [f["id"] for f in data["fu"]]
    links = {(bl_ids[i], fu_ids[j]) for i, j in edges}
    found_bl, found_fu = {b["id"] for b in data["bl"] if b["node"]}, {f["id"] for f in data["fu"] if f["node"]}
    return scoring.score_pair(replace(case, found_bl=found_bl, found_fu=found_fu, pred_links=links))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, gpu=False, rescore=True)
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="Longitudinal-CT layout root (meta/, targetsTrBL|FU/, inputsTrBL|FU/)")
    ap.add_argument("--prop-fill", choices=("none", "unigradicon"), default="unigradicon", help="propagated point for BL lesions that lack one: none, or the uniGradICON bl_click where its sanity check passed")
    ap.add_argument("--workers", type=int, default=8, help="processes for the per-patient input preparation")
    ap.add_argument("--ours-run", type=Path, default=None, help="exp03 RUN_DIR whose per_patient counts give the paired deltas (default: no deltas)")
    ap.add_argument("--ours-decoder", default="hungarian", help="decoder value of the exp03 per_patient rows to compare against")
    args = ap.parse_args()
    problems = missing_paths({"data root": args.data_root, "meta dir": args.data_root / "meta"}, "mount /nnunet_data or pass --data-root <Longitudinal-CT root>")
    if args.ours_run is not None and not (args.ours_run / "results.json").is_file():
        problems.append(problem(f"--ours-run {args.ours_run} has no results.json", "a finished exp03 run directory", "ls /nnunet_data/experiments/exp03_matcher_alone/"))
    pids = limited(all_patients(), args)
    if args.rescore is not None:
        absent = [p for p in pids if not (args.rescore / "artifacts" / "inputs" / f"{p}.pkl").is_file()]
        if absent:
            problems.append(problem(f"--rescore {args.rescore} lacks inputs for {len(absent)} of {len(pids)} patients (e.g. {absent[0]})", "the artifacts/inputs/ written by the original run",
                                    "rerun without --rescore, or match --limit-patients to the original run"))
    abort_if(problems)
    fold_of = assign_folds(all_patients())
    cases = scoring.load_pairs(args.data_root, pids)  # headline pairs: linking_unclear lesions are not nodes, exactly like the graph cache
    run = start_run(EXP, ap, args, paper=PAPER, inputs={"meta dir": args.data_root / "meta"})
    cache = run.artifacts / "inputs"
    todo = [p for p in pids if not (cache / f"{p}.pkl").is_file()]
    cprint(f"status: preparing inputs | patients: {len(pids)} | to build: {len(todo)} | workers: {args.workers}")
    if todo:
        with ProcessPoolExecutor(args.workers) as ex, nano_progress(len(todo), "inputs") as adv:
            for _ in ex.map(_prepare, [(str(args.data_root), p, cases[p], args.prop_fill, str(cache), True) for p in todo]):
                adv(1)
    data = {p: inputs.get(args.data_root, p, cases[p], args.prop_fill, cache, write=False) for p in pids}
    missing = [{"pid": p, "side": s, "lesion": lid, "reason": why} for p, d in data.items() for s, lid, why in d["missing"]]
    cprint(f"status: scoring | lesions without a node: {len(missing)} (kept, counted as missed)")

    methods = {"diveroli": [dict(d=d, p=p, r=r) for d in diveroli.GRID["d"] for p in diveroli.GRID["p"] for r in diveroli.GRID["r"]],
               "qahqaie": qahqaie.grid_points()}
    grid_counts: dict[str, list[dict]] = {}
    with nano_progress(sum(len(g) for g in methods.values()), "grid") as adv:
        for m, grid in methods.items():
            fn = edges_diveroli if m == "diveroli" else edges_qahqaie
            grid_counts[m] = [{p: score(cases[p], data[p], fn(data[p], prm)) for p in pids} for prm in grid]
            adv(len(grid))
    rows, chosen, tuning, edges_out = [], [], [], {}
    macro = lambda counts: scoring.pooled(counts)["recall_macro"]
    for m, grid in methods.items():  # nested tuning: fold k is scored with the grid point that is best on the other folds
        per_patient, chosen_idx = {}, {}
        for k in sorted(set(fold_of[p] for p in pids)):
            tune = [p for p in pids if fold_of[p] != k]
            if not tune:
                raise SystemExit(f"fold {k} has no tuning patients among the {len(pids)} selected\nExpected patients from at least two folds.\nFix: raise --limit-patients (smoke runs need >= 6)")
            score_tune = [macro([gc[p] for p in tune]) for gc in grid_counts[m]]
            best = int(np.nanargmax(score_tune))
            chosen.append({"method": m, "fold": k, "params": grid[best], "tuning_macro_recall": score_tune[best], "n_tune_patients": len(tune)})
            tuning += [{"method": m, "fold": k, "grid_index": i, "params": g, "tuning_macro_recall": s} for i, (g, s) in enumerate(zip(grid, score_tune))]
            for p in pids:
                if fold_of[p] == k:
                    per_patient[p], chosen_idx[p] = grid_counts[m][best][p], best
        variants = {f"{m}_tuned": (per_patient, chosen_idx)}
        if m == "diveroli":
            for r in PUBLISHED_R:
                idx = grid.index(dict(diveroli.PUBLISHED, r=r))
                variants[f"diveroli_published_r{r}"] = (grid_counts[m][idx], dict.fromkeys(pids, idx))
        fn = edges_diveroli if m == "diveroli" else edges_qahqaie
        for name, (counts, idx_of) in variants.items():
            rows += [{"method": name, "pid": p, "fold": fold_of[p], **counts[p]} for p in pids]
            edges_out[name] = {p: [(data[p]["bl"][i]["id"], data[p]["fu"][j]["id"]) for i, j in fn(data[p], grid[idx_of[p]])] for p in pids}
    for name, per_pid in edges_out.items():
        (run.dir / f"edges_{name}.json").write_text(json.dumps(per_pid))
    by_method = {n: {r["pid"]: {k: r[k] for k in scoring.COUNT_KEYS} for r in rows if r["method"] == n} for n in dict.fromkeys(r["method"] for r in rows)}
    summary = {n: scoring.bootstrap(c) for n, c in by_method.items()}
    summary["expressible_classes"] = {n: list(CLASSES) for n in by_method}
    notes = ["node supply Lstar; headline excludes linking_unclear lesions (load_pairs include_unclear=False)",
             f"prop-fill: {args.prop_fill}; {len(missing)} lesions own no node and are scored as missed (table `missing`)",
             "Qahqaie registration-trust term omitted (w_J = 0): no deformation field available", "Di Veroli: cumulative dilation reading, no size filter",
             "secondary Di Veroli input (uniGradICON-warped BL masks) not implemented in this version"]
    if args.ours_run is not None:
        ours = json.loads((args.ours_run / "results.json").read_text())["tables"]["per_patient"]
        o = {r["pid"]: {k: r[k] for k in scoring.COUNT_KEYS} for r in ours if r.get("decoder", args.ours_decoder) == args.ours_decoder and r["pid"] in pids}
        summary["paired_delta_vs_ours"] = {n: scoring.paired_delta(c, o) for n, c in by_method.items() if set(c) == set(o)}
        notes.append(f"paired deltas vs exp03 run {args.ours_run.name} (decoder {args.ours_decoder}): {len(o)} patients")
    cols = ["recall_unchanged", "recall_disappeared", "recall_new", "recall_merged", "recall_macro", "edge_f1"]
    md = "| method | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n" + "\n".join(
        f"| {n} | " + " | ".join("n/a" if not np.isfinite(s[c][0]) else f"{s[c][0]:.3f} [{s[c][1]:.3f}, {s[c][2]:.3f}]" for c in cols) + " |"
        for n, s in summary.items() if n in by_method) + "\n"
    run.finish(summary, {"per_patient": rows, "chosen_params": chosen, "tuning_scores": tuning, "missing": missing}, table_md=md, notes=notes,
               definitions={**scoring.DEFINITIONS, "prop_fill": args.prop_fill, "folds": "experiments.exp03_matcher_alone.folds.assign_folds"},
               next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
