"""exp06 - Limits  (paper: Sec. "Identity" > "Limits: merges and splits"; Table 'nine experiments' row 6)

QUESTION   What can the formulation express that the data cannot score, and how far does the many-to-one decoder get on the merges we do have?
WHY        Supports the paper's merge and split statements: the data holds no split labels, merges are rare (38 events), a strictly one-to-one
           decoder cannot express them, and the `sinkhorn` rule recovers a k-way merge only while 1/k >= tau.
DATA       The per-patient raw matcher scores stored by an exp03 run (`artifacts/scores/<pid>.npz`, out-of-fold by construction) and the
           Longitudinal-CT meta CSVs of all 300 patients. No new inference, CPU only.
METHOD     1. Audit: count `SPLIT` topology rows over all meta CSVs (expected 0) and record split recall as null with the reason.
           2. Merge structure from the annotation: events, group sizes (histogram), which events are in the held-out 60.
           3. Decode every stored patient with `hungarian` (strictly one-to-one) and `sinkhorn` (every plan cell whose row-normalised mass
              clears tau) at the checkpoint's dust_tau; a merged BL lesion is correct iff its (id, merge target) edge is decoded and both
              nodes exist; recall pooled per contributing lesion, with patient-level bootstrap intervals, overall and by group size.
           4. The analytical limit: a contributor holds about 1/k of the shared column's unit mass, so sinkhorn can only recover k <= 1/tau.
           5. An expressibility table: which classes each decoder or baseline can express at all.
OUTPUT     results.json tables: merge_events, merge_contributors (one row per contributing lesion and decoder), merge_recall_by_k, expressibility;
           summary: split audit, group-size histogram, merge recall per decoder with CIs, tau and k_max.
COMMAND    python -m experiments.exp06_limits.run --tag paper_v1 --from-run /nnunet_data/experiments/exp03_matcher_alone/<run_id>
DEPENDS ON experiments.common, experiments.scoring, an exp03 run (stored scores).
RUNTIME    Seconds (CPU). Resumable trivially: rerun.
CAVEATS    Merge recall is only meaningful for scores produced on a cache built with the fixed graph builder (merge-target nodes, plan Sec. 2);
           on older caches every merge target lacks a node and recall is 0 by construction. Events whose target is linking_unclear are excluded
           from the headline (as in `scoring.load_pairs`) but counted in the structure table. Patients without stored scores are listed in `notes`.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from core.ui import cprint
from experiments import scoring
from experiments.common import HOLDOUT_CSV, LONGI_ROOT, abort_if, add_common_args, limited, missing_paths, problem, start_run
from lesionglue.model.decode import decode_pairs

EXP = "exp06_limits"
PAPER = {"section": "Identity > Limits: merges and splits", "table_row": 6, "supports": "merge recall under both decoders, merge structure, split audit"}
DECODERS = ("hungarian", "sinkhorn")
K_BINS = (("2", 2, 2), ("3", 3, 3), ("4-7", 4, 7), (">=8", 8, 10**6))
EXPRESSIBILITY = [
    {"method": "hungarian (strict one-to-one)", "unchanged": True, "disappeared": True, "new": True, "merged": False, "split": False, "note": "at most one contributor per target survives"},
    {"method": "sinkhorn (row mass >= tau)", "unchanged": True, "disappeared": True, "new": True, "merged": "k <= 1/tau", "split": "k <= 1/tau (no labels)", "note": "a contributor holds about 1/k of the column"},
    {"method": "dense (sigmoid pair logits)", "unchanged": True, "disappeared": True, "new": True, "merged": True, "split": True, "note": "ignores the transport plan"},
    {"method": "Di Veroli / Qahqaie (degrees)", "unchanged": True, "disappeared": True, "new": True, "merged": True, "split": True, "note": "classes come from node degrees"},
]


def holdout_ids() -> set[str]:
    with open(HOLDOUT_CSV, newline="", encoding="utf-8") as f:
        return {r["patient"].strip() for r in csv.DictReader(f)}


def split_audit(root: Path, pids: list[str]) -> dict:
    n = Counter()
    for p in pids:
        with open(root / "meta" / f"{p}.csv", newline="", encoding="utf-8") as f:
            n.update(r["topology_class"].strip() for r in csv.DictReader(f))
    return {"topology_rows": dict(n), "split_rows": n.get("SPLIT", 0) + n.get("SPLITTING", 0), "split_recall": None,
            "reason": "no annotated split events exist in Longitudinal-CT, so split recall cannot be scored"}


def events_of(case: scoring.PairCase) -> dict[int, list[int]]:
    ev: dict[int, list[int]] = defaultdict(list)
    for b, f in sorted(case.links):
        if case.topology[b] == "MERGED":
            ev[f].append(b)
    return ev


def decoded(npz: dict, decoder: str, tau: float) -> tuple[list[int], list[int], set[tuple[int, int]]]:
    bl, fu = [int(i) for i in npz["bl_ids"]], [int(i) for i in npz["fu_ids"]]
    pairs = decode_pairs(decoder, torch.from_numpy(npz["pair"]), torch.from_numpy(npz["dust_bl"]), torch.from_numpy(npz["dust_fu"]), len(bl), len(fu), thresh=0.5,
                         sinkhorn_iters=int(npz["sinkhorn_iters"]), sinkhorn_tau=tau)
    return bl, fu, scoring.pairs_to_links(pairs, bl, fu)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, gpu=False)
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="Longitudinal-CT layout root (meta/)")
    ap.add_argument("--from-run", type=Path, required=True, help="exp03 RUN_DIR whose artifacts/scores/<pid>.npz are decoded again")
    args = ap.parse_args()
    scores = args.from_run / "artifacts" / "scores"
    problems = missing_paths({"data root": args.data_root, "meta dir": args.data_root / "meta", "exp03 scores": scores, "holdout csv": HOLDOUT_CSV},
                             "run exp03 first (python -m experiments.exp03_matcher_alone.run ...) and pass its run dir as --from-run")
    abort_if(problems)
    all_pids = sorted(p.stem for p in (args.data_root / "meta").glob("*.csv"))
    pids = limited(sorted(f.stem for f in scores.glob("*.npz")), args)
    abort_if([problem(f"{scores} holds no scores", "artifacts/scores/<pid>.npz from exp03", "rerun exp03")] if not pids else [])
    run = start_run(EXP, ap, args, paper=PAPER, inputs={"meta dir": args.data_root / "meta", "exp03 scores": scores})
    held = holdout_ids()
    structure = scoring.load_pairs(args.data_root, all_pids, include_unclear=True)
    events = [{"pid": p, "target": f, "k": len(b), "in_holdout": p in held, "contributors": b} for p, c in structure.items() for f, b in events_of(c).items()]
    hist = Counter(e["k"] for e in events)
    cases = scoring.load_pairs(args.data_root, pids)
    npz = {p: dict(np.load(scores / f"{p}.npz")) for p in pids}
    tau = float(next(iter(npz.values()))["dust_tau"])
    rows, by_patient = [], {d: {} for d in DECODERS}
    for p in pids:
        for d in DECODERS:
            bl, fu, links = decoded(npz[p], d, tau)
            items = []
            for f, contrib in events_of(cases[p]).items():
                for b in contrib:
                    node = b in bl and f in fu
                    ok = node and (b, f) in links
                    rows.append({"pid": p, "decoder": d, "target": f, "contributor": b, "k": len(contrib), "nodes_exist": node, "correct": ok})
                    items.append((len(contrib), int(ok), int(node)))
            if items:
                by_patient[d][p] = items
    kmax = int(math.floor(1.0 / tau + 1e-9))
    summary = {"split_audit": split_audit(args.data_root, all_pids), "merge_events_all": len(events), "merge_group_size_histogram": dict(sorted(hist.items())),
               "merge_events_holdout": [{"pid": e["pid"], "k": e["k"]} for e in events if e["in_holdout"]], "tau": tau, "sinkhorn_k_max": kmax}
    by_k = []
    for d in DECODERS:
        if not by_patient[d]:
            summary[d] = {"merge_recall": None, "reason": "no merge events among the stored patients"}
            continue
        rec = lambda lo, hi: (lambda it: sum(o for k, o, _ in it if lo <= k <= hi) / max(1, sum(1 for k, _, _ in it if lo <= k <= hi)))
        summary[d] = {"merge_recall": scoring.bootstrap_stat(by_patient[d], rec(2, 10**6)),
                      "merge_ceiling": scoring.bootstrap_stat(by_patient[d], lambda it: sum(c for _, _, c in it) / max(1, len(it)))}
        for name, lo, hi in K_BINS:
            n = sum(1 for it in by_patient[d].values() for k, _, _ in it if lo <= k <= hi)
            if n:
                by_k.append({"decoder": d, "k": name, "n_contributors": n, "recall": scoring.bootstrap_stat(by_patient[d], rec(lo, hi)), "sinkhorn_recoverable": lo <= kmax})
    missing = [p for p in all_pids if p not in npz][:5]
    notes = ["merge recall counts a contributor correct iff (its id, merge target) is decoded and both nodes exist", f"tau = checkpoint dust_tau = {tau}; k <= {kmax} is recoverable by sinkhorn",
             f"{len(all_pids) - len(pids)} of {len(all_pids)} patients have no stored scores (e.g. {missing}); structure counts use all {len(all_pids)}",
             "on caches built before the merge-target fix every merge recall is 0 by construction"]
    md = "| decoder | merge recall [CI] | ceiling |\n|---|---|---|\n" + "\n".join(
        f"| {d} | " + (f"{summary[d]['merge_recall'][0]:.3f} [{summary[d]['merge_recall'][1]:.3f}, {summary[d]['merge_recall'][2]:.3f}] | {summary[d]['merge_ceiling'][0]:.3f}" if summary[d].get("merge_ceiling")
                      else "n/a | n/a") + " |" for d in DECODERS) + f"\n\nGroup sizes (all patients): {dict(sorted(hist.items()))}; split rows: {summary['split_audit']['split_rows']}\n"
    cprint(f"status: merge events (all patients): {len(events)} | contributors scored: {len(rows) // 2}")
    run.finish(summary, {"merge_events": events, "merge_contributors": rows, "merge_recall_by_k": by_k, "expressibility": EXPRESSIBILITY}, table_md=md, notes=notes,
               definitions={**scoring.DEFINITIONS, "decoders": list(DECODERS)}, next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
