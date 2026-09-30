"""exp03 - Matcher alone  (paper: Sec. "Identity" > "The matcher alone"; Table 'nine experiments' row 3)

QUESTION   Given the annotated lesions of both scans (nothing to miss), which pairs are the same lesion? This is the ceiling of the identity
           task: every later experiment's drop (segmenter-supplied lesions, other datasets) is measured from here.
WHY        Fills the matcher row of the identity table: per-class recall (unchanged, disappeared, new, merged), identity ceiling and edge
           P/R/F1 with patient-level bootstrap intervals, for the `hungarian` and `sinkhorn` decoders.
DATA       All 300 Longitudinal-CT patients, 5 patient-level folds (`folds.assign_folds`, the same split `lesionglue_train --pool all` uses),
           one scan pair per patient (the graph of the dominant `img_id_fu`), node supply Lstar. Every patient is scored by a model that
           never saw them. Headline excludes linking_unclear lesions (the graph cache omits them).
METHOD     1. For each fold k train the selection-free recipe of the deployed model (`lesionglue/configs/complete.json`: fixed max_steps, no
              validation, no early stopping, `last.ckpt`) on the other four folds: `lesionglue_train --pool all --fold k --no-val`. Resumable:
              a fold whose last.ckpt exists is skipped.
           2. Score fold k with its own model (EMA weights): one inference pass stores the raw pair logits and dustbin scores per patient
              in artifacts/scores/<pid>.npz, so decoders and tau can be rerun offline (exp06 reads these files).
           3. Decode with `hungarian` and `sinkhorn` at tau = the checkpoint's dust_tau (fixed, never chosen on scored patients) and score
              with `experiments.scoring`; the whole tau grid is stored as a sensitivity table only.
           4. Patients without a graph are kept: empty BL or FU set by annotation = scored trivially (all disappeared / all new); a patient
              the graph builder could not build otherwise is counted as fully missed (status no_graph).
OUTPUT     results.json tables: per_patient (patient x decoder x tau: fold, status, node counts, class counts), per_fold, tau_sensitivity;
           summary: per decoder recall per class + macro, ceilings, matcher error, edge P/R/F1 with CIs. artifacts/folds/fold<k>/ (checkpoints,
           train logs), artifacts/scores/<pid>.npz.
COMMAND    python -m experiments.exp03_matcher_alone.run --tag paper_v1 --cache /nnunet_data/lesion_tracking/<rebuilt cache dir>
DEPENDS ON experiments.common, experiments.scoring, folds.py (this folder), the lesionglue console entry `lesionglue.cli.train`.
RUNTIME    5 trainings of 7400 steps (about 30 min each alone on one A100; `--parallel-folds` runs several at once) plus seconds of scoring.
CAVEATS    Numbers are only valid on a cache built with the fixed graph builder (merge-target nodes; plan Sec. 2) — the tag in the cache name
           must be the current lesionglue CACHE_TAG. The with-unclear sensitivity row is not produced by this version. Training uses the
           owner's recipe seed unless --seed is given; the fold split seed is fixed at 0.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
import argparse
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from core.ui import cprint, nano_progress
from experiments import scoring
from experiments.common import LONGI_ROOT, abort_if, add_common_args, limited, missing_paths, problem, start_run
from experiments.exp03_matcher_alone.folds import all_patients, assign_folds
from lesionglue.cli.eval import DUST_GRID
from lesionglue.common import eval_device
from lesionglue.data.cache.dataset import LesionDataset
from lesionglue.infer import graph_cfg_from_ckpt
from lesionglue.model.decode import decode_pairs
from lesionglue.train.module import MatcherModule
from lesionglue.train.objective import split_per_graph

EXP = "exp03_matcher_alone"
PAPER = {"section": "Identity > The matcher alone", "table_row": 3, "supports": "matcher row of the identity table; ceiling for every later experiment"}
DECODERS = ("hungarian", "sinkhorn")
POOL_SPLITS = ("train", "val", "test")  # `--pool all`: the caches that together hold all 300 patients
N_FOLDS = 5


def dominant_fu_id(root: Path, pid: str) -> int:
    """img_id_fu of the patient's scan pair: the most frequent one in the meta CSV (ties: the smallest), as `scoring.load_pairs`."""
    with open(root / "meta" / f"{pid}.csv", encoding="utf-8") as f:
        col = f.readline().strip().split(",").index("img_id_fu")
        n = Counter(int(float(line.split(",")[col])) for line in f if line.strip())
    return max(n, key=lambda k: (n[k], -k))


def train_fold(k: int, args: argparse.Namespace, fold_dir: Path) -> None:
    if (fold_dir / "last.ckpt").is_file():
        cprint(f"status: fold {k} training skipped | last.ckpt exists")
        return
    cmd = [sys.executable, "-m", "lesionglue.cli.train", "--config", str(args.config), "--root", str(args.data_root), "--cache", str(args.cache), "--out", str(fold_dir),
           "--pool", "all", "--fold", str(k), "--no-val", "--seed", str(args.seed)] + (["--max-steps", str(args.max_steps)] if args.max_steps > 0 else [])
    fold_dir.mkdir(parents=True, exist_ok=True)
    cprint(f"status: fold {k} training | cmd: {' '.join(cmd)}")
    with open(fold_dir / "train.log", "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    if code != 0 or not (fold_dir / "last.ckpt").is_file():
        raise SystemExit(f"fold {k} training failed (exit {code}); see {fold_dir / 'train.log'}\nExpected {fold_dir / 'last.ckpt'}.\nFix: {' '.join(cmd)}")


def predict_fold(k: int, pids: list[str], graphs: dict[str, object], ckpt: Path, scores: Path, dev: torch.device) -> None:
    mod = MatcherModule.load_from_checkpoint(str(ckpt), map_location=dev).to(dev).eval()
    todo = [graphs[p] for p in pids if p in graphs and not (scores / f"{p}.npz").is_file()]
    scores.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in PyGDataLoader(todo, batch_size=8, shuffle=False):
            batch = batch.to(dev)
            out = mod.predict_batch(batch, use_ema=True)
            for g, p, b, f in zip(*split_per_graph(batch, out)):
                np.savez_compressed(scores / f"{g.pid}.npz", pair=p.cpu().numpy(), dust_bl=b.cpu().numpy(), dust_fu=f.cpu().numpy(), bl_ids=g["bl"].lesion_id.cpu().numpy(),
                                    fu_ids=g["fu"].lesion_id.cpu().numpy(), sinkhorn_iters=int(mod.hparams.sinkhorn_iters), dust_tau=float(mod.hparams.dust_tau), fold=k,
                                    graph_id=str(g.graph_id))
    del mod


def score_patient(case: scoring.PairCase, npz: dict | None, decoder: str, tau: float) -> tuple[dict, str, int, int]:
    """(counts, status, n_bl_nodes, n_fu_nodes) for one patient, decoder and tau."""
    if npz is None:
        trivial = not case.bl_ids or not case.fu_ids
        return scoring.score_pair(case if trivial else replace(case, found_bl=set(), found_fu=set())), "trivial_empty_side" if trivial else "no_graph", 0, 0
    bl_ids, fu_ids = [int(i) for i in npz["bl_ids"]], [int(i) for i in npz["fu_ids"]]
    n_bl, n_fu = len(bl_ids), len(fu_ids)
    pairs = decode_pairs(decoder, torch.from_numpy(npz["pair"]), torch.from_numpy(npz["dust_bl"]), torch.from_numpy(npz["dust_fu"]), n_bl, n_fu, thresh=0.5,
                         sinkhorn_iters=int(npz["sinkhorn_iters"]), sinkhorn_tau=tau)
    c = replace(case, found_bl=set(bl_ids) & set(case.bl_ids), found_fu=set(fu_ids) & set(case.fu_ids), pred_links=scoring.pairs_to_links(pairs, bl_ids, fu_ids))
    return scoring.score_pair(c), "ok", n_bl, n_fu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap, rescore=True)
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="Longitudinal-CT layout root (meta/, targetsTrBL|FU/)")
    ap.add_argument("--cache", type=Path, required=True, help="graph cache root built with the fixed builder (must hold train, val and test caches of the current CACHE_TAG)")
    ap.add_argument("--config", type=Path, default=Path("lesionglue/configs/complete.json"), help="lesionglue training config (fixed max_steps recipe)")
    ap.add_argument("--max-steps", type=int, default=-1, help="override the config's max_steps; -1 = use the config (smoke runs use a few steps)")
    ap.add_argument("--parallel-folds", type=int, default=1, help="trainings run at the same time (one fits the GPU well; five saturate it)")
    ap.add_argument("--skip-training", action="store_true", help="only score folds whose last.ckpt already exists in this run (with --resume)")
    args = ap.parse_args()
    problems = missing_paths({"data root": args.data_root, "meta dir": args.data_root / "meta", "graph cache": args.cache / "processed", "config": args.config},
                             "mount /nnunet_data; build the cache with lesionglue_preprocess (see experiments/README.md, exp03)")
    if args.rescore is not None and args.skip_training:
        problems.append(problem("--rescore together with --skip-training", "--rescore never trains", "drop --skip-training"))
    pids = limited(all_patients(), args)
    if len(pids) == 0:
        problems.append(problem("no patients selected", "at least one patient", "raise --limit-patients or drop it"))
    if args.rescore is not None:
        absent = [p for p in pids if not (args.rescore / "artifacts" / "scores" / f"{p}.npz").is_file() and (args.data_root / "meta" / f"{p}.csv").is_file()]
        if len(absent) == len(pids):
            problems.append(problem(f"--rescore {args.rescore} has no scores for the selected patients", "artifacts/scores/<pid>.npz from a finished run", "rerun without --rescore"))
    abort_if(problems)
    fold_of = assign_folds(all_patients(), N_FOLDS, 0)
    cases = scoring.load_pairs(args.data_root, pids)
    run = start_run(EXP, ap, args, paper=PAPER, inputs={"meta dir": args.data_root / "meta", "config": args.config, "graph cache": args.cache / "processed"})
    scores_dir = (args.rescore / "artifacts" if args.rescore is not None else run.artifacts) / "scores"
    folds_dir = run.artifacts / "folds"
    need = sorted({fold_of[p] for p in pids})
    if args.rescore is None:
        if not args.skip_training:
            with ThreadPoolExecutor(max(1, args.parallel_folds)) as ex:
                list(ex.map(lambda k: train_fold(k, args, folds_dir / f"fold{k}"), need))
        dev = eval_device("auto")
        mod = MatcherModule.load_from_checkpoint(str(folds_dir / f"fold{need[0]}" / "last.ckpt"), map_location="cpu")
        gcfg = graph_cfg_from_ckpt(mod, int(getattr(mod.hparams, "k_intra", 8)))
        del mod
        want = {p: dominant_fu_id(args.data_root, p) for p in pids}
        graphs: dict[str, object] = {}
        for sp in POOL_SPLITS:
            ds = LesionDataset(root=args.cache, split=sp, dataset_root=args.data_root, cfg=gcfg)
            for i in range(len(ds)):
                g = ds[i]
                if str(g.pid) in want and int(torch.as_tensor(g.img_id_fu_used).reshape(-1)[0]) == want[str(g.pid)]:
                    graphs[str(g.pid)] = g
        cprint(f"status: graphs | selected patients: {len(pids)} | with a dominant-region graph: {len(graphs)}")
        for k in need:
            predict_fold(k, [p for p in pids if fold_of[p] == k], graphs, folds_dir / f"fold{k}" / "last.ckpt", scores_dir, dev)
    npz = {}
    for p in pids:
        f = scores_dir / f"{p}.npz"
        npz[p] = dict(np.load(f)) if f.is_file() else None
    have = [d for d in npz.values() if d is not None]
    tau_head = float(have[0]["dust_tau"]) if have else float("nan")
    rows, taus = [], sorted(set(DUST_GRID) | {tau_head})
    with nano_progress(len(pids), "scoring") as adv:
        for p in pids:
            for dec in DECODERS:
                for tau in taus:
                    counts, status, nb, nf = score_patient(cases[p], npz[p], dec, tau)
                    rows.append({"pid": p, "fold": fold_of[p], "decoder": dec, "tau": tau, "headline_tau": tau == tau_head, "status": status, "n_bl_nodes": nb, "n_fu_nodes": nf, **counts})
            adv(1)
    head = {d: {r["pid"]: {k: r[k] for k in scoring.COUNT_KEYS} for r in rows if r["decoder"] == d and r["headline_tau"]} for d in DECODERS}
    summary = {d: scoring.bootstrap(c) for d, c in head.items()}
    summary["headline_tau"] = tau_head
    per_fold = [{"decoder": d, "fold": k, **scoring.pooled([c for p, c in head[d].items() if fold_of[p] == k])} for d in DECODERS for k in need]
    sens = [{"decoder": d, "tau": t, **{m: v for m, v in scoring.pooled([{k: r[k] for k in scoring.COUNT_KEYS} for r in rows if r["decoder"] == d and r["tau"] == t]).items()
                                         if m in ("recall_macro", "recall_merged", "edge_f1")}} for d in DECODERS for t in taus]
    n_missed = sum(1 for r in rows if r["decoder"] == DECODERS[0] and r["headline_tau"] and r["status"] == "no_graph")
    cprint(f"status: patients without a usable graph (kept, fully missed): {n_missed}")
    cols = ["recall_unchanged", "recall_disappeared", "recall_new", "recall_merged", "recall_macro", "ceiling_unchanged", "ceiling_merged", "edge_f1"]
    md = "| decoder | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n" + "\n".join(
        f"| {d} | " + " | ".join("n/a" if not np.isfinite(summary[d][c][0]) else f"{summary[d][c][0]:.3f} [{summary[d][c][1]:.3f}, {summary[d][c][2]:.3f}]" for c in cols) + " |" for d in DECODERS) + "\n"
    notes = ["node supply Lstar; headline excludes linking_unclear lesions (graph cache omits them)", f"tau fixed at the checkpoint dust_tau = {tau_head}; grid only as sensitivity",
             f"{n_missed} patients had no usable graph and are counted as fully missed", "with-unclear sensitivity row not produced by this version",
             f"graph cache: {args.cache}", "numbers are valid only on a cache built with the fixed graph builder (plan Sec. 2)"]
    run.finish(summary, {"per_patient": rows, "per_fold": per_fold, "tau_sensitivity": sens}, table_md=md, notes=notes,
               definitions={**scoring.DEFINITIONS, "folds": "experiments.exp03_matcher_alone.folds.assign_folds (lesionglue fold_map, 5 folds, seed 0)", "decoders": list(DECODERS)},
               next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
