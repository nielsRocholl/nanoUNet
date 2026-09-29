"""Stage A audit: class balance, imputation rates, lesion-size quartiles, and (with --ckpt) error
stratification by registration provenance. Read-only; trains nothing.

Usage:
    PYTHONPATH=. python3 lesionglue/cli/audit.py --root /nnunet_data/Longitudinal-CT \
        --cache /nnunet_data/lesion_tracking/cache --split val --out runs/audit [--ckpt PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from lesionglue.common import dump_json, print0
from lesionglue.data.cache.dataset import LesionDataset
from lesionglue.data.source.meta import LesionRow, V2Paths, load_split_json, parse_meta_csv
from lesionglue.data.source.provenance import lesion_provenance
from lesionglue.model.decode import decode_sinkhorn_hungarian
from lesionglue.train.objective import split_per_graph
from lesionglue.train.module import MatcherModule

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/nnunet_data/Longitudinal-CT")
ap.add_argument("--cache", default="/nnunet_data/lesion_tracking/cache")
ap.add_argument("--split", choices=("train", "val", "test"), default="val")
ap.add_argument("--out", default="runs/audit")
ap.add_argument("--ckpt", default=None, help="if given, run section 4 (error stratification)")
ap.add_argument("--no-ema", action="store_true")
args = ap.parse_args()

root = Path(args.root)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

pids: list[str] = load_split_json(root / "data_split.json")[args.split]
rows_by_pid: dict[str, list[LesionRow]] = {pid: parse_meta_csv(V2Paths(root, pid).meta) for pid in pids}

# --- section 1: class balance ---------------------------------------------------------------
topo_counts: dict[str, int] = {}
for rows in rows_by_pid.values():
    for r in rows:
        topo_counts[r.topology] = topo_counts.get(r.topology, 0) + 1
n_split_native = topo_counts.get("SPLIT", 0)
print0(f"[{args.split}] class balance: {topo_counts}")
print0(f"[{args.split}] native SPLIT count = {n_split_native} (expected 0)")

# --- section 2: imputation rates -------------------------------------------------------------
n_bl = n_bl_imp = n_fu = n_fu_imp = 0
sanity_bad_cases: set[str] = set()
for pid in pids:
    for p in lesion_provenance(pid, root):
        if p.side == "bl":
            n_bl += 1
            n_bl_imp += int(p.imputed)
        else:
            n_fu += 1
            n_fu_imp += int(p.imputed)
        if p.sanity_bad:
            sanity_bad_cases.add(pid)
bl_imp_rate = n_bl_imp / max(n_bl, 1)
fu_imp_rate = n_fu_imp / max(n_fu, 1)
print0(f"[{args.split}] BL imputed rate: {bl_imp_rate:.4f} ({n_bl_imp}/{n_bl})")
print0(f"[{args.split}] FU imputed rate: {fu_imp_rate:.4f} ({n_fu_imp}/{n_fu})")
print0(f"[{args.split}] sanity_bad cases: {len(sanity_bad_cases)}/{len(pids)} patients")

# --- section 3: lesion-size distribution ------------------------------------------------------
vol_bl_all, vol_fu_all = [], []
for pid in pids:
    df = pd.read_csv(V2Paths(root, pid).meta)
    if "linking_unclear" in df.columns:
        df = df[df["linking_unclear"].fillna(False) != True]  # noqa: E712
    vol_bl_all.append(df["volume_bl"].dropna())
    vol_fu_all.append(df["volume_fu"].dropna())
vol_bl = pd.concat(vol_bl_all)
vol_fu = pd.concat(vol_fu_all)
qs = (0.25, 0.5, 0.75)
vol_bl_q = {str(q): float(v) for q, v in zip(qs, np.quantile(vol_bl, qs))}
vol_fu_q = {str(q): float(v) for q, v in zip(qs, np.quantile(vol_fu, qs))}
print0(f"[{args.split}] volume_bl quartiles (n={len(vol_bl)}): {vol_bl_q}")
print0(f"[{args.split}] volume_fu quartiles (n={len(vol_fu)}): {vol_fu_q}")

report = {
    "split": args.split,
    "n_patients": len(pids),
    "class_balance": topo_counts,
    "n_split_native": n_split_native,
    "imputation": {
        "bl_rate": bl_imp_rate, "bl_n": n_bl, "bl_n_imputed": n_bl_imp,
        "fu_rate": fu_imp_rate, "fu_n": n_fu, "fu_n_imputed": n_fu_imp,
        "n_sanity_bad_patients": len(sanity_bad_cases),
    },
    "volume_bl_quartiles": vol_bl_q,
    "volume_fu_quartiles": vol_fu_q,
}

# --- section 4: error stratification (only with --ckpt) ---------------------------------------
if args.ckpt is not None:
    ds = LesionDataset(root=args.cache, split=args.split, dataset_root=root)
    loader = PyGDataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    mod.set_eval_weights(not args.no_ema)
    mod.eval()
    strata = {
        "imputed": {"unchanged_split": [0, 0], "disappeared": [0, 0], "newly_appearing": [0, 0]},
        "observed": {"unchanged_split": [0, 0], "disappeared": [0, 0], "newly_appearing": [0, 0]},
    }
    with torch.no_grad():
        for batch in loader:
            out = mod.predict_batch(batch, use_ema=not args.no_ema)
            graphs, pp, db, dfu = split_per_graph(batch, out)
            for g, p, b, f in zip(graphs, pp, db, dfu):
                pid = g.pid
                prov = {(x.side, x.lesion_id): x.imputed for x in lesion_provenance(pid, root)}
                n_bl_g, n_fu_g = g["bl"].num_nodes, g["fu"].num_nodes
                lab = g["bl", "cross", "fu"].edge_label.reshape(n_bl_g, n_fu_g)
                # same decode call graph_val_counts uses -- do not reimplement Hungarian decoding.
                dec = decode_sinkhorn_hungarian(p, b, f, n_bl_g, n_fu_g, iters=mod.hparams.sinkhorn_iters, tau=float(mod.hparams.dust_tau))
                bl_ids, fu_ids = g["bl"].lesion_id.tolist(), g["fu"].lesion_id.tolist()
                claimed = {int(dec[i]) for i in range(n_bl_g) if int(dec[i]) >= 0}
                for i in range(n_bl_g):
                    pos = torch.where(lab[i] > 0.5)[0]
                    key = "imputed" if prov.get(("bl", bl_ids[i]), False) else "observed"
                    if pos.numel():
                        strata[key]["unchanged_split"][1] += 1
                        strata[key]["unchanged_split"][0] += int((pos == int(dec[i])).any().item())
                    elif float(g["bl"].no_match_label[i]) > 0.5:
                        strata[key]["disappeared"][1] += 1
                        strata[key]["disappeared"][0] += int(int(dec[i]) < 0)
                for j in range(n_fu_g):
                    if float(g["fu"].no_match_label[j]) > 0.5:
                        key = "imputed" if prov.get(("fu", fu_ids[j]), False) else "observed"
                        strata[key]["newly_appearing"][1] += 1
                        strata[key]["newly_appearing"][0] += int(j not in claimed)
    strat_report = {
        grp: {name: {"acc": ok / tot if tot else None, "n": tot} for name, (ok, tot) in sub.items()}
        for grp, sub in strata.items()
    }
    print0(f"[{args.split}] error stratification: {json.dumps(strat_report, indent=2)}")
    report["error_stratification"] = strat_report

dump_json(out_dir / f"audit_{args.split}.json", report)
print0(f"wrote {out_dir / f'audit_{args.split}.json'}")
