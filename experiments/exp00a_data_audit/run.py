# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
"""exp00a - Data audit  (paper: Sec. "Data and evaluation tiers", Table 'datasets'; plan Sec. 6 exp00a)

QUESTION   Is every number the paper states about its data true of the files on disk, and where is it not?
WHY        Fixes the counts the paper quotes (300 patients, 4530 lesions, 38 merge events, 5690 volumes, PanTrack sizes...)
           and shows the exact effect of the `linking_unclear` decision on them. A mismatch is reported, never fixed here:
           it means the paper text (or the data) must change.
DATA       Longitudinal-CT meta CSVs (300 patients, the released split, the held-out 60), the merged corpus of the
           segmenter (`NanoUNet_preprocessed/Dataset900_Merged`: cohorts, splits, `*_centroids.json` sidecars), the
           matcher's graph-cache metadata, and PanTrack (`patients/tracking/organ_annotations.json` + its 161 label files).
METHOD     1. The paper's pair rule: per patient the rows whose `img_id_fu` is the most frequent one, NO `linking_unclear`
              filter (the paper's counts only reproduce this way). The graph builder instead drops unclear rows and keeps
              every FU region; both views are tabulated (`unclear_reconciliation`).
           2. Every quoted number becomes one row of `claims` {claim, paper_value, measured_value, match, source}. Integers
              must be equal; a rounded number (4.6, 42 %) must equal the measurement rounded to the paper's precision.
           3. Graph-cache membership is NOT read from the big `.pt` caches. The builder's rules
              (lesionglue/data/graph/dense.py: drop unclear rows, BL nodes need `cog_propagated`, FU nodes need `cog_fu`,
              one graph per FU region, empty side = no graph) are replayed on the meta CSVs and checked against the
              cache's tiny `*_meta.pt` files (total cross edges and positives per split must be equal).
           4. Prompt-encoding numbers come from the `*_centroids.json` sidecars of the 5690 preprocessed volumes: median
              equivalent-sphere radius (from `volume_vox`) and the share of lesions whose nearest other lesion centroid is
              closer than 30 voxels (all lesions in the denominator). Alternative definitions go to `prompt_variants`.
           5. PanTrack: counts from the three JSON files plus a scan of the 161 label NIfTIs (instance values per scan).
OUTPUT     results.json `claims`, `per_patient`, `merge_events`, `unclear_reconciliation`, `graph_cache`, `graph_missing`,
           `empty_sets`, `click_audit`, `no_cog_propagated` (lesions the graph builder drops for lack of a propagated point),
           `special_patients`, `cohorts`, `prompt_variants`, `pantrack_scans`, `pantrack_pairs`;
           `table.md` shows the claims (OK/MISMATCH), the reconciliation and the graph-cache check.
COMMAND    python -m experiments.exp00a_data_audit.run --tag paper_v1
DEPENDS ON experiments/common.py only (no other experiment).
RUNTIME    About 2 min on CPU (5690 small JSON reads, 161 PanTrack label scans); not resumable (nothing heavy to resume).
CAVEATS    `match` is exact for counts and rounded for the two prompt-encoding numbers; whether "42 %" means the definition
           used here is unknown (the script that produced it is not in the repo), so the variants are listed next to it.
           Graph-cache membership is inferred by replay, verified by totals, not by opening the cache.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from rich.table import Table
from scipy.spatial import cKDTree

from core.ui import console, cprint, nano_progress
from experiments.common import HOLDOUT_CSV, LONGI_ROOT, REPO, abort_if, add_common_args, limited, missing_paths, problem, start_run

EXP = "exp00a_data_audit"
CORPUS_DIR = Path("/nnunet_data/NanoUNet_preprocessed/Dataset900_Merged")
PANTRACK_DIR = Path("/nnunet_data/raw/PanTrack")
GRAPH_SPLIT = REPO / "lesionglue" / "configs" / "split.json"
GRAPH_CACHE_DIR = Path("/nnunet_data/lesion_tracking/cache_v9_merge/processed")
GRAPH_CACHE_TAG = "v8_native"
RAW_DIR = Path("/nnunet_data/NanoUNet_raw")
CLASSES = ("UNCHANGED", "DISAPPEARING", "NEWLYAPPEARING", "MERGING")
BL_TOPO, FU_TOPO = ("UNCHANGED", "DISAPPEARING", "MERGING"), ("UNCHANGED", "NEWLYAPPEARING")
SCOPES = ("all300", "official_train240", "official_val30", "official_test30", "holdout60")
KNOWN_BAD = "3988c7f88e"
# paper Table 'datasets': (region, cohort as printed, corpus id, volumes as printed)
PAPER_COHORTS = [
    ("Whole body", "Longitudinal-CT", "d013", 537), ("Thorax", "LIDC-IDRI", "d024", 897), ("Thorax", "LNDb", "d012", 216),
    ("Thorax", "MSD Lung", "d015", 63), ("Thorax", "RIDER Lung CT", "d029", 59), ("Thorax", "Mediastinal LN", "d030", 120),
    ("Liver", "MCT-LTDiag", "d027", 516), ("Liver", "WAW-TACE", "d019", 232), ("Liver", "CRLM", "d011", 197),
    ("Liver", "MSD Liver", "d017", 130), ("Liver", "LiTS", "d023", 126), ("Liver", "WORC CRLM", "d020", 77),
    ("Pancreas", "PanTS", "d028", 880), ("Pancreas", "MSD Pancreas", "d016", 281), ("Pancreas", "RUMC Pancreas", "d026", 13),
    ("Other abdomen", "KiTS23", "d022", 489), ("Other abdomen", "MSWAL", "d018", 284), ("Other abdomen", "WORC GIST", "d021", 245),
    ("Other abdomen", "MSD Colon", "d014", 126), ("Other abdomen", "Adrenal-ACC-Ki67", "d031", 51), ("Bone", "RUMC Bone", "d025", 151),
]


def claim(rows: list[dict], name: str, paper, measured, source: str, *, digits: int | None = None, stated_in: str = "paper", note: str = "") -> None:
    """One `claims` row; digits=None means exact equality, else the measurement rounded to `digits` must equal the paper value."""
    ok = (measured == paper) if digits is None else (round(measured, digits) == paper)
    rows.append({"claim": name, "paper_value": paper, "measured_value": measured, "match": bool(ok), "source": source, "stated_in": stated_in, "note": note})


def md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |\n" for r in rows)


def load_meta(root: Path, pids: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Meta CSV per patient; a CSV without the `linking_unclear` column (2 of 300) counts as not unclear, as parse_meta_csv does."""
    metas, no_col = {}, []
    for pid in pids:
        df = pd.read_csv(root / "meta" / f"{pid}.csv")
        if "linking_unclear" not in df.columns:
            no_col.append(pid)
            df["linking_unclear"] = False
        assert not df["linking_unclear"].isna().any(), f"{pid}: NaN in linking_unclear (bool(NaN) is True in parse_meta_csv, so rows would silently drop)"
        df["linking_unclear"] = df["linking_unclear"].astype(bool)
        assert set(df["topology_class"]) <= set(CLASSES), f"{pid}: topology {sorted(set(df['topology_class']) - set(CLASSES))} not in {CLASSES}"
        metas[pid] = df
    return metas, no_col


def dominant_pair(df: pd.DataFrame) -> pd.DataFrame:
    """The paper's scan pair: rows of the most frequent img_id_fu (same expression as lesionglue.data.source.meta._img_ids)."""
    return df[df["img_id_fu"] == int(df["img_id_fu"].value_counts().idxmax())]


def topo_counts(df: pd.DataFrame) -> dict[str, int]:
    c = Counter(df["topology_class"])
    return {k: int(c.get(k, 0)) for k in CLASSES}


def merge_groups(df: pd.DataFrame) -> dict[float, list[int]]:
    """merged_into (FU lesion id) -> BL lesion ids merging into it."""
    m = df[df["topology_class"] == "MERGING"]
    assert not m["merged_into"].isna().any(), "MERGING row without merged_into"
    return {float(k): sorted(int(x) for x in g["lesion_id"]) for k, g in m.groupby("merged_into")}


def replay_graphs(df: pd.DataFrame) -> list[dict]:
    """Graphs the cache builder makes for one patient (lesionglue/data/graph/dense.py replayed on the CSV): unclear rows dropped, one graph per FU region."""
    df = df[~df["linking_unclear"]]
    out = []
    for fu_id in sorted(set(df["img_id_fu"])):
        rows = df[df["img_id_fu"] == fu_id]
        has = lambda c: rows[c].notna() & (rows[c].astype(str).str.strip() != "")
        bl = set(rows.loc[rows["topology_class"].isin(BL_TOPO) & has("cog_propagated"), "lesion_id"])
        fu = set(rows.loc[rows["topology_class"].isin(FU_TOPO) & has("cog_fu"), "lesion_id"])
        if not bl or not fu:
            out.append({"fu_id": int(fu_id), "graph": False, "n_bl": len(bl), "n_fu": len(fu), "edges": 0, "positives": 0, "merge_rows_without_fu_node": 0})
            continue
        pos = {(r.lesion_id, r.lesion_id) for r in rows.itertuples() if r.topology_class == "UNCHANGED" and r.lesion_id in bl and r.lesion_id in fu}
        pos |= {(r.lesion_id, int(r.merged_into)) for r in rows.itertuples() if r.topology_class == "MERGING" and pd.notna(r.merged_into) and r.lesion_id in bl and int(r.merged_into) in fu}
        lost = sum(1 for r in rows.itertuples() if r.topology_class == "MERGING" and int(r.merged_into) not in fu)  # merge target has no FU row, so no FU node to link to
        out.append({"fu_id": int(fu_id), "graph": True, "n_bl": len(bl), "n_fu": len(fu), "edges": len(bl) * len(fu), "positives": len(pos), "merge_rows_without_fu_node": lost})
    return out


def patient_rows(metas: dict, official: dict, holdout: set, graph_split: dict) -> tuple[list[dict], list[dict], dict]:
    """per_patient rows, merge_events rows and the replayed graphs per patient."""
    g_of = {p: s for s in ("train", "val", "test") for p in graph_split[s]}
    per, events, graphs = [], [], {}
    for pid, df in metas.items():
        pair = dominant_pair(df)
        dom = int(pair["img_id_fu"].iloc[0])
        groups = merge_groups(pair)
        gr = replay_graphs(df)
        graphs[pid] = gr
        clear = pair[~pair["linking_unclear"]]
        row = {"patient": pid, "official_split": official.get(pid, "unknown"), "in_holdout60": pid in holdout, "graph_split": g_of.get(pid, "none"),
               "n_rows_all_regions": len(df), "n_fu_regions": int(df["img_id_fu"].nunique()), "dominant_fu": dom, "n_lesions_pair": len(pair),
               **{f"n_{k.lower()}": v for k, v in topo_counts(pair).items()},
               "n_unclear_pair": int(pair["linking_unclear"].sum()), "n_unclear_all_regions": int(df["linking_unclear"].sum()),
               **{f"n_unclear_{k.lower()}": int(((pair["topology_class"] == k) & pair["linking_unclear"]).sum()) for k in CLASSES},
               "n_merge_events": len(groups), "merge_group_sizes": sorted(len(v) for v in groups.values()),
               "n_bl_pair": int(pair["topology_class"].isin(BL_TOPO).sum()), "n_fu_pair": int(pair["topology_class"].isin(FU_TOPO).sum()),
               "n_bl_pair_clear": int(clear["topology_class"].isin(BL_TOPO).sum()), "n_fu_pair_clear": int(clear["topology_class"].isin(FU_TOPO).sum()),
               "n_graphs_cache": sum(g["graph"] for g in gr), "dominant_graph_present": any(g["graph"] and g["fu_id"] == dom for g in gr)}
        per.append(row)
        for k, ids in groups.items():
            unclear = sorted(int(x) for x in pair[(pair["topology_class"] == "MERGING") & (pair["merged_into"] == k) & pair["linking_unclear"]]["lesion_id"])
            events.append({"patient": pid, "merged_into": int(k), "group_size": len(ids), "bl_lesion_ids": ids, "unclear_ids": unclear, "in_holdout60": pid in holdout})
    return per, events, graphs


def scope_members(per: list[dict], scope: str) -> list[dict]:
    if scope == "all300":
        return per
    if scope == "holdout60":
        return [r for r in per if r["in_holdout60"]]
    return [r for r in per if r["official_split"] == scope.split("_")[1].rstrip("0123456789")]


def reconciliation(per: list[dict]) -> list[dict]:
    """Counts per class for the paper rule (unclear kept), the same without unclear rows, and the unclear rows alone, per scope."""
    views = {"paper_rule_unclear_kept": lambda r, k: r[f"n_{k.lower()}"], "unclear_excluded": lambda r, k: r[f"n_{k.lower()}"] - r[f"n_unclear_{k.lower()}"],
             "unclear_only": lambda r, k: r[f"n_unclear_{k.lower()}"]}
    rows = []
    for scope in SCOPES:
        sel = scope_members(per, scope)
        for view, count in views.items():
            c = {k: sum(count(r, k) for r in sel) for k in CLASSES}
            rows.append({"scope": scope, "view": view, "n_patients": len(sel), "n_lesions": sum(c.values()), **c})
    return rows


def read_cache_meta(paths: dict[str, Path]) -> dict[str, dict]:
    """The three ~1 kB metadata dicts of the graph caches (cross edges, positives); the big caches are never opened."""
    import torch

    return {s: torch.load(p, weights_only=False) for s, p in paths.items()}


def graph_check(per: list[dict], graphs: dict, split: dict, meta_pt: dict) -> tuple[list[dict], list[dict]]:
    """Replayed totals vs the cache's own *_meta.pt totals per split, and the split patients that end up with no graph."""
    rows, missing, by_pid = [], [], {r["patient"]: r for r in per}
    for s in ("train", "val", "test"):
        gs = [(p, g) for p in split[s] if p in graphs for g in graphs[p] if g["graph"]]
        pats = {p for p, _ in gs}
        edges, pos = sum(g["edges"] for _, g in gs), sum(g["positives"] for _, g in gs)
        rows.append({"split": s, "n_patients_in_split": len(split[s]), "n_patients_with_graph": len(pats), "n_graphs": len(gs), "n_patients_multi_graph": sum(1 for p in pats if sum(g["graph"] for g in graphs[p]) > 1),
                     "edges_replayed": edges, "positives_replayed": pos, "edges_cache_meta": meta_pt[s]["edges"], "positives_cache_meta": meta_pt[s]["positives"],
                     "matches_cache_meta": edges == meta_pt[s]["edges"] and pos == meta_pt[s]["positives"]})
        for p in (p for p in split[s] if p in graphs and p not in pats):
            r = by_pid[p]
            missing.append({"patient": p, "graph_split": s, "why": "; ".join(f"FU region {g['fu_id']}: {g['n_bl']} BL nodes, {g['n_fu']} FU nodes" for g in graphs[p]) or "no rows left after dropping linking_unclear",
                            "n_bl_pair": r["n_bl_pair"], "n_fu_pair": r["n_fu_pair"], "n_unclear_pair": r["n_unclear_pair"], "in_holdout60": r["in_holdout60"]})
    return rows, missing


def click_audit(root: Path, metas: dict) -> list[dict]:
    """Click JSONs vs meta: every BL scan's `inputsTrBL/{pid}_{idx}.json` must hold each lesion's cog_bl, every FU scan's `inputsTrFU` its cog_propagated."""
    rows = []
    xyz = lambda s: np.array([float(x) for x in s.split()])
    for pid, df in metas.items():
        for side, col, idcol, folder in (("BL", "cog_bl", "img_id_bl", "inputsTrBL"), ("FU", "cog_propagated", "img_id_fu", "inputsTrFU")):
            for idx in sorted(set(df[idcol])):
                path, sub = root / folder / f"{pid}_{int(idx):02d}.json", df[df[idcol] == idx]
                want = sub[sub[col].notna() & (sub[col].astype(str).str.strip() != "")]
                if not path.is_file():
                    rows.append({"patient": pid, "side": side, "scan_index": int(idx), "status": "json missing", "n_expected": len(want), "n_missing_points": len(want), "n_wrong_points": 0})
                    continue
                pts = {p["name"]: np.array(p["point"], float) for p in json.loads(path.read_text())["points"]}
                absent = [str(int(r.lesion_id)) for r in want.itertuples() if str(int(r.lesion_id)) not in pts]
                wrong = [str(int(r.lesion_id)) for r in want.itertuples() if str(int(r.lesion_id)) in pts and np.abs(pts[str(int(r.lesion_id))] - xyz(getattr(r, col))).max() > 1e-3]
                if absent or wrong:
                    rows.append({"patient": pid, "side": side, "scan_index": int(idx), "status": "points missing/wrong", "n_expected": len(want), "n_missing_points": len(absent), "n_wrong_points": len(wrong)})
    return rows


def corpus_checks(claims: list, corpus: Path, holdout: set, official: dict, workers: int, limit_args, raw_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """tab:datasets, the 537/240 arithmetic, the 15 % validation share and the prompt-encoding numbers from the sidecars."""
    cohorts_json = json.loads((corpus / "cohorts.json").read_text())
    dataset = json.loads((corpus / "dataset.json").read_text())["dataset"]
    split = json.loads((corpus / "splits_final.json").read_text())[0]
    plan = json.loads((corpus / "nnUNetResEncUNetLPlans_h200_smallpv.json").read_text())["configurations"]["3d_fullres"]
    cohort_of = lambda k: k.split("_")[0]
    n_train, n_val = Counter(map(cohort_of, split["train"])), Counter(map(cohort_of, split["val"]))
    n_dataset, n_json = Counter(map(cohort_of, dataset)), {k: v["cases"] for k, v in cohorts_json["counts"].items()}
    rows = []
    for region, name, cid, paper_n in PAPER_COHORTS:
        n = n_train[cid] + n_val[cid]
        share = n_val[cid] / n
        rows.append({"cohort_id": cid, "paper_name": name, "region": region, "paper_volumes": paper_n, "volumes_splits_final": n, "volumes_cohorts_json": n_json[cid], "volumes_dataset_json": n_dataset[cid],
                     "n_train": n_train[cid], "n_val": n_val[cid], "val_share": round(share, 4), "val_within_one_case_of_15pct": abs(n_val[cid] - 0.15 * n) <= 1})
        claim(claims, f"tab:datasets volumes, {name}", paper_n, n, f"splits_final.json train+val, prefix {cid}")
    total = sum(r["volumes_splits_final"] for r in rows)
    claim(claims, "tab:datasets total volumes", 5690, total, "splits_final.json train+val")
    claim(claims, "5690 volumes: cohorts.json total", 5690, sum(n_json.values()), "cohorts.json counts")
    claim(claims, "5690 volumes: dataset.json entries", 5690, len(dataset), "dataset.json", note="dataset.json also lists 106 RUMC pancreas volumes (d026: 119 vs 13) that are in neither splits_final.json nor the preprocessed folder")
    claim(claims, "21 cohorts (corpus)", 21, len({cohort_of(k) for k in split["train"] + split["val"]}), "splits_final.json prefixes")
    claim(claims, "21 cohorts (NanoUNet_raw dirs Dataset011..031)", 21, len([p for p in raw_dir.glob("Dataset0[1-3][0-9]_*") if 11 <= int(p.name[7:10]) <= 31]), "NanoUNet_raw listing")
    claim(claims, "15 % of each cohort held out for validation (pooled share, %)", 15, 100 * sum(n_val.values()) / total, "splits_final.json", digits=0)
    bad = [r["paper_name"] for r in rows if not r["val_within_one_case_of_15pct"]]
    claim(claims, "15 % of each cohort held out for validation (every cohort within one case of 15 %)", 0, len(bad), "splits_final.json", note=f"cohorts off by more than one case: {bad}")
    longi_keys = [k for k in split["train"] + split["val"] if cohort_of(k) == "d013"]
    pid_of = lambda k: k.split("_")[3]
    pids = {pid_of(k) for k in longi_keys}
    claim(claims, "Longitudinal-CT enters the corpus as 537 separate scans", 537, len(longi_keys), "splits_final.json d013")
    claim(claims, "... from 240 patients", 240, len(pids), "splits_final.json d013 case names")
    claim(claims, "... the 240 are the released split's training patients", 240, len(pids & {p for p, s in official.items() if s == "train"}), "data_split.json train")
    claim(claims, "... none of the held-out 60 is in the corpus", 0, len(pids & holdout), "splits_final.json d013 vs test_patients.csv")
    tr_p, va_p = {pid_of(k) for k in split["train"] if cohort_of(k) == "d013"}, {pid_of(k) for k in split["val"] if cohort_of(k) == "d013"}
    cprint(f"corpus: d013 validation split is patient-level: train {len(tr_p)} / val {len(va_p)} patients, overlap {len(tr_p & va_p)}")
    ids = limited(split["train"] + split["val"], limit_args)
    sidecar = lambda c: json.loads((corpus / "nnUNetPlans_3d_fullres" / f"{c}_centroids.json").read_text())
    with nano_progress(len(ids), "reading centroid sidecars") as advance, ThreadPoolExecutor(workers) as ex:
        side = []
        for r in ex.map(sidecar, ids):
            side.append(r)
            advance()
    vol, coh, nn = [], [], {"centroid": [], "seed": []}
    for c, d in zip(ids, side):
        vol += d["volume_vox"]
        coh += [cohort_of(c)] * len(d["volume_vox"])
        for key, pts in (("centroid", d["centroids_zyx"]), ("seed", d["seed_zyx"])):
            pts = np.array(pts, float).reshape(-1, 3)
            nn[key].append((cohort_of(c), len(pts), cKDTree(pts).query(pts, k=2)[0][:, 1] if len(pts) >= 2 else np.full(len(pts), np.inf), pts))
    radius = (3 * np.array(vol, float) / (4 * math.pi)) ** (1 / 3)
    d_all = np.concatenate([x[2] for x in nn["centroid"]])
    claim(claims, "median lesion radius (equivalent sphere, vox) in the pooled corpus", 4.6, float(np.median(radius)), "centroids.json volume_vox, all lesions of the 5690 volumes", digits=1)
    claim(claims, "neighbour within 30 vox for x % of lesions", 42, 100 * float((d_all < 30).mean()), "centroids.json centroids_zyx, nearest other lesion, Euclidean voxels, all lesions", digits=0)
    scale = np.array(plan["spacing"])
    multi = np.concatenate([x[2] for x in nn["centroid"] if x[1] >= 2])
    mm = np.concatenate([cKDTree(x[3] * scale).query(x[3] * scale, k=2)[0][:, 1] for x in nn["centroid"] if x[1] >= 2])
    per_coh = {c: float(np.median(radius[np.array(coh) == c])) for c in sorted(set(coh))}
    variants = [
        {"variant": "PRIMARY: share of all lesions with nearest other centroid < 30 vox (single-lesion volumes count as no neighbour)", "value": float((d_all < 30).mean()), "n": len(d_all)},
        {"variant": "share among lesions in volumes with >= 2 lesions, voxels", "value": float((multi < 30).mean()), "n": len(multi)},
        {"variant": "share among lesions in volumes with >= 2 lesions, distance in mm (plan spacing zyx)", "value": float((mm < 30).mean()), "n": len(mm)},
        {"variant": "share of all lesions, distance in mm", "value": float((mm < 30).sum() / len(d_all)), "n": len(d_all)},
        {"variant": "share of all lesions using EDT seed points instead of centroids", "value": float((np.concatenate([x[2] for x in nn["seed"]]) < 30).mean()), "n": len(d_all)},
        {"variant": "PRIMARY: median equivalent-sphere radius, vox", "value": float(np.median(radius)), "n": len(radius)},
        {"variant": "median half-mean bounding-box extent, vox", "value": float(np.median([(b[1] - b[0] + 1 + b[3] - b[2] + 1 + b[5] - b[4] + 1) / 6 for c in side for b in c["bboxes_zyx"]])), "n": len(radius)},
        *[{"variant": f"median equivalent-sphere radius, cohort {c}, vox", "value": v, "n": int((np.array(coh) == c).sum())} for c, v in per_coh.items()],
    ]
    return rows, variants, {"n_volumes_read": len(ids), "n_lesions": len(vol)}


def pantrack_checks(claims: list, root: Path, workers: int, limit_args) -> tuple[list[dict], list[dict], dict]:
    patients, tracking = json.loads((root / "patients.json").read_text()), json.loads((root / "tracking.json").read_text())
    organs = json.loads((root / "organ_annotations.json").read_text())
    pids = limited(sorted(patients), limit_args)
    scans = [s for p in pids for s in patients[p]]

    with nano_progress(len(scans), "scanning PanTrack labels") as advance, ThreadPoolExecutor(workers) as ex:
        found = []
        for r in ex.map(lambda sc: label_values(root / "labels" / f"{sc}.nii.gz"), scans):
            found.append(r)
            advance()
    scan_rows = []
    for s, inst in zip(scans, found):
        ann = organs[s]
        scan_rows.append({"scan": s, "patient": s.rsplit("_", 1)[0], "n_annotated_lesions": len(ann), "n_label_instances": len(inst), "n_pancreas": sum(v["organ"] == "pancreas" for v in ann.values()),
                          "n_liver": sum(v["organ"] == "liver" for v in ann.values()), "n_lymph_node": sum(v["organ"] == "lymph node" for v in ann.values()),
                          "ids_equal_mod_256": sorted(int(k) % 256 for k in ann) == inst, "max_annotation_id": max((int(k) for k in ann), default=0)})
    pair_rows, n_prop, n_entries = [], 0, 0
    is_nan = lambda y: y is None or (isinstance(y, float) and math.isnan(y))
    is_null = lambda x: is_nan(x) or (isinstance(x, (list, tuple)) and any(map(is_nan, x)))  # vanishing lesions carry NaN (a bare float) instead of a point
    for p in pids:
        for i, step in enumerate(tracking[p]):
            van = [k for k, v in step.items() if is_null(v["fu_point"])]
            n_entries += len(step)
            n_prop += sum(not is_null(v["fu_point_prop"]) for v in step.values())
            pair_rows.append({"patient": p, "pair_index": i, "bl_scan": next(iter(step.values()))["img_bl"], "fu_scan": next(iter(step.values()))["img_fu"], "n_tracked_lesions": len(step), "n_vanishing": len(van), "vanishing_ids": van})
    tot = lambda k: sum(r[k] for r in scan_rows)
    src = "PanTrack patients/tracking/organ_annotations.json"
    claim(claims, "PanTrack patients", 45, len(pids), src)
    claim(claims, "PanTrack CT scans", 161, len(scans), "patients.json")
    claim(claims, "PanTrack label files", 161, len(list((root / "labels").glob("*.nii.gz"))), "labels/ listing", note="counted over the whole folder even in a smoke run")
    claim(claims, "PanTrack annotated lesion instances", 292, tot("n_annotated_lesions"), "organ_annotations.json entries")
    claim(claims, "PanTrack annotated lesion instances (label NIfTIs)", 292, tot("n_label_instances"), "labels/*.nii.gz distinct non-zero values per scan", note="labels are uint8: annotation ids above 255 wrap (307 -> 51)")
    claim(claims, "PanTrack pancreas instances", 165, tot("n_pancreas"), "organ_annotations.json", stated_in="dataset README (via plan)")
    claim(claims, "PanTrack liver instances", 124, tot("n_liver"), "organ_annotations.json", stated_in="dataset README (via plan)")
    claim(claims, "PanTrack lymph-node instances", 3, tot("n_lymph_node"), "organ_annotations.json", stated_in="dataset README (via plan)")
    claim(claims, "PanTrack consecutive scan pairs", 116, len(pair_rows), "tracking.json")
    claim(claims, "PanTrack vanishing lesion pairs", 36, sum(r["n_vanishing"] for r in pair_rows), "tracking.json fu_point null", stated_in="dataset README (via plan)")
    claim(claims, "PanTrack propagated point for every lesion (count = tracked lesion entries)", n_entries, n_prop, "tracking.json fu_point_prop")
    return scan_rows, pair_rows, {"scans_with_label_ids_not_equal_mod_256": sum(not r["ids_equal_mod_256"] for r in scan_rows), "max_annotation_id": max(r["max_annotation_id"] for r in scan_rows)}


def label_values(path: Path) -> list[int]:
    img = sitk.ReadImage(str(path))  # held in a variable: an inline view of a temporary image is a use-after-free
    a = sitk.GetArrayFromImage(img)
    return sorted(int(v) for v in np.unique(a[a > 0]))


def special_patient(pid: str, root: Path, metas: dict, per: list[dict], table: dict, audit: list[dict]) -> list[dict]:
    df, r = metas[pid], next(x for x in per if x["patient"] == pid)
    files = sorted(f"{f}/{p.name}" for f in ("inputsTrBL", "inputsTrFU", "targetsTrBL", "targetsTrFU") for p in (root / f).glob(f"{pid}_*") if p.name.endswith((".json", ".nii.gz")))
    clickfix = pd.read_csv(root / "derivatives/unigrad-icon-registration/clickfix_report.csv")
    facts = {"rows_in_meta": len(df), "img_id_bl_values": sorted(set(map(int, df["img_id_bl"]))), "img_id_fu_values": sorted(set(map(int, df["img_id_fu"]))), "classes_pair": topo_counts(dominant_pair(df)),
             "n_unclear": r["n_unclear_pair"], "official_split": r["official_split"], "in_holdout60": r["in_holdout60"], "graph_split": r["graph_split"], "graphs_in_cache": r["n_graphs_cache"],
             "merge_group_sizes": r["merge_group_sizes"], "files_on_disk": files, "click_audit_rows": [a for a in audit if a["patient"] == pid],
             "in_registration_table_case_failures": {b: pid in table["excluded"][b]["case_level_failure_patients"] for b in table["excluded"]},
             "clickfix_report": clickfix[clickfix["case"].str.startswith(pid)].to_dict("records")}
    return [{"fact": k, "value": v} for k, v in facts.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, gpu=False)
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="Longitudinal-CT root (meta/, inputsTr*, data_split.json, derivatives/)")
    ap.add_argument("--holdout-csv", type=Path, default=HOLDOUT_CSV, help="CSV listing the held-out 60 patients (column `patient`)")
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR, help="preprocessed merged corpus (cohorts.json, dataset.json, splits_final.json, nnUNetPlans_3d_fullres/*_centroids.json)")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="NanoUNet_raw, only listed to count the cohort folders Dataset011..031")
    ap.add_argument("--pantrack-dir", type=Path, default=PANTRACK_DIR, help="PanTrack root (patients.json, tracking.json, organ_annotations.json, labels/)")
    ap.add_argument("--graph-split", type=Path, default=GRAPH_SPLIT, help="lesionglue split.json (train 192 / val 48 / test 60) the graph caches were built from")
    ap.add_argument("--graph-cache-dir", type=Path, default=GRAPH_CACHE_DIR, help="graph cache `processed/` dir; only the tiny <split>_<tag>_meta.pt files are opened, never the big caches")
    ap.add_argument("--graph-cache-tag", default=GRAPH_CACHE_TAG, help="cache tag in the file names (train_<tag>_meta.pt)")
    ap.add_argument("--workers", type=int, default=8, help="threads for the JSON sidecar reads and the PanTrack label scans")
    args = ap.parse_args()
    meta_pt = {s: args.graph_cache_dir / f"{s}_{args.graph_cache_tag}_meta.pt" for s in ("train", "val", "test")}
    paths = {"Longitudinal-CT meta dir": args.data_root / "meta", "released split": args.data_root / "data_split.json", "holdout csv": args.holdout_csv, "corpus cohorts.json": args.corpus_dir / "cohorts.json",
             "corpus dataset.json": args.corpus_dir / "dataset.json", "corpus splits_final.json": args.corpus_dir / "splits_final.json", "corpus plans": args.corpus_dir / "nnUNetResEncUNetLPlans_h200_smallpv.json",
             "corpus sidecars": args.corpus_dir / "nnUNetPlans_3d_fullres", "NanoUNet_raw": args.raw_dir, "PanTrack patients.json": args.pantrack_dir / "patients.json", "PanTrack tracking.json": args.pantrack_dir / "tracking.json",
             "PanTrack organ_annotations.json": args.pantrack_dir / "organ_annotations.json", "PanTrack labels": args.pantrack_dir / "labels", "graph split": args.graph_split,
             "registration error table": args.data_root / "derivatives/registration_error_table.json", "clickfix report": args.data_root / "derivatives/unigrad-icon-registration/clickfix_report.csv",
             **{f"graph cache meta ({s})": p for s, p in meta_pt.items()}}
    problems = missing_paths(paths, "mount /nnunet_data, or pass the matching --data-root/--corpus-dir/--pantrack-dir/--graph-split/--graph-cache-dir flag")
    holdout = {ln.strip() for ln in args.holdout_csv.read_text().splitlines()[1:] if ln.strip()} if args.holdout_csv.is_file() else set()
    if not problems:
        pids = sorted(p.stem for p in (args.data_root / "meta").glob("*.csv"))
        unknown = sorted(holdout - set(pids))
        if unknown:
            problems.append(problem(f"{len(unknown)} held-out patient(s) have no meta CSV: {unknown[:5]}", "every id in --holdout-csv to have meta/<id>.csv under --data-root", "pass the matching --holdout-csv and --data-root"))
    abort_if(problems)
    run = start_run(EXP, ap, args, inputs={"holdout csv": args.holdout_csv, "released split": args.data_root / "data_split.json", "meta dir": args.data_root / "meta", "cohorts.json": args.corpus_dir / "cohorts.json",
                                            "dataset.json": args.corpus_dir / "dataset.json", "splits_final.json": args.corpus_dir / "splits_final.json", "graph split": args.graph_split,
                                            "PanTrack tracking.json": args.pantrack_dir / "tracking.json", "registration error table": args.data_root / "derivatives/registration_error_table.json", **{f"graph cache meta ({s})": p for s, p in meta_pt.items()}},
                    paper={"section": "Experimental design > Data and evaluation tiers; Table datasets", "supports": "every data count the paper states"})
    official_raw = json.loads((args.data_root / "data_split.json").read_text())
    official = {p: s for s, ps in official_raw.items() for p in ps}
    graph_split = json.loads(args.graph_split.read_text())
    all_pids = sorted(p.stem for p in (args.data_root / "meta").glob("*.csv"))
    pids = limited(all_pids, args)
    metas, no_col = load_meta(args.data_root, pids)
    per, events, graphs = patient_rows(metas, official, holdout, graph_split)
    claims: list[dict] = []
    src = "Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept"
    pair = pd.concat([dominant_pair(df) for df in metas.values()])
    hold_pair = pd.concat([dominant_pair(df) for p, df in metas.items() if p in holdout]) if holdout & set(pids) else pair.iloc[:0]
    tc, htc = topo_counts(pair), topo_counts(hold_pair)
    all_events, hold_events = [e for e in events], [e for e in events if e["in_holdout60"]]
    sizes = Counter(e["group_size"] for e in all_events)
    claim(claims, "Longitudinal-CT patients", 300, len(pids), "meta/*.csv")
    claim(claims, "linked lesions, one scan pair per patient", 4530, len(pair), src)
    for k, n in zip(CLASSES, (2424, 1382, 558, 166)):
        claim(claims, f"{k.lower()} lesions", n, tc[k], src)
    claim(claims, "topology labels per lesion (distinct values)", 4, len(set(pair["topology_class"])), src)
    claim(claims, "merge events", 38, len(all_events), src + ", groups of MERGING rows by merged_into")
    claim(claims, "merge events of two lesions into one", 24, sizes[2], src)
    claim(claims, "split labels (SPLIT rows)", 0, int(sum((df["topology_class"] == "SPLIT").sum() for df in metas.values())), "meta/*.csv topology_class (all rows)", note="load_meta asserts every topology is one of the four classes, so this is 0 by construction unless it aborts")
    claim(claims, "held-out patients", 60, len(holdout), "test_patients.csv")
    claim(claims, "held-out = released val (30) + test (30)", 60, sum(1 for p in holdout if official.get(p) in ("val", "test")), "data_split.json vs test_patients.csv")
    claim(claims, "released split: validation patients", 30, sum(1 for s in official.values() if s == "val"), "data_split.json")
    claim(claims, "released split: test patients", 30, sum(1 for s in official.values() if s == "test"), "data_split.json")
    claim(claims, "remaining training pool patients", 240, sum(1 for s in official.values() if s == "train"), "data_split.json")
    claim(claims, "held-out lesions", 774, len(hold_pair), src + ", held-out 60")
    for k, n in zip(CLASSES, (475, 213, 67, 19)):
        claim(claims, f"held-out {k.lower()} lesions", n, htc[k], src + ", held-out 60")
    claim(claims, "held-out merge events", 5, len(hold_events), src + ", held-out 60")
    claim(claims, "held-out merge events are in five patients", 5, len({e["patient"] for e in hold_events}), src + ", held-out 60")
    claim(claims, "held-out largest merge: nine lesions into one", 9, max((e["group_size"] for e in hold_events), default=0), src + ", held-out 60")
    rec = reconciliation(per)
    graph_rows, missing = graph_check(per, graphs, graph_split, read_cache_meta(meta_pt))
    empty = [{"patient": r["patient"], "n_bl": r["n_bl_pair"], "n_fu": r["n_fu_pair"], "n_bl_unclear_excluded": r["n_bl_pair_clear"], "n_fu_unclear_excluded": r["n_fu_pair_clear"], "n_unclear": r["n_unclear_pair"], "in_holdout60": r["in_holdout60"],
             "empty_under_paper_rule": r["n_bl_pair"] == 0 or r["n_fu_pair"] == 0, "empty_after_dropping_unclear": r["n_bl_pair_clear"] == 0 or r["n_fu_pair_clear"] == 0}
            for r in per if r["n_bl_pair"] == 0 or r["n_fu_pair"] == 0 or r["n_bl_pair_clear"] == 0 or r["n_fu_pair_clear"] == 0]
    audit = click_audit(args.data_root, metas)
    table = json.loads((args.data_root / "derivatives/registration_error_table.json").read_text())
    special = special_patient(KNOWN_BAD, args.data_root, metas, per, table, audit) if KNOWN_BAD in metas else []
    cohort_rows, variants, side = corpus_checks(claims, args.corpus_dir, holdout, official, args.workers, args, args.raw_dir)
    scan_rows, pair_rows, pan = pantrack_checks(claims, args.pantrack_dir, args.workers, args)
    no_prop = [{"patient": p, "lesion_id": int(r.lesion_id), "topology_class": r.topology_class, "img_id_fu": int(r.img_id_fu)} for p, df in metas.items() for r in df[~df["linking_unclear"]].itertuples()
               if r.topology_class in BL_TOPO and (pd.isna(r.cog_propagated) or not str(r.cog_propagated).strip())]  # BL nodes need cog_propagated; the builder drops these lesions
    bad = [c for c in claims if not c["match"]]
    tbl = Table(title=f"claims that do NOT match ({len(bad)} of {len(claims)})", box=None, padding=(0, 2))
    for col in ("claim", "paper", "measured"):
        tbl.add_column(col)
    for c in bad:
        tbl.add_row(c["claim"], str(c["paper_value"]), str(c["measured_value"]))
    console().print(tbl)
    pr = {(r["scope"], r["view"]): r for r in rec}
    notes = [f"{len(bad)} of {len(claims)} claims do not match: " + "; ".join(f"{c['claim']} (paper {c['paper_value']}, measured {c['measured_value']})" for c in bad),
             f"CSVs without a linking_unclear column (treated as False, as parse_meta_csv does): {no_col}",
             "Pair rule: one pair per patient = rows of the most frequent img_id_fu, linking_unclear kept (reproduces the paper). The graph cache instead drops unclear rows and keeps every FU region.",
             "Graph-cache membership replayed from the CSVs; replayed cross edges and positives equal the cache's *_meta.pt totals per split (see graph_cache.matches_cache_meta).",
             "Scoring on the graph cache must use only the dominant-FU-region graph per patient: per_patient.dominant_graph_present is False where the dominant region has no graph although another region has one.",
             f"`{KNOWN_BAD}` facts are in special_patients; the plan's description (no inputsTrFU/{KNOWN_BAD}_01.json, corrupted BL _00 clicks) is checked there, not assumed.",
             f"PanTrack labels are stored as uint8 although the README says uint16: scans whose annotation ids are not equal to the label values modulo 256: {pan['scans_with_label_ids_not_equal_mod_256']}; largest annotation id {pan['max_annotation_id']}.",
             f"Sidecar reads: {side['n_volumes_read']} volumes, {side['n_lesions']} lesions."]
    summary = {"claims": {"n": len(claims), "n_match": len(claims) - len(bad), "n_mismatch": len(bad), "mismatches": [c["claim"] for c in bad]},
               "linking_unclear": {"paper_rule_all300": {k: pr[("all300", "paper_rule_unclear_kept")][k] for k in ("n_lesions", *CLASSES)}, "unclear_excluded_all300": {k: pr[("all300", "unclear_excluded")][k] for k in ("n_lesions", *CLASSES)},
                                   "unclear_rows_in_pairs": pr[("all300", "unclear_only")]["n_lesions"], "unclear_rows_all_regions": int(sum(r["n_unclear_all_regions"] for r in per)),
                                   "paper_rule_holdout60": {k: pr[("holdout60", "paper_rule_unclear_kept")][k] for k in ("n_lesions", *CLASSES)}, "unclear_excluded_holdout60": {k: pr[("holdout60", "unclear_excluded")][k] for k in ("n_lesions", *CLASSES)},
                                   "rows_all_regions": int(sum(r["n_rows_all_regions"] for r in per)), "patients_with_several_fu_regions": int(sum(r["n_fu_regions"] > 1 for r in per))},
               "graph_cache": {r["split"]: {"patients": r["n_patients_in_split"], "with_graph": r["n_patients_with_graph"], "graphs": r["n_graphs"], "matches_cache_meta": r["matches_cache_meta"]} for r in graph_rows},
               "patients_missing_from_graph_cache": [m["patient"] for m in missing], "patients_empty_bl_or_fu_paper_rule": [e["patient"] for e in empty if e["empty_under_paper_rule"]],
               "merge_group_size_histogram": dict(sorted(sizes.items())), "click_audit_rows": len(audit),
               "graph_builder_gaps": {"lesions_without_cog_propagated_after_unclear_filter": dict(Counter(r["topology_class"] for r in no_prop)), "n_lesions_without_cog_propagated": len(no_prop),
                                      "merge_rows_whose_target_has_no_fu_node": int(sum(g["merge_rows_without_fu_node"] for gs in graphs.values() for g in gs))}}
    md = "# exp00a data audit\n\n## Claims (paper value vs measured)\n\n" + md_table([{**c, "match": "OK" if c["match"] else "MISMATCH"} for c in claims], ["claim", "paper_value", "measured_value", "match", "source"]) \
        + "\n## linking_unclear reconciliation (paper rule = unclear kept)\n\n" + md_table(rec, ["scope", "view", "n_patients", "n_lesions", *CLASSES]) \
        + "\n## Graph cache replay vs cache metadata\n\n" + md_table(graph_rows, list(graph_rows[0])) + "\n## Patients with no graph in the cache\n\n" + md_table(missing, list(missing[0]) if missing else ["patient"])
    tables = {"claims": claims, "per_patient": per, "merge_events": events, "unclear_reconciliation": rec, "graph_cache": graph_rows, "graph_missing": missing, "empty_sets": empty, "click_audit": audit,
              "no_cog_propagated": no_prop, "special_patients": special, "cohorts": cohort_rows, "prompt_variants": variants, "pantrack_scans": scan_rows, "pantrack_pairs": pair_rows}
    run.finish(summary, tables, table_md=md, notes=notes, next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
