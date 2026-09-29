"""Holdout seg + track metrics. Not wired into nanounet_predict / lesionglue_track.

Seg: volume Dice + per-lesion DSC / NSD / LDR (IoU>0.1) via nanounet.score.
Track: same weighted match_score as training (0.5 unchanged, 0.25 disappeared,
0.25 new) on dense pairs vs meta. Coupling: topology accuracy split by whether
each endpoint was a click-on-FG instance with LDR hit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from nanounet.score import IOU_HIT, _agg, score_case, write
from lesionglue.bootstrap import bootstrap_match_score
from lesionglue.data.graph import _node_rows, _positive_matrix
from lesionglue.data.meta import parse_meta_csv, resolve_track_case
from lesionglue.data.splits import load_tracking_split


def _stem(p: Path) -> str:
    n = p.name
    return n[: -len(".nii.gz")] if n.endswith(".nii.gz") else p.stem


def _img_i(p: Path) -> int:
    return int(_stem(p).split("_")[-1])


def _nodes(path: Path) -> tuple[set[int], set[int]]:
    if not path.is_file():
        return set(), set()
    o = json.loads(path.read_text())
    return set(map(int, o["bl_ids"])), set(map(int, o["fu_ids"]))


def _score_folder(pred_dir: Path, gt_dir: Path, click_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(pred_dir.glob("*.nii.gz")):
        cid = p.name[: -len(".nii.gz")]
        jp = click_dir / f"{cid}.json"
        gt = gt_dir / p.name
        if not jp.is_file() or not gt.is_file():
            raise FileNotFoundError(
                f"Missing GT or clicks for {cid}.\nExpected {gt} and {jp}.\n"
                f"Fix: --gt-dir targetsTrXX and click dir inputsTrXX"
            )
        rows.append(score_case(cid, str(p), str(gt), str(jp)))
    return rows


def _ldr_map(row: dict) -> dict[int, float]:
    return {int(L["id"]): float(L["ldr"]) for L in row["lesions"]}


def _pairs(path: Path) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    with path.open() as f:
        for r in csv.DictReader(f):
            out[int(r["bl_lesion_id"])].add(int(r["fu_lesion_id"]))
    return out


def _counts(bl_ids, fu_ids, Y, pred: dict[int, set[int]], bl_nodes: set[int], fu_nodes: set[int]) -> dict:
    claimed = {f for s in pred.values() for f in s}
    uc_ok = uc_tot = dis_ok = dis_tot = new_ok = new_tot = 0
    for i, bid in enumerate(bl_ids):
        pos = {int(fu_ids[j]) for j in range(len(fu_ids)) if float(Y[i, j]) > 0.5}
        got = pred.get(int(bid), set())
        if pos:
            uc_tot += 1
            uc_ok += int(int(bid) in bl_nodes and pos <= fu_nodes and pos == got)
        else:
            dis_tot += 1
            dis_ok += int(int(bid) in bl_nodes and not got)
    for j, fid in enumerate(fu_ids):
        if float(Y[:, j].sum()) > 0.5:
            continue
        new_tot += 1
        new_ok += int(int(fid) in fu_nodes and int(fid) not in claimed)
    return dict(uc_ok=uc_ok, uc_tot=uc_tot, dis_ok=dis_ok, dis_tot=dis_tot, new_ok=new_ok, new_tot=new_tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/nnunet_data/Longitudinal-CT")
    ap.add_argument("--preds-bl", required=True)
    ap.add_argument("--preds-fu", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--oracle-matches", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fu = _score_folder(Path(args.preds_fu), root / "targetsTrFU", root / "inputsTrFU")
    bl = _score_folder(Path(args.preds_bl), root / "targetsTrBL", root / "inputsTrBL")
    write(fu, str(out / "seg_fu"))
    write(bl, str(out / "seg_bl"))
    fu_by, bl_by = {r["case_id"]: r for r in fu}, {r["case_id"]: r for r in bl}
    pids = list(map(str, load_tracking_split()["test"]))
    per, coup = {}, defaultdict(lambda: {"ok": 0, "tot": 0})
    for pid in pids:
        case = resolve_track_case(root, pid)
        mp = Path(args.matches) / f"{pid}.csv"
        if case is None or not mp.is_file():
            continue
        rows = [r for r in parse_meta_csv(case.propagated) if r.img_id_fu == _img_i(case.fu_img)]
        bl_rep, fu_rep = _node_rows(rows, pid)
        bl_ids, fu_ids = sorted(bl_rep), sorted(fu_rep)
        if not bl_ids or not fu_ids:
            continue
        Y = _positive_matrix(rows, {lid: i for i, lid in enumerate(bl_ids)}, {lid: j for j, lid in enumerate(fu_ids)})
        pred = _pairs(mp)
        bn, fn = _nodes(mp.with_suffix(".json"))
        per[pid] = _counts(bl_ids, fu_ids, Y, pred, bn, fn)
        sbl, sfu = bl_by.get(_stem(case.bl_img)), fu_by.get(_stem(case.fu_img))
        if sbl is None or sfu is None:
            continue
        lb, lf = _ldr_map(sbl), _ldr_map(sfu)
        for i, bid in enumerate(bl_ids):
            pos = {int(fu_ids[j]) for j in range(len(fu_ids)) if float(Y[i, j]) > 0.5}
            got = pred.get(int(bid), set())
            ok = (int(bid) in bn and pos <= fn and pos == got) if pos else (int(bid) in bn and not got)
            hb, hf = lb.get(int(bid), 0.0) >= 0.5, (min((lf.get(f, 0.0) for f in pos), default=0.0) >= 0.5) if pos else True
            key = "both_hit" if hb and hf else ("bl_miss" if not hb else "fu_miss")
            coup[key]["tot"] += 1
            coup[key]["ok"] += int(ok)
            coup["all"]["tot"] += 1
            coup["all"]["ok"] += int(ok)
    point, lo, hi = bootstrap_match_score(per) if per else (0.0, 0.0, 0.0)
    oracle = None
    if args.oracle_matches:
        oper = {}
        for pid in per:
            op = Path(args.oracle_matches) / f"{pid}.csv"
            if not op.is_file():
                continue
            case = resolve_track_case(root, pid)
            rows = [r for r in parse_meta_csv(case.propagated) if r.img_id_fu == _img_i(case.fu_img)]
            bl_rep, fu_rep = _node_rows(rows, pid)
            bl_ids, fu_ids = sorted(bl_rep), sorted(fu_rep)
            Y = _positive_matrix(rows, {lid: i for i, lid in enumerate(bl_ids)}, {lid: j for j, lid in enumerate(fu_ids)})
            bn, fn = _nodes(op.with_suffix(".json"))
            oper[pid] = _counts(bl_ids, fu_ids, Y, _pairs(op), bn, fn)
        if oper:
            op, olo, ohi = bootstrap_match_score(oper)
            oracle = {"n": len(oper), "match_score": op, "ci95": [olo, ohi], "pooled": {k: sum(oper[p][k] for p in oper) for k in next(iter(oper.values()))}}
    pooled = {k: sum(per[p][k] for p in per) for k in next(iter(per.values()))} if per else {}
    summary = {
        "protocol": {"ldr_iou": IOU_HIT, "match_score": "0.5 unchanged-set-equality + 0.25 disappeared + 0.25 new", "decode": "dense"},
        "seg_fu": _agg(fu), "seg_bl": _agg(bl),
        "track_e2e": {"n": len(per), "match_score": point, "ci95": [lo, hi], "pooled": pooled},
        "track_oracle_dense": oracle,
        "coupling": {k: {**v, "acc": (v["ok"] / v["tot"] if v["tot"] else None)} for k, v in coup.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
