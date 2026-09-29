"""Pool per-fold, per-selector val_per_patient.json (from oof.py) into one out-of-fold estimate
with a patient bootstrap CI.

Layout expected: {runs}/fold_*/oof_<selector>/val_per_patient.json, one file per
(fold, selector) pair, written by lesionglue/cli/oof.py. Each patient sits in the val side of
exactly one CV fold, so concatenating per-patient records across all folds gives one properly
out-of-fold score per patient over the whole pool, per selector — see
round12_measurement_features_data.md §3 (Stage B.2). --stratify-registration reruns the same
pooled estimate on the flagged/clean split, which IS the reformulated Gate A (round12_findings.md
§A.3.4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lesionglue.common import DATASET_ROOT, dump_json, load_json
from lesionglue.data.source.provenance import CLICKFIX_REL
from lesionglue.eval.bootstrap import bootstrap_match_score

REG_TABLE_REL = "derivatives/registration_error_table.json"
SUB_NAMES = ("uc", "dis", "new")


def flagged_patients(root: Path) -> set[str]:
    reg = load_json(root / REG_TABLE_REL)
    flagged = set(reg["excluded"]["original"]["case_level_failure_patients"])
    flagged |= set(reg["excluded"]["unigradicon"]["case_level_failure_patients"])
    cf = pd.read_csv(root / CLICKFIX_REL)
    bad = cf[(cf["status"] != "ok") | (cf["n_sanity_bad"] > 0)]
    flagged |= {str(c).rsplit("_", 1)[0] for c in bad["case"]}
    return flagged


def report(label: str, per_patient: dict) -> dict:
    if not per_patient:
        print(f"{label}: n_patients=0, skipping")
        return {"n_patients": 0}
    point, lo, hi = bootstrap_match_score(per_patient)
    row = {"n_patients": len(per_patient), "match_score": point, "ci95": [lo, hi]}
    print(f"{label}: n_patients={len(per_patient)}  match_score={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    for name in SUB_NAMES:
        ok = sum(p[f"{name}_ok"] for p in per_patient.values())
        tot = sum(p[f"{name}_tot"] for p in per_patient.values())
        acc = ok / tot if tot else float("nan")
        row[f"acc_{name}"] = acc
        print(f"    {name}: {ok}/{tot} = {acc:.4f}")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir containing fold_*/oof_<selector>/val_per_patient.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(DATASET_ROOT), help="dataset root, for --stratify-registration")
    ap.add_argument("--stratify-registration", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.runs).glob("fold_*/oof_*/val_per_patient.json"))
    if not files:
        raise SystemExit(f"no fold_*/oof_*/val_per_patient.json under {args.runs}")

    by_selector: dict[str, dict[str, dict]] = {}
    seen_by_selector: dict[str, dict[str, Path]] = {}
    for f in files:
        raw = load_json(f)
        selector = f.parent.name.removeprefix("oof_")
        per_patient = raw["per_patient"]
        pool = by_selector.setdefault(selector, {})
        seen = seen_by_selector.setdefault(selector, {})
        for pid, counts in per_patient.items():
            # Leakage check: a patient must appear in exactly one fold's val side. If this fires,
            # the CV fold assignment leaked a patient across folds and the OOF estimate is invalid.
            assert pid not in seen, (
                f"patient {pid!r} appears in both {seen[pid]} and {f} for selector {selector!r} "
                "-- CV fold leakage, pooled estimate is invalid"
            )
            seen[pid] = f
            pool[pid] = counts

    print(f"pooled {len(files)} files -> selectors: {sorted(by_selector)}")
    summary: dict[str, dict] = {}
    for selector, pool in sorted(by_selector.items()):
        print(f"\n=== selector: {selector} ===")
        summary[selector] = {"all": report("all", pool)}

    if args.stratify_registration:
        flagged = flagged_patients(Path(args.root))
        for selector, pool in sorted(by_selector.items()):
            print(f"\n=== selector: {selector}, stratified by registration quality ===")
            flagged_pool = {p: c for p, c in pool.items() if p in flagged}
            clean_pool = {p: c for p, c in pool.items() if p not in flagged}
            summary[selector]["flagged"] = report("flagged", flagged_pool)
            summary[selector]["clean"] = report("clean", clean_pool)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "pool_summary.json", summary)
