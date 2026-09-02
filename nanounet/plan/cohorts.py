"""Derive sampling cohort weights from the merged dataset itself, rule "site_balanced".

No human hand-authors cohorts.json: each source dataset may declare "lesion_site" in its raw
dataset.json (absent -> the dataset is its own site, named by its bare prefix e.g. "d029").
Cohorts are grouped by site; the 1.0 probability mass is split EQUALLY across sites, then split
again within each site proportional to case count. Recomputed fresh every preprocess run from
merged_sources.json (written by plan/prep/merge.py) and the source datasets' own dataset.json."""

from __future__ import annotations

from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json

from nanounet.common import raw_dir
from nanounet.plan.dataset_id import convert_id_to_dataset_name

RULE = "site_balanced"


def _source_rows(dataset_id: int, raw_name: str) -> list[dict]:
    """[{prefix, cases, site}], one row per source dataset feeding this pool.

    A single (non-merged) dataset has no merged_sources.json -- it is treated as its own lone
    source, cohort and site, matching the degenerate n_sites=1 case of the rule below."""
    ms_path = join(raw_dir(), raw_name, "merged_sources.json")
    if isfile(ms_path):
        rows = []
        for s in load_json(ms_path)["sources"]:
            src_dj = load_json(join(raw_dir(), s["name"], "dataset.json"))
            prefix = s["prefix"].rstrip("_")
            site = src_dj.get("lesion_site") or prefix
            rows.append({"prefix": prefix, "cases": int(s["num_cases"]), "site": site})
        return rows
    dj = load_json(join(raw_dir(), raw_name, "dataset.json"))
    prefix = f"d{dataset_id:03d}"
    site = dj.get("lesion_site") or prefix
    return [{"prefix": prefix, "cases": int(dj["numTraining"]), "site": site}]


def site_balanced_weights(rows: list[dict]) -> tuple[dict[str, list[str]], dict[str, float]]:
    sites: dict[str, list[str]] = {}
    for r in rows:
        sites.setdefault(r["site"], []).append(r["prefix"])
    cases_by_prefix = {r["prefix"]: r["cases"] for r in rows}
    per_site_mass = 1.0 / len(sites)
    weights: dict[str, float] = {}
    for prefixes in sites.values():
        site_total = sum(cases_by_prefix[p] for p in prefixes)
        for p in prefixes:
            weights[p] = per_site_mass * cases_by_prefix[p] / site_total
    return sites, weights


def run_cohorts(dataset_id: int, out_dir: str) -> str:
    """Write cohorts.json into `out_dir` (the preprocessed dataset dir). Returns the path."""
    raw_name = convert_id_to_dataset_name(dataset_id)
    rows = _source_rows(dataset_id, raw_name)
    sites, weights = site_balanced_weights(rows)
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, f"cohort weights sum to {total}, expected 1.0 (internal bug)"
    doc = {
        "rule": RULE,
        "sites": {site: sorted(prefixes) for site, prefixes in sorted(sites.items())},
        "weights": {p: round(w, 6) for p, w in sorted(weights.items())},
        "counts": {r["prefix"]: {"cases": r["cases"], "site": r["site"]} for r in rows},
    }
    out_path = join(out_dir, "cohorts.json")
    save_json(doc, out_path, sort_keys=False)
    return out_path
