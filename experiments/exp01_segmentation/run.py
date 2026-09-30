"""exp01 - Segmentation, four prompt scenarios and prompt drop  (paper: Experiments > Segmentation; Table 'nine experiments' row 1)

QUESTION   Does the segmenter answer the point rather than the image, in the four situations deployment produces: every lesion clicked (S1), only some
           clicked (S2), nothing clicked (S3) and a click on empty tissue (S4)? And how much of its Dice does the prompt itself carry (prompt drop)?
WHY        Fills the segmentation row of the paper: per-lesion Dice, NSD@1 mm and detection with patient-level CIs, the selectivity margin of S2, the
           quietness of S3/S4, and the paired S1 minus prompt-zeroed drop; the same scenario click files feed ULS+ and nnInteractive.
DATA       Cases of the eval manifest (experiments/exp00c_seg_eval_manifest/seg_eval_v1.json, one scan per case, tiers seen-cohort / outside / healthy),
           filtered by --tier, --sources, --max-cases-per-source, --max-lesions-per-case and the overlap policy. Lesions = cc3d-26 components of the
           binary GT. Partial / pseudo / spheres annotations support S1 (+ its prompt-drop twin) only, spheres and pseudo score detection only; S2
           needs a fully annotated case with >= 2 lesions; S3/S4 run on healthy scans and fully annotated cases.
METHOD     1. Per case, once: GT instances in the native grid; a seed per lesion (argmax EDT, inside the lesion); the S2 subset (strict, seeded), the S4
              decoy (tissue, >= 5 mm from every lesion). Cached as artifacts/lesions/<case>.json.
           2. Clicks = seed + one draw of the empirical registration-error table (size-bin matched, resampled voxels -> mm -> native voxels), i.e. the
              propagated-style noisy clicks of deployment; written as click JSONs to artifacts/prompts/<scenario>/<case>.json (S3 = no click).
           3. nanoUNet, whole volume, EMA weights, clustered tiles near the clicks, one preprocessing per scan: S1, S1 with the prompt channels zeroed
              (prompt drop), S2, S4. S3 is a tile with no click in it: the S1 tiles with zeroed prompt on scans with lesions (a call with an empty
              click list does nothing), the decoy tile with zeroed prompt on lesion-free scans. Masks go to artifacts/preds/nanounet/<scenario>/.
           4. External systems (ULS+, nnInteractive) are not run here: their masks for the same click files are read from --external NAME=DIR
              (layout in external_preds.py) and scored by the same code. --emit-prompts-only stops after step 2.
           5. Score per lesion (Dice, NSD@1 mm, hit = IoU > 0.1 with the connectivity-18 prediction component of largest overlap), per case for S3/S4
              (predicted foreground voxels, any-FG rate, split inside/outside GT) and for S2 (foreground Dice vs the clicked lesions and vs all lesions,
              their difference = selectivity margin, leak = hit rate of the unclicked lesions). Case-mean, then mean over cases, patient bootstrap CIs.
OUTPUT     results.json tables: per_lesion (method, tier, cohort, organ, case, lesion_id, size_mm, size_bin, scenario, dice, nsd, hit, click offset ...),
           per_case, summary_table (every CI incl. by cohort / organ / size bin), cases. artifacts/: lesions, prompts, preds. Run dir in the log.
COMMAND    python -m experiments.exp01_segmentation.run --tag paper_v1
DEPENDS ON experiments/common.py, experiments/segment.py, experiments/scoring.py (bootstrap_stat, paired_delta_stat), the exp00c manifest; exp02 reuses
           the same clicks (same --seed, --backends, --max-lesions-per-case) for its s=1 replicate 0 and cross-checks against this run's S1.
RUNTIME    About 20-40 s per case on one A100 (4 passes), i.e. 1-2 h at the default caps; the external systems take their own time. Resumable per pass.
CAVEATS    S3 for nanoUNet is a tile with no click, for the external systems it is an empty click list (they emit nothing without a click), so S3 is only
           comparable in spirit. Decoys sit on voxels above -500 HU, so the CT couch can be hit rarely. --max-lesions-per-case makes S1 partly selective.
           Overlap policy `common` (default) scores every method on the cases clean for all three systems; `own` on each method's own clean set.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from core.ui import cprint, nano_progress
from experiments import segment as S
from experiments.common import REPO, SEG_CKPT, SEG_EMA, abort_if, add_common_args, limited, missing_paths, problem, start_run
from experiments.exp01_segmentation.external_preds import EXTERNAL_SCENARIOS, external_path, inventory_problems, parse_external
from experiments.scoring import bootstrap_stat, paired_delta_stat
from nanounet.data.patch.error_table import DEFAULT_ERROR_TABLE, load_table
from nanounet.prompt.coords import load_points_xyz
from nanounet.score import IOU_HIT, NSD_TOL_MM

EXP = "exp01_segmentation"
PAPER = {"section": "Experiments > Segmentation", "table_row": 1, "supports": "segmentation row: per-lesion Dice/NSD/detection, selectivity (S2), quietness (S3, S4), prompt drop"}
DEFAULT_MANIFEST = REPO / "experiments" / "exp00c_seg_eval_manifest" / "seg_eval_v1.json"
ALL_SYSTEMS = ["nanounet", "nninteractive", "uls_plus"]
PROMPT_OF = {"S1": "S1", "S1_noprompt": "S1", "S2": "S2", "S3": "S4", "S4": "S4"}  # click file a scenario's inference reads (S3 tile sits at the decoy on lesion-free scans)
LESION_SCENARIOS = ("S1", "S1_noprompt", "S2")


def mean(xs: list) -> float:
    return float(np.mean(xs))


def ci(items: list[dict], key: str, stat=mean) -> list[float] | None:
    """[point, lo, hi] of stat over the items' `key`, patient-level bootstrap; None if no item has the key."""
    by = by_patient(items, key)
    return [float(v) for v in bootstrap_stat(by, stat)] if by else None


def by_patient(items: list[dict], key: str) -> dict[str, list[float]]:
    out = defaultdict(list)
    for r in items:
        if r.get(key) is not None:
            out[r["patient"]].append(r[key])
    return dict(out)


def lesion_file(art: Path, cid: str) -> Path:
    return art / "lesions" / f"{cid}.json"


def prompt_file(art: Path, scenario: str, cid: str) -> Path:
    return art / "prompts" / scenario / f"{cid}.json"


def pred_file(art: Path, ext_dirs: dict[str, Path], method: str, scenario: str, cid: str, n_used: int) -> Path:
    """Where a method's mask for (scenario, case) lives; nanoUNet's S3 on a scan with lesions is its S1_noprompt pass."""
    if method != "nanounet":
        return external_path(ext_dirs[method], scenario, cid)
    return art / "preds" / "nanounet" / ("S1_noprompt" if scenario == "S3" and n_used > 0 else scenario) / f"{cid}.nii.gz"


def scenarios(case: dict, cache: dict, method: str) -> tuple[str, ...]:
    """Scenarios a method is scored on for this case (the prompt-drop twin exists for nanoUNet only)."""
    return tuple(s for s in S.scenarios_for(case, len(cache["used"])) if method == "nanounet" or s in EXTERNAL_SCENARIOS)


def parse() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, rescore=True)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="eval manifest (schema seg-eval-manifest/1) written by exp00c")
    ap.add_argument("--tier", nargs="+", choices=S.TIERS, default=list(S.TIERS), help="tiers to run (default: all three)")
    ap.add_argument("--sources", nargs="+", default=None, help="manifest source names to keep (default: all sources)")
    ap.add_argument("--max-cases-per-source", type=int, default=-1, help="keep the first N cases of every source in manifest order; -1 = all")
    ap.add_argument("--max-lesions-per-case", type=int, default=-1, help="click at most N lesions per case (seeded subset); -1 = all lesions")
    ap.add_argument("--methods", nargs="*", choices=["nanounet"], default=["nanounet"], help="methods run here (nanounet); leave empty to score only --external folders")
    ap.add_argument("--external", nargs="*", action="extend", default=[], metavar="NAME=DIR", help="prediction folders of systems run elsewhere, NAME in nninteractive, uls_plus; repeatable (layout: external_preds.py)")
    ap.add_argument("--overlap-policy", choices=["common", "own"], default="common", help="common: score every method on the cases clean for all three systems; own: each method on its own clean set")
    ap.add_argument("--backends", nargs="+", choices=list(S.BACKEND_CHOICES), default=list(S.BACKEND_CHOICES), help="registration-error table backends the click offsets are drawn from")
    ap.add_argument("--emit-prompts-only", action="store_true", help="write lesion caches and click files to artifacts/, then stop (input for the external systems)")
    return ap, ap.parse_args()


def validate(args: argparse.Namespace) -> tuple[list[dict], dict[str, Path], list[str]]:
    """All startup problems at once (E6); returns the cases to run, the external folders and the methods to score."""
    abort_if(missing_paths({"manifest": args.manifest}, "python -m experiments.exp00c_seg_eval_manifest.run --tag seg_eval_v1   (or pass --manifest <file>)"))
    manifest = json.loads(args.manifest.read_text())
    abort_if(S.manifest_problems(manifest, args.manifest))
    ext_dirs, problems = parse_external(args.external)
    methods = ALL_SYSTEMS if args.emit_prompts_only else list(args.methods) + list(ext_dirs)
    if not methods:
        problems.append(problem("nothing to score: --methods is empty and no --external was given", "nanounet and/or --external NAME=DIR", "rerun with --methods nanounet"))
    if args.emit_prompts_only and (args.rescore or args.external):
        problems.append(problem("--emit-prompts-only together with --rescore or --external", "prompts are written before any system runs", "drop --rescore / --external"))
    cases, p = S.filter_cases(manifest, args.tier, args.sources, args.max_cases_per_source)
    problems += p
    pool = [c for c in cases if any(S.method_ok(c, m, args.overlap_policy) for m in methods)]
    if cases and not pool:
        flags = sorted({f"{k}={v}" for c in cases for k, v in c["overlap"].items()})
        problems.append(problem(f"no case of {len(cases)} selected passes --overlap-policy {args.overlap_policy} for {methods}", f"cases clean for the scored systems (flags seen: {flags})",
                                "rerun with --overlap-policy own, or select other tiers / sources"))
    elif not cases and not p:
        problems.append(problem("the tier / source filters select no case", "at least one manifest case", f"rerun with --tier {' '.join(S.TIERS)}"))
    pool = limited(pool, args)
    cp, sizes = S.case_problems(pool)
    problems += cp + missing_paths({"registration-error table": Path(DEFAULT_ERROR_TABLE)}, "mount /nnunet_data (Longitudinal-CT/derivatives)")
    if args.rescore is None and not args.emit_prompts_only:
        problems += missing_paths({"segmenter checkpoint": SEG_CKPT}, "mount /nnunet_data or fix common.SEG_CKPT") if "nanounet" in methods else []
        if "nanounet" in methods and args.device.startswith("cuda") and not torch.cuda.is_available():
            problems.append(problem(f"--device {args.device} is not available", "a CUDA GPU", "rerun on the GPU machine or with --device cpu (slow)"))
    if args.rescore is not None:
        problems += rescore_problems(args, pool, methods, ext_dirs, sizes)
    abort_if(problems)
    return pool, ext_dirs, methods


def ext_problems(cases: list[dict], methods: list[str], ext_dirs: dict[str, Path], sizes: dict, caches: dict, args: argparse.Namespace) -> list[str]:
    out = []
    for m, folder in ext_dirs.items():
        out += inventory_problems(m, folder, [(c, sizes[c["case_id"]], scenarios(c, caches[c["case_id"]], m)) for c in cases if S.method_ok(c, m, args.overlap_policy)])
    return out


def rescore_problems(args: argparse.Namespace, cases: list[dict], methods: list[str], ext_dirs: dict[str, Path], sizes: dict) -> list[str]:
    """--rescore needs the source run's caches, prompts and nanoUNet masks for every case it scores; externals are checked like a fresh run."""
    art, out, caches, absent = args.rescore / "artifacts", [], {}, []
    for c in cases:
        f = lesion_file(art, c["case_id"])
        if not f.is_file():
            absent.append(str(f))
            continue
        caches[c["case_id"]] = json.loads(f.read_text())
        for m in methods:
            if m == "nanounet" and S.method_ok(c, m, args.overlap_policy):
                absent += [str(p) for s in scenarios(c, caches[c["case_id"]], m) if not (p := pred_file(art, ext_dirs, m, s, c["case_id"], len(caches[c["case_id"]]["used"]))).is_file()]
    if absent:
        out.append(problem(f"{len(absent)} artifact(s) missing in the source run {args.rescore}, e.g. {absent[0]}", "the source run scored the same cases with the same filters",
                           f"rerun with the source run's --tier/--sources/--max-cases-per-source/--limit-patients, or --resume {args.rescore} to complete it"))
    return out + (ext_problems(cases, methods, ext_dirs, sizes, caches, args) if not absent else [])


def get_cache(case: dict, args: argparse.Namespace, art: Path) -> dict:
    f = lesion_file(art, case["case_id"])
    if f.is_file():
        cache = json.loads(f.read_text())
        want = {"seed": args.seed, "max_lesions_per_case": args.max_lesions_per_case}
        abort_if([problem(f"{f} was built with {cache['params']}, this run has {want}", "the same --seed and --max-lesions-per-case as the run being resumed", "rerun without --resume, or with the original values")] if args.rescore is None and cache["params"] != want else [])
        return cache
    cache, _data, _props = S.build_lesion_cache(case, args.seed, args.max_lesions_per_case)
    S.write_json(f, cache)
    return cache


def make_prompts(cache: dict, args: argparse.Namespace, wanted: tuple[str, ...]) -> dict[str, list]:
    """Click lists per scenario: the noisy S1 clicks are seed + replicate-0 draw for every lesion (draw order independent of the cap)."""
    sp, shape, cid = tuple(cache["spacing_zyx"]), tuple(cache["shape_zyx"]), cache["case_id"]
    offs = dict(zip((l["id"] for l in cache["lesions"]), S.draw_offsets(cache["lesions"], S.rng_for(args.seed, cid, "offset", 0), tuple(args.backends))))
    s1 = [(str(l["id"]), S.offset_click(l, offs[l["id"]], 1.0, sp, shape)) for l in cache["lesions"] if l["id"] in cache["used"]]
    all_p = {"S1": s1, "S2": [c for c in s1 if int(c[0]) in (cache["subset"] or [])], "S3": [], "S4": [("decoy", tuple(cache["decoy"]))] if cache["decoy"] else []}
    return {s: all_p[s] for s in ("S1", "S2", "S3", "S4") if s in wanted}


def passes(sc: tuple[str, ...], n_used: int) -> list[tuple[str, bool]]:
    """(scenario, prompt on) inference passes of nanoUNet; S3 on a scan with lesions is served by S1_noprompt."""
    p = [("S1", True), ("S1_noprompt", False)] if "S1" in sc else []
    return p + ([("S2", True)] if "S2" in sc else []) + ([("S3", False)] if "S3" in sc and n_used == 0 else []) + ([("S4", True)] if "S4" in sc else [])


def emit(cases: list[dict], args: argparse.Namespace, art: Path) -> dict[str, dict]:
    """Phase A: lesion caches and click files for every case (deterministic, cheap to redo)."""
    caches = {}
    with nano_progress(len(cases), "lesions + prompts") as adv:
        for c in cases:
            cache = get_cache(c, args, art)
            for s, clicks in make_prompts(cache, args, S.scenarios_for(c, len(cache["used"]))).items():
                S.write_clicks(prompt_file(art, s, c["case_id"]), clicks)
            caches[c["case_id"]] = cache
            adv()
    return caches


def segment_all(cases: list[dict], caches: dict, args: argparse.Namespace, art: Path) -> None:
    """Phase B: nanoUNet passes for the cases it is scored on; a pass whose mask exists is skipped (resume)."""
    sg = None  # loaded on the first missing pass, so a finished run resumes without touching the GPU
    with nano_progress(len(cases), "nanoUNet passes") as adv:
        for i, c in enumerate(cases, 1):
            cid, cache, t0 = c["case_id"], caches[c["case_id"]], time.time()
            todo = [(s, enc) for s, enc in passes(S.scenarios_for(c, len(cache["used"])), len(cache["used"])) if not pred_file(art, {}, "nanounet", s, cid, len(cache["used"])).is_file()]
            if todo:
                sg = sg or S.load_segmenter(args.device)
                data, props = S.read_ct(c["image"])
                scan = S.prepare_scan(sg, data, props)
                for s, enc in todo:
                    mask = S.segment_points(sg, scan, load_points_xyz(str(prompt_file(art, PROMPT_OF[s], cid))), encode_prompt=enc)
                    S.write_mask(mask, scan.sitk_stuff, pred_file(art, {}, "nanounet", s, cid, len(cache["used"])))
                del scan, data
            cprint(f"[dim]{i}/{len(cases)} {cid}: {len(cache['used'])} lesions, {len(todo)} passes, {time.time() - t0:.0f} s[/dim]")
            adv()


def score_scenario(base: dict, sc: str, cache: dict, inst: np.ndarray | None, mask: np.ndarray, clicks: list) -> tuple[list[dict], dict]:
    """Per-lesion rows (S1, S1_noprompt, S2) and the per-case row of one (method, case, scenario)."""
    used = [l for l in cache["lesions"] if l["id"] in cache["used"]]
    sp, crow, lrows = tuple(cache["spacing_zyx"]), {**base, "scenario": sc, "status": "ok", "n_lesions": len(used)}, []
    if sc in LESION_SCENARIOS:
        click_of = {int(n): z for n, z in clicks}
        res = S.score_lesions(mask, inst, used, sp, detection_only=base["annotation"] in S.DETECTION_ONLY)
        for l in used:
            k = click_of.get(l["id"])
            lrows.append({**base, "lesion_id": l["id"], "size_mm": l["size_mm"], "size_bin": l["size_bin"], "scenario": sc, "clicked": k is not None,
                          "offset_mm": S.offset_mm(k, l["seed_zyx"], sp)[1] if k else None, "click_hit": bool(inst[k] == l["id"]) if k else None, **res[l["id"]]})
        cl, un = [r for r in lrows if r["clicked"]], [r for r in lrows if not r["clicked"]]
        avg = lambda rs, key: mean([r[key] for r in rs if r[key] is not None]) if any(r[key] is not None for r in rs) else None
        crow.update(n_clicked=len(cl), dsc=avg(cl, "dice"), nsd=avg(cl, "nsd"), hit_rate=avg(cl, "hit"), leak_rate=avg(un, "hit") if un else None)
        crow.update(S.fg_stats(mask, inst, [l for l in used if l["id"] in click_of] if sc == "S2" else None))
    else:
        crow.update(S.fg_stats(mask, inst))
    return lrows, crow


def score_all(cases: list[dict], methods: list[str], ext_dirs: dict[str, Path], caches: dict, args: argparse.Namespace, art: Path) -> tuple[list, list, dict]:
    """Phase C (CPU): read every mask, score it; the same code for nanoUNet and the external systems."""
    lrows, crows, dropped = [], [], defaultdict(list)
    with nano_progress(len(cases), "scoring") as adv:
        for c in cases:
            cid, cache = c["case_id"], caches[c["case_id"]]
            inst = S.read_instances(c["label"], c["lesion_label_values"]) if c["label"] is not None else None
            for m in methods:
                if not S.method_ok(c, m, args.overlap_policy):
                    dropped[m].append(cid)
                    continue
                base = {"method": m, "tier": c["tier"], "cohort": c["source"], "organ": c["cancer_type"], "patient": c["patient_id"], "case": cid, "annotation": c["annotation"]}
                sc = scenarios(c, cache, m)
                if not sc:
                    crows.append({**base, "scenario": None, "status": "no supported scenario (no lesion)", "n_lesions": 0})
                for s in sc:
                    mask = S.read_mask(pred_file(art, ext_dirs, m, s, cid, len(cache["used"])))
                    assert mask.shape == tuple(cache["shape_zyx"]), f"{m} {s} {cid}: mask {mask.shape} vs scan {cache['shape_zyx']}"
                    lr, cr = score_scenario(base, s, cache, inst, mask, S.read_clicks(prompt_file(art, PROMPT_OF[s], cid)) if s in LESION_SCENARIOS else [])
                    lrows += lr
                    crows.append(cr)
            adv()
    return lrows, crows, dropped


def stat_rows(method: str, tier: str, group: str, value: str, cr: list[dict], lr: list[dict]) -> list[dict]:
    """Every CI of one (method, tier, group value): lesion metrics case-mean and lesion-level, S2 selectivity, S3/S4 quietness, prompt drop."""
    out = []

    def add(scenario: str, metric: str, items: list[dict], key: str, stat=mean) -> None:
        r = ci(items, key, stat)
        if r is not None:
            got = [x for x in items if x.get(key) is not None]
            out.append({"method": method, "tier": tier, "group": group, "value": value, "scenario": scenario, "metric": metric, "n": len(got),
                        "n_patients": len({x["patient"] for x in got}), "point": r[0], "lo": r[1], "hi": r[2]})

    for sc in LESION_SCENARIOS:
        c, l = [r for r in cr if r["scenario"] == sc], [r for r in lr if r["scenario"] == sc and r["clicked"]]
        for metric, key in (("dsc_case_mean", "dsc"), ("nsd_case_mean", "nsd"), ("ldr_case_mean", "hit_rate")):
            add(sc, metric, c, key)
        for metric, key in (("dsc_lesion_mean", "dice"), ("nsd_lesion_mean", "nsd"), ("ldr_lesion_mean", "hit")):
            add(sc, metric, l, key)
    c2 = [r for r in cr if r["scenario"] == "S2"]
    add("S2", "leak_rate_case_mean", c2, "leak_rate")
    add("S2", "fg_dice_margin_case_mean", c2, "fg_dice_margin")
    for sc in ("S3", "S4"):
        c = [r for r in cr if r["scenario"] == sc]
        for metric, key in (("any_fg_rate", "any_fg"), ("any_fg_outside_gt_rate", "any_fg_outside_gt"), ("fg_vox_mean", "fg_vox")):
            add(sc, metric, c, key)
        add(sc, "fg_vox_median", c, "fg_vox", lambda xs: float(np.median(xs)))
    for metric, items, key in (("dsc_case_mean", cr, "dsc"), ("ldr_case_mean", cr, "hit_rate"), ("dsc_lesion_mean", lr, "dice"), ("ldr_lesion_mean", lr, "hit")):
        a = by_patient([r for r in items if r["scenario"] == "S1" and r.get("clicked", True)], key)
        b = by_patient([r for r in items if r["scenario"] == "S1_noprompt" and r.get("clicked", True)], key)
        if a and a.keys() == b.keys():
            d = [float(v) for v in paired_delta_stat(a, b, mean)]
            out.append({"method": method, "tier": tier, "group": group, "value": value, "scenario": "S1 minus S1_noprompt", "metric": metric,
                        "n": sum(map(len, a.values())), "n_patients": len(a), "point": d[0], "lo": d[1], "hi": d[2]})
    return out


def summarise(lrows: list[dict], crows: list[dict], methods: list[str]) -> list[dict]:
    """Flat table of CIs over method x tier x group (all, cohort, organ, size bin); size bins have lesion-level metrics only."""
    table, ok = [], [r for r in crows if r["status"] == "ok"]
    for m in methods:
        for tier in S.TIERS:
            cr, lr = [r for r in ok if r["method"] == m and r["tier"] == tier], [r for r in lrows if r["method"] == m and r["tier"] == tier]
            if not cr:
                continue
            table += stat_rows(m, tier, "all", "all", cr, lr)
            for group, key in (("cohort", "cohort"), ("organ", "organ")):
                for v in sorted({r[key] for r in cr}):
                    table += stat_rows(m, tier, group, v, [r for r in cr if r[key] == v], [r for r in lr if r[key] == v])
            for v in sorted({r["size_bin"] for r in lr}):
                table += stat_rows(m, tier, "size_bin", v, [], [r for r in lr if r["size_bin"] == v])
    return table


def nest(table: list[dict], crows: list[dict]) -> dict:
    """summary = {method: {tier: {scenario: {metric: [point, lo, hi], n_cases}}}} from the group `all` rows."""
    out = {}
    for r in table:
        if r["group"] == "all":
            out.setdefault(r["method"], {}).setdefault(r["tier"], {}).setdefault(r["scenario"], {})[r["metric"]] = [r["point"], r["lo"], r["hi"]]
    for r in crows:
        if r["status"] == "ok":
            out.setdefault(r["method"], {}).setdefault(r["tier"], {}).setdefault(r["scenario"], {}).setdefault("n_cases", 0)
            out[r["method"]][r["tier"]][r["scenario"]]["n_cases"] += 1
    return out


def markdown(summary: dict) -> str:
    f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" if v else "-"
    lines = ["# exp01 segmentation (patient-level bootstrap 95% CI; case-mean of per-lesion metrics)", ""]
    for m, tiers in summary.items():
        for tier, scs in tiers.items():
            lines += [f"## {m} / {tier}", "", "| scenario | cases | DSC | NSD@1mm | LDR | leak (unclicked hit) | S2 margin | any FG | FG voxels (median) |", "|---|---|---|---|---|---|---|---|---|"]
            for sc, v in scs.items():
                lines.append(f"| {sc} | {v.get('n_cases', '-')} | {f(v.get('dsc_case_mean'))} | {f(v.get('nsd_case_mean'))} | {f(v.get('ldr_case_mean'))} | {f(v.get('leak_rate_case_mean'))} | "
                             f"{f(v.get('fg_dice_margin_case_mean'))} | {f(v.get('any_fg_rate'))} | {f(v.get('fg_vox_median'))} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap, args = parse()
    cases, ext_dirs, methods = validate(args)
    run = start_run(EXP, ap, args, inputs={"manifest": args.manifest, "segmenter_checkpoint": SEG_CKPT, "registration_error_table": Path(DEFAULT_ERROR_TABLE)}, paper=PAPER)
    art = run.artifacts
    caches = {c["case_id"]: json.loads(lesion_file(art, c["case_id"]).read_text()) for c in cases} if args.rescore is not None else emit(cases, args, art)
    if args.emit_prompts_only:
        rows = [{"case": c["case_id"], "tier": c["tier"], "annotation": c["annotation"], "n_lesions": len(caches[c["case_id"]]["lesions"]), "n_used": len(caches[c["case_id"]]["used"]),
                 "scenarios": S.scenarios_for(c, len(caches[c["case_id"]]["used"]))} for c in cases]
        run.finish({"cases": len(rows), "prompts_dir": str(art / "prompts")}, {"cases": rows}, notes=["prompts only: no inference, no scoring"],
                   next_cmd=f"run ULS+ / nnInteractive on {art}/prompts/<scenario>/<case>.json into <dir>/<scenario>/<case>.nii.gz, then rescore with --external")
        return
    if args.rescore is None:
        abort_if(ext_problems(cases, methods, ext_dirs, {k: tuple(v["shape_zyx"]) for k, v in caches.items()}, caches, args))  # before any GPU work
        if "nanounet" in methods:
            segment_all([c for c in cases if S.method_ok(c, "nanounet", args.overlap_policy)], caches, args, art)
    lrows, crows, dropped = score_all(cases, methods, ext_dirs, caches, args, art)
    table = summarise(lrows, crows, methods)
    summary = nest(table, crows)
    noise = load_table(DEFAULT_ERROR_TABLE)
    definitions = {"protocol": "longiseg_lesion_v1: per-lesion Dice, NSD, detection; case-mean then mean over cases", "iou_hit": IOU_HIT, "nsd_tol_mm": NSD_TOL_MM,
                   "gt_lesion_connectivity": S.GT_CONNECTIVITY, "pred_component_connectivity": S.PRED_CONNECTIVITY, "click_seed": "argmax EDT (mm) of the native GT component, inside the lesion",
                   "click_noise": {"table": DEFAULT_ERROR_TABLE, "table_spacing_zyx": noise["spacing_zyx"], "backends": args.backends, "scale": 1.0, "draw": "empirical, size-bin matched, once per lesion"},
                   "inference": {"mode": "clustered", "border_expand": True, "amp": True, "batch_size": 8, "cluster_margin_frac": 0.1, "seg_ckpt": str(SEG_CKPT), "ema": SEG_EMA},
                   "decoy": {"min_distance_mm": S.DECOY_GUARD_MM, "tissue_hu_above": S.TISSUE_HU}, "overlap_policy": args.overlap_policy,
                   "s3": "nanounet: tile with zeroed prompt at the S1 clicks (scan with lesions) or at the decoy (scan without); external: empty click list"}
    notes = [f"{m}: {len(v)} of {len(cases)} pooled cases not scored under --overlap-policy {args.overlap_policy}" for m, v in dropped.items() if v]
    notes += [f"{r['method']} {r['case']}: {r['status']}" for r in crows if r["status"] != "ok"]
    notes += ["S3 for nanoUNet is a tile with no click (a call with an empty click list does nothing); external systems get an empty click list",
              f"click offsets drawn from backends {args.backends}; --max-lesions-per-case {args.max_lesions_per_case} (-1 = all): a cap leaves the other lesions unclicked in S1"]
    run.finish(summary, {"per_lesion": lrows, "per_case": crows, "summary_table": table, "cases": [{k: c[k] for k in ("case_id", "patient_id", "tier", "source", "cancer_type", "annotation", "overlap")} for c in cases]},
               table_md=markdown(summary), definitions=definitions, notes=notes,
               next_cmd=f"python -m experiments.exp02_prompt_noise.run --tag paper_v1 --crosscheck-run {run.dir}")


if __name__ == "__main__":
    main()
