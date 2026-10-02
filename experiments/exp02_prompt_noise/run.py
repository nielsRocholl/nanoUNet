"""exp02 - Prompt-noise sweep  (paper: Experiments > Prompt noise; Table 'nine experiments' row 2)

QUESTION   What does registration error cost the segmenter, and at which click offset (in mm, per lesion size) does the lesion stop being found?
WHY        Gives the Dice-and-detection-versus-offset curves per size bin that turn the calibration table (sec:calibration) into a segmentation
           cost, and shows how far the click may drift before deployment breaks.
DATA       Same manifest, tiers (seen-cohort, outside; scans without lesions are skipped), overlap policy and filters as exp01, but capped by default
           (--max-cases-per-source 10, --max-lesions-per-case 8) because one case costs 1 + 5 x replicates passes. All lesions of a scan are clicked
           at once (deployment); partial annotations are fine (only S1-type clicks), pseudo / spheres labels score detection only.
METHOD     1. Per lesion and replicate r, ONE offset is drawn from the empirical registration-error table (size-bin matched, backends --backends) and
              only scaled afterwards: click(s) = seed + s * offset (table voxels -> mm -> native voxels, rounded), so the curves over s are nested.
              Replicate 0 at s = exp01's --click-noise-scale (default 0: the seed itself) is exactly exp01's S1 clicks (same --seed, --backends,
              --max-lesions-per-case).
           2. One pass per (s, r) over the whole scan with nanoUNet (EMA, clustered tiles, one preprocessing per scan); s = 0 (true seed) has one
              replicate only. Masks go to artifacts/preds/s<s>_r<r>/, clicks to artifacts/prompts/s<s>_r<r>/.
           3. Score per lesion (Dice, NSD@1 mm, hit = IoU > 0.1) and store for every (lesion, replicate, s): the effective click offset (native voxels,
              mm vector, magnitude), size mm / bin, cohort, click_hit (click inside its own lesion), Dice, hit. Summaries: by scale and by offset
              magnitude (mm bins), each per size bin, patient-level bootstrap CIs over lesion-level values.
           4. Cross-check (--crosscheck-run EXP01_RUN_DIR): exp02 at s = exp01's click-noise scale (read from its results.json), replicate 0, must equal
              exp01's S1 Dice lesion by lesion (identical click files); when that scale is not 1, exp01's S1 minus exp02's s = 1 replicate 0 is the cost
              of the noise itself. Both go to `notes` and `summary.crosscheck`.
OUTPUT     results.json tables: per_lesion (every (lesion, replicate, s) value), by_scale, by_offset_mm; artifacts/: lesions, prompts, preds.
COMMAND    python -m experiments.exp02_prompt_noise.run --tag paper_v1 --crosscheck-run /nnunet_data/experiments/exp01_segmentation/<RUN_ID>
DEPENDS ON experiments/common.py, segment.py, scoring.py; helpers of experiments/exp01_segmentation/run.py (lesion cache, CI); an exp01 run for --crosscheck-run.
RUNTIME    About 16 passes per case, 40-90 s per case: 2-4 h at the default caps on one A100. Resumable per pass; --rescore redoes only the scoring.
CAVEATS    Offsets are rounded to the native grid (3 mm slices), so the effective magnitude differs a little from s times the draw; the effective value is
           the one stored. Clicks outside the scan are clipped to its border. Lesion-level values are averaged over replicates only in the summaries.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core.ui import cprint, nano_progress
from experiments import segment as S
from experiments.common import REPO, SEG_CKPT, SEG_EMA, abort_if, add_common_args, limited, missing_paths, problem, start_run
from experiments.exp01_segmentation.run import ci, get_cache, lesion_file, mean, prompt_file
from nanounet.data.patch.error_table import DEFAULT_ERROR_TABLE, load_table
from nanounet.prompt.coords import load_points_xyz
from nanounet.score import IOU_HIT, NSD_TOL_MM

EXP = "exp02_prompt_noise"
PAPER = {"section": "Experiments > Prompt noise", "table_row": 2, "supports": "Dice and detection versus click offset in mm per lesion size"}
DEFAULT_MANIFEST = REPO / "experiments" / "exp00c_seg_eval_manifest" / "seg_eval_v1.json"
OFFSET_EDGES_MM = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 1e9]


def key_of(scale: float, rep: int) -> str:
    return f"s{scale:g}_r{rep}"


def pass_list(scales: list[float], reps: int) -> list[tuple[float, int]]:
    """(scale, replicate) passes; scale 0 is the true seed, identical for every replicate, so it runs once."""
    return [(s, r) for s in scales for r in ([0] if s == 0 else range(reps))]


def offset_label(mag: float) -> str:
    i = next(i for i in range(len(OFFSET_EDGES_MM) - 1) if OFFSET_EDGES_MM[i] <= mag < OFFSET_EDGES_MM[i + 1])
    lo, hi = OFFSET_EDGES_MM[i], OFFSET_EDGES_MM[i + 1]
    return f"{lo:04.1f}-{hi:04.1f}" if hi < 1e8 else f">={lo:04.1f}"


def parse() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, rescore=True)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="eval manifest (schema seg-eval-manifest/1) written by exp00c")
    ap.add_argument("--tier", nargs="+", choices=S.TIERS, default=["seen-cohort", "outside"], help="tiers to run (scans without lesions are skipped)")
    ap.add_argument("--sources", nargs="+", default=None, help="manifest source names to keep (default: all sources)")
    ap.add_argument("--max-cases-per-source", type=int, default=10, help="keep the first N cases of every source in manifest order; -1 = all")
    ap.add_argument("--max-lesions-per-case", type=int, default=8, help="click at most N lesions per case (seeded subset); -1 = all lesions")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5], help="offset scales s (0 = true seed, 1 = the full empirical draw, 1.5 = stress point)")
    ap.add_argument("--replicates", type=int, default=3, help="independent offset draws per lesion (scale 0 needs one)")
    ap.add_argument("--overlap-policy", choices=["common", "own"], default="common", help="common: only cases clean for all three systems; own: cases clean for nanoUNet")
    ap.add_argument("--backends", nargs="+", choices=list(S.BACKEND_CHOICES), default=list(S.BACKEND_CHOICES), help="registration-error table backends the offsets are drawn from")
    ap.add_argument("--crosscheck-run", type=Path, default=None, help="exp01 RUN_DIR (same manifest, --seed, --backends, --max-lesions-per-case) to cross-check s=1/s=0 against its S1 (default: skip)")
    return ap, ap.parse_args()


def validate(args: argparse.Namespace) -> list[dict]:
    """All startup problems at once (E6); returns the cases to run."""
    abort_if(missing_paths({"manifest": args.manifest}, "python -m experiments.exp00c_seg_eval_manifest.run --tag seg_eval_v1   (or pass --manifest <file>)"))
    manifest = json.loads(args.manifest.read_text())
    abort_if(S.manifest_problems(manifest, args.manifest))
    cases, problems = S.filter_cases(manifest, args.tier, args.sources, args.max_cases_per_source)
    cases = [c for c in cases if c["annotation"] != "healthy" and S.method_ok(c, "nanounet", args.overlap_policy)]
    if not cases and not problems:
        problems.append(problem(f"no case with lesions passes --tier {args.tier} and --overlap-policy {args.overlap_policy}", "at least one non-healthy manifest case clean for nanoUNet",
                                "rerun with --overlap-policy own, or other --tier / --sources"))
    if args.replicates < 1 or any(s < 0 for s in args.scales) or not args.scales:
        problems.append(problem(f"--scales {args.scales} / --replicates {args.replicates} invalid", "scales >= 0 (at least one) and replicates >= 1", "rerun with --scales 0 0.5 1.0 --replicates 3"))
    cases = limited(cases, args)
    cp, _ = S.case_problems(cases)
    problems += cp + missing_paths({"registration-error table": Path(DEFAULT_ERROR_TABLE)}, "mount /nnunet_data (Longitudinal-CT/derivatives)")
    if args.crosscheck_run is not None:
        problems += missing_paths({"--crosscheck-run results": args.crosscheck_run / "results.json"}, "pass the RUN_DIR of a finished exp01 run")
    if args.rescore is None:
        problems += missing_paths({"segmenter checkpoint": SEG_CKPT}, "mount /nnunet_data or fix common.SEG_CKPT")
    else:
        art = args.rescore / "artifacts"
        absent = [str(art / "lesions" / f"{c['case_id']}.json") for c in cases if not lesion_file(art, c["case_id"]).is_file()]
        if absent:
            problems.append(problem(f"{len(absent)} lesion cache(s) missing in the source run {args.rescore}, e.g. {absent[0]}", "the source run scored the same cases with the same filters",
                                    f"rerun with the source run's --tier/--sources/--max-cases-per-source/--limit-patients, or --resume {args.rescore}"))
        else:
            for c in cases:
                cache = json.loads(lesion_file(art, c["case_id"]).read_text())
                miss = [k for s, r in pass_list(args.scales, args.replicates) if cache["used"] and not (art / "preds" / (k := key_of(s, r)) / f"{c['case_id']}.nii.gz").is_file()]
                if miss:
                    problems.append(problem(f"{c['case_id']}: no mask for passes {miss[:3]} in {args.rescore}", "a source run with the same --scales and --replicates", f"rerun with the source run's flags or --resume {args.rescore}"))
                    break
    abort_if(problems)
    return cases


def make_prompts(cache: dict, args: argparse.Namespace, rep: int) -> dict[float, list]:
    """Clicks per scale for replicate `rep`: one draw per lesion, scaled; every lesion in id order draws, so a cap never changes a lesion's draw."""
    sp, shape, cid = tuple(cache["spacing_zyx"]), tuple(cache["shape_zyx"]), cache["case_id"]
    offs = dict(zip((l["id"] for l in cache["lesions"]), S.draw_offsets(cache["lesions"], S.rng_for(args.seed, cid, "offset", rep), tuple(args.backends))))
    used = [l for l in cache["lesions"] if l["id"] in cache["used"]]
    return {s: [(str(l["id"]), S.offset_click(l, offs[l["id"]], s, sp, shape)) for l in used] for s in args.scales}


def emit_and_segment(cases: list[dict], args: argparse.Namespace, art: Path) -> dict[str, dict]:
    """Phase A+B: lesion caches, click files and one nanoUNet pass per (scale, replicate) with a missing mask (resume)."""
    caches, sg = {}, None
    with nano_progress(len(cases), "noise sweep") as adv:
        for i, c in enumerate(cases, 1):
            cid, t0 = c["case_id"], time.time()
            cache = caches[cid] = get_cache(c, args, art)
            todo = []
            if cache["used"]:
                for r in range(args.replicates):
                    for s, clicks in make_prompts(cache, args, r).items():
                        if (s, r) in pass_list(args.scales, args.replicates):
                            S.write_clicks(prompt_file(art, key_of(s, r), cid), clicks)
                todo = [(s, r) for s, r in pass_list(args.scales, args.replicates) if not (art / "preds" / key_of(s, r) / f"{cid}.nii.gz").is_file()]
            if todo:
                sg = sg or S.load_segmenter(args.device)
                data, props = S.read_ct(c["image"])
                scan = S.prepare_scan(sg, data, props)
                for s, r in todo:
                    mask = S.segment_points(sg, scan, load_points_xyz(str(prompt_file(art, key_of(s, r), cid))), encode_prompt=True)
                    S.write_mask(mask, scan.sitk_stuff, art / "preds" / key_of(s, r) / f"{cid}.nii.gz")
                del scan, data
            cprint(f"[dim]{i}/{len(cases)} {cid}: {len(cache['used'])} lesions, {len(todo)} passes, {time.time() - t0:.0f} s[/dim]")
            adv()
    return caches


def score_all(cases: list[dict], caches: dict, args: argparse.Namespace, art: Path) -> tuple[list[dict], list[dict]]:
    """Phase C (CPU): every (lesion, replicate, scale) row from the stored masks and click files."""
    rows, status = [], []
    with nano_progress(len(cases), "scoring") as adv:
        for c in cases:
            cid, cache = c["case_id"], caches[c["case_id"]]
            used = [l for l in cache["lesions"] if l["id"] in cache["used"]]
            if not used:
                status.append({"case": cid, "patient": c["patient_id"], "status": "no lesion in the label: nothing to sweep"})
                adv()
                continue
            inst, sp = S.read_instances(c["label"], c["lesion_label_values"]), tuple(cache["spacing_zyx"])
            base = {"tier": c["tier"], "cohort": c["source"], "organ": c["cancer_type"], "patient": c["patient_id"], "case": cid, "annotation": c["annotation"]}
            for s, r in pass_list(args.scales, args.replicates):
                mask, click_of = S.read_mask(art / "preds" / key_of(s, r) / f"{cid}.nii.gz"), {int(n): z for n, z in S.read_clicks(prompt_file(art, key_of(s, r), cid))}
                res = S.score_lesions(mask, inst, used, sp, detection_only=c["annotation"] in S.DETECTION_ONLY)
                for l in used:
                    k = click_of[l["id"]]
                    vec_mm, mag = S.offset_mm(k, l["seed_zyx"], sp)
                    rows.append({**base, "lesion_id": l["id"], "size_mm": l["size_mm"], "size_bin": l["size_bin"], "scale": s, "replicate": r,
                                 "offset_vox_zyx": [int(a - b) for a, b in zip(k, l["seed_zyx"])], "offset_mm_zyx": vec_mm, "offset_mm": mag, "offset_bin": offset_label(mag),
                                 "click_hit": bool(inst[k] == l["id"]), **res[l["id"]]})
            status.append({"case": cid, "patient": c["patient_id"], "status": "ok", "n_lesions": len(used)})
            adv()
    return rows, status


def curve_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """by_scale and by_offset_mm tables: Dice / NSD / hit (lesion-level mean over lesions and replicates, patient bootstrap) per size bin and overall."""
    out = {"by_scale": [], "by_offset_mm": []}
    for table, xkey in (("by_scale", "scale"), ("by_offset_mm", "offset_bin")):
        for x in sorted({r[xkey] for r in rows}):
            for sb in ["all"] + sorted({r["size_bin"] for r in rows}, key=lambda b: float(b.strip(">").split("-")[0])):
                sel = [r for r in rows if r[xkey] == x and (sb == "all" or r["size_bin"] == sb)]
                row = {xkey: x, "size_bin": sb, "n_lesion_values": len(sel), "n_patients": len({r["patient"] for r in sel})}
                for name, key in (("dsc", "dice"), ("nsd", "nsd"), ("hit", "hit"), ("click_hit", "click_hit")):
                    vals = [{**r, key: float(r[key])} for r in sel if r[key] is not None]
                    v = ci(vals, key)
                    row.update({name: v[0] if v else None, f"{name}_lo": v[1] if v else None, f"{name}_hi": v[2] if v else None})
                out[table].append(row)
    return out["by_scale"], out["by_offset_mm"]


def crosscheck(rows: list[dict], args: argparse.Namespace, art: Path) -> tuple[dict, list[str]]:
    """exp02 at exp01's click-noise scale (replicate 0) against exp01's S1 on the same clicks; the noise cost is exp01's S1 minus exp02's s=1."""
    if args.crosscheck_run is None:
        return {"done": False}, ["cross-check against exp01 NOT run (no --crosscheck-run): exp02 at exp01's scale vs its S1 is unverified"]
    res = json.loads((args.crosscheck_run / "results.json").read_text())
    ref = float(res["definitions"].get("click_noise", {}).get("scale", 1.0))  # runs before the flag existed were noisy at s = 1
    abort_if([problem(f"exp01 run {args.crosscheck_run.name} used click-noise scale {ref:g}, which is not in --scales {args.scales}", "exp02 scales that include exp01's scale (0 by default)", f"rerun with --scales 0 {ref:g} ...")] if ref not in args.scales else [])
    s1 = {(r["case"], r["lesion_id"]): r for r in res["tables"]["per_lesion"] if r["method"] == "nanounet" and r["scenario"] == "S1" and r["dice"] is not None}
    at = lambda sc: {(r["case"], r["lesion_id"]): r for r in rows if r["scale"] == sc and r["replicate"] == 0 and r["dice"] is not None}
    same_ref = at(ref)
    same = [k for k in same_ref if k in s1 and (args.crosscheck_run / "artifacts" / "prompts" / "S1" / f"{k[0]}.json").is_file()
            and S.read_clicks(args.crosscheck_run / "artifacts" / "prompts" / "S1" / f"{k[0]}.json") == S.read_clicks(prompt_file(art, key_of(ref, 0), k[0]))]
    diffs = [abs(same_ref[k]["dice"] - s1[k]["dice"]) for k in same]
    out = {"done": True, "exp01_run": str(args.crosscheck_run), "exp01_click_noise_scale": ref, "lesions_compared": len(same), "max_abs_dice_diff_s1": max(diffs) if diffs else None,
           "lesions_with_identical_dice": sum(d < 1e-6 for d in diffs), "click_files_identical_for_cases": len({k[0] for k in same})}
    note = f"cross-check vs exp01 {args.crosscheck_run.name}: s={ref:g} replicate 0 vs S1 on identical click files, {len(same)} lesions, max |Dice diff| {out['max_abs_dice_diff_s1']}"
    one = at(1.0)
    both = [k for k in one if k in s1] if ref != 1.0 else []
    if both:
        out["noise_cost_dsc_s1_minus_s1r0"] = mean([s1[k]["dice"] for k in both]) - mean([one[k]["dice"] for k in both])
        out["lesions_noise_cost"] = len(both)
        note += f"; exp01 S1 minus exp02 s=1 replicate 0 (the cost of the noise): mean Dice difference {out['noise_cost_dsc_s1_minus_s1r0']} over {len(both)} lesions"
    return out, [note]


def markdown(by_scale: list[dict]) -> str:
    f = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]" if r[k] is not None else "-"
    lines = ["# exp02 prompt noise (lesion-level mean over lesions and replicates; patient-bootstrap 95% CI)", "", "| scale | size bin | lesion values | DSC | NSD@1mm | detection | click inside lesion |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['scale']:g} | {r['size_bin']} | {r['n_lesion_values']} | {f(r, 'dsc')} | {f(r, 'nsd')} | {f(r, 'hit')} | {f(r, 'click_hit')} |" for r in by_scale]
    return "\n".join(lines)


def main() -> None:
    ap, args = parse()
    cases = validate(args)
    run = start_run(EXP, ap, args, inputs={"manifest": args.manifest, "segmenter_checkpoint": SEG_CKPT, "registration_error_table": Path(DEFAULT_ERROR_TABLE)}, paper=PAPER)
    art = run.artifacts
    caches = {c["case_id"]: json.loads(lesion_file(art, c["case_id"]).read_text()) for c in cases} if args.rescore is not None else emit_and_segment(cases, args, art)
    rows, status = score_all(cases, caches, args, art)
    by_scale, by_offset = curve_rows(rows)
    check, check_notes = crosscheck(rows, args, art)
    allrows = [r for r in by_scale if r["size_bin"] == "all"]
    summary = {"crosscheck": check, "n_cases": len(cases), "n_lesion_values": len(rows), "dsc_by_scale": {f"{r['scale']:g}": [r["dsc"], r["dsc_lo"], r["dsc_hi"]] for r in allrows if r["dsc"] is not None},
               "detection_by_scale": {f"{r['scale']:g}": [r["hit"], r["hit_lo"], r["hit_hi"]] for r in allrows}}
    definitions = {"protocol": "longiseg_lesion_v1 per lesion (Dice, NSD, hit = IoU > 0.1); lesion-level means", "iou_hit": IOU_HIT, "nsd_tol_mm": NSD_TOL_MM, "gt_lesion_connectivity": S.GT_CONNECTIVITY,
                   "pred_component_connectivity": S.PRED_CONNECTIVITY, "scales": args.scales, "replicates": args.replicates, "offset_bins_mm": OFFSET_EDGES_MM,
                   "click_noise": {"table": DEFAULT_ERROR_TABLE, "table_spacing_zyx": load_table(DEFAULT_ERROR_TABLE)["spacing_zyx"], "backends": args.backends, "draw": "once per (lesion, replicate), then scaled"},
                   "inference": {"mode": "clustered", "border_expand": True, "amp": True, "batch_size": 8, "seg_ckpt": str(SEG_CKPT), "ema": SEG_EMA}, "overlap_policy": args.overlap_policy}
    notes = check_notes + [f"{r['case']}: {r['status']}" for r in status if r["status"] != "ok"] + [
        f"scale 0 has one replicate (the seed itself); other scales have {args.replicates}", f"caps: {args.max_cases_per_source} cases per source, {args.max_lesions_per_case} lesions per case (-1 = all)",
        f"offsets drawn from backends {args.backends}; effective offsets (rounded to the native grid, clipped to the scan) are stored"]
    run.finish(summary, {"per_lesion": rows, "by_scale": by_scale, "by_offset_mm": by_offset, "cases": status}, table_md=markdown(by_scale), definitions=definitions, notes=notes,
               next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
