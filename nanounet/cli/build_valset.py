"""Offline build of the fixed validation manifest: 1500 patches over 4 prompt scenarios.

Everything expensive happens HERE, once, so validation stays pure tensor work: connected
components, the clicked-subset targets, both prompt draws, and the click-inside flags are all
resolved and written to disk. Nothing in nanounet/data/valset.py randomises or recomputes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nanounet.common import cprint, nano_header, nano_progress, preprocessed_dir, resolve_user_config_path
from nanounet.config import load_config
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.data.valset import SCENARIOS, SCHEMA_VERSION, SIZE_BUCKETS, SMALL_LESION_MAX_VOX, _sidecar_path
from nanounet.data.valset_alloc import allocate, load_cohorts, scenario_allocation
from nanounet.data.valset_build import (
    LabelCache,
    build_subset_target,
    case_info,
    report_composition,
    try_foreground,
    try_lesion_free_decoy,
)
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.plans import Plans
from nanounet.plan.splits import cohort_of
from nanounet.train.patch_render import click_inside_flags

MIX_ORDER = ("all_clicked", "lesion_free_decoy", "subset_clicked", "none_clicked")


def _parse_mix(s: str) -> dict[str, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--mix needs 4 comma-separated shares (got {s!r}), order: {MIX_ORDER}")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError(f"--mix shares must sum to 1.0, got {sum(parts)} ({s!r})")
    return dict(zip(MIX_ORDER, parts))


def _fill_scenario(scenario, ds, case_dir, ids, want, max_tries, rngs, patch_size, prop_cfg, label_cache, case_cache, reject):
    """Rejection-sample `want` accepted patches for one (cohort, scenario). `rngs` is the single
    build-wide RNG tuple, consumed in deterministic order -- reproducibility depends on every
    caller visiting (cohort, scenario) pairs in the same fixed order every run."""
    rng_case, rng_bbox, rng_choice, rng1, rng2, rng_decoy = rngs
    out = []
    tries, budget = 0, max_tries * max(want, 1)
    while len(out) < want:
        tries += 1
        if tries > budget:
            raise RuntimeError(
                f"Could not fill scenario '{scenario}': needed {want}, got {len(out)} after {tries} "
                f"attempts across {len(ids)} cases.\n"
                f"Most likely this cohort has too few multi-lesion cases for 'subset_clicked'.\n"
                f"Fix: lower its share with --mix, or raise --max-tries"
            )
        cid = ids[int(rng_case.integers(len(ids)))]
        case = case_info(case_dir, cid, case_cache)
        if scenario == "lesion_free_decoy":
            res = try_lesion_free_decoy(ds, cid, case, patch_size, rng_bbox, rng_decoy)
        else:
            res = try_foreground(scenario, ds, cid, case, patch_size, prop_cfg, SMALL_LESION_MAX_VOX, rng_bbox, rng_choice, rng1, rng2)
        if res is None:
            reject[(scenario, "not_accepted")] += 1
            continue
        if res == "reject_zero_clicks":
            reject[(scenario, "zero_clicks")] += 1
            continue
        entry = {
            "case": cid, "cohort": cohort_of(cid), "scenario": scenario, "bbox": res["bbox"],
            "clicks_zyx": [list(p) for p in res["clicks"]], "clicks2_zyx": [list(p) for p in res["clicks2"]],
            "n_false_pos": res["n_fp"], "size_bucket": res["size_bucket"],
        }
        row = None
        if res["instance_ids"] is not None:
            labels = label_cache.get(ds, cid, len(case.cts_global))
            row = build_subset_target(labels, res["bbox"], res["instance_ids"])
        out.append((entry, row, res["seg_crop"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-patches", type=int, default=1500)
    ap.add_argument("--floor", type=int, default=40)
    ap.add_argument("--mix", default="0.40,0.25,0.20,0.15")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tries", type=int, default=60)
    args = ap.parse_args()

    ds_name = convert_id_to_dataset_name(args.dataset_id)
    nano_header(f"nanoUNet build-valset  {ds_name}  n={args.n_patches}", color="green")
    pp = preprocessed_dir()
    pm = Plans(join(pp, ds_name, args.plans + ".json"))
    cm = pm.get_configuration("3d_fullres")
    case_dir = join(pp, ds_name, cm.data_identifier)
    cfg_path = resolve_user_config_path(args.config)
    roi_cfg = load_config(cfg_path)
    patch_size = np.array(cm.patch_size)

    by_cohort, cohort_weights = load_cohorts(pp, ds_name, cohort_of)
    per_cohort_total = allocate({c: len(v) for c, v in by_cohort.items()}, args.n_patches, args.floor)
    shares = _parse_mix(args.mix)

    case_cache: dict = {}
    # subset_clicked is allocated GLOBALLY by multi-lesion supply: single-primary-tumor cohorts
    # have zero cases with >=2 lesion instances, so a plain per-cohort --mix split is unsatisfiable.
    per_cohort_counts, multi_counts = scenario_allocation(
        case_dir, by_cohort, per_cohort_total, shares, case_info, case_cache
    )

    bds = Blosc2Folder(case_dir, identifiers=sorted({cid for ids in by_cohort.values() for cid in ids}))
    label_cache = LabelCache()
    rejects: Counter = Counter()
    all_entries: list[dict] = []
    packed_rows: list[np.ndarray] = []
    # ONE generator per role for the whole build -- reproducibility relies on every run visiting
    # (cohort, scenario) pairs in the same order and consuming these streams identically.
    rngs = (
        np.random.default_rng(args.seed + 111_111),  # case draw
        np.random.default_rng(args.seed + 222_222),  # bbox
        np.random.default_rng(args.seed + 333_333),  # subset-k choice
        np.random.default_rng(args.seed),  # draw-1 displacement
        np.random.default_rng(args.seed + 777_777),  # draw-2 displacement
        np.random.default_rng(args.seed + 444_444),  # decoy
    )

    with nano_progress(args.n_patches, "build-valset") as advance:
        for cohort in sorted(per_cohort_total):
            ids = by_cohort[cohort]
            counts = per_cohort_counts[cohort]
            for scenario in SCENARIOS:
                filled = _fill_scenario(
                    scenario, bds, case_dir, ids, counts[scenario], args.max_tries, rngs, patch_size,
                    roi_cfg.sampling.propagated, label_cache, case_cache, rejects,
                )
                for entry, row, seg_crop in filled:
                    pp3 = torch.tensor(np.asarray(entry["clicks_zyx"], dtype=np.float32)).reshape(-1, 3)
                    ci_entry = {"pp": pp3, "pn": torch.zeros((0, 3)), "n_fp": entry["n_false_pos"]}
                    seg_t = torch.from_numpy(np.maximum(seg_crop, 0).astype(np.int16))
                    entry["click_inside"] = click_inside_flags([ci_entry], seg_t)[0]
                    if row is not None:
                        entry["subset_target_index"] = len(packed_rows)
                        packed_rows.append(row)
                    else:
                        entry["subset_target_index"] = -1
                    all_entries.append(entry)
                    advance(1)

    scenario_counts = {s: sum(1 for e in all_entries if e["scenario"] == s) for s in SCENARIOS}
    # Largest-remainder rounding is exact WITHIN a cohort but drifts globally: each cohort's
    # contribution to a scenario is off by <1 patch, so the total is off by <n_cohorts. Anything
    # larger is a real allocation bug. The realised counts land in the manifest header and the
    # composition table -- those are the numbers to quote, not the requested --mix.
    n_cohorts = len(per_cohort_total)
    for s in SCENARIOS:
        target = round(args.n_patches * shares[s])
        assert abs(scenario_counts[s] - target) <= n_cohorts, (
            f"scenario '{s}' landed on {scenario_counts[s]} patches, more than {n_cohorts} "
            f"(one per cohort) from the {target} implied by --mix {args.mix}. That is an "
            f"allocation bug, not rounding."
        )

    npz_path = _sidecar_path(args.out)
    packed = np.stack(packed_rows) if packed_rows else np.zeros((0, int(np.prod(patch_size)) // 8), dtype=np.uint8)
    np.savez_compressed(npz_path, packed=packed, shape=np.array(patch_size))

    header = {
        "schema": SCHEMA_VERSION, "dataset": ds_name, "plans": args.plans, "config_path": cfg_path,
        "seed": args.seed, "patch_size": [int(x) for x in patch_size], "small_lesion_max_vox": SMALL_LESION_MAX_VOX,
        "scenario_counts": scenario_counts, "cohort_weights": cohort_weights, "entries": all_entries,
        "subset_capable_cohorts": sorted(c for c, n in multi_counts.items() if n > 0),
        "cohort_multi_counts": multi_counts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(header, f)

    report_composition(all_entries, sorted(by_cohort))

    size_mb = os.path.getsize(npz_path) / 1e6
    cprint(f"wrote {args.out}  ({len(all_entries)} patches, {len(packed_rows)} subset targets, {size_mb:.1f} MB sidecar)")
    cprint(f"rejections: {dict(rejects)}")
    cprint(f"next: nanounet_train -d {args.dataset_id} --plans {args.plans} --val-manifest {args.out} ...")
