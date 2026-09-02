"""Fingerprint -> plan (ResEnc) -> preprocess 3d_fullres -> balanced splits -> cohorts -> valset.

Orchestration only (R13): each step's logic lives in nanounet/plan/ (or, for the valset step,
nanounet/cli/build_valset.py, invoked in-process rather than duplicated). The output is a
training-ready preprocessed dataset -- nanounet_train needs nothing else."""

from __future__ import annotations

import argparse
import json
import sys

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nanounet.common import cprint, nano_header, nano_rule, preprocessed_dir, raw_dir
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.plan.cohorts import run_cohorts
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.plans import Plans
from nanounet.plan.prep.fingerprint import run_fingerprint
from nanounet.plan.prep.merge import build_merged_raw
from nanounet.plan.resenc.planner import run_plan
from nanounet.plan.prep.preprocess import run_preprocess
from nanounet.plan.splits import make_balanced_split

PATCH_VOL = {"small": 128, "medium": 192, "large": 256, "xlarge": 320}


def _run_build_valset(did: int, ident: str, config_path: str, out_path: str, n_patches: int) -> None:
    """In-process call into nanounet_build_valset's own main(), which owns its argparse and all
    the patch-sampling logic -- reused as-is rather than duplicated here."""
    from nanounet.cli import build_valset

    old_argv = sys.argv
    sys.argv = [
        "nanounet_build_valset",
        "-d", str(did),
        "--plans", ident,
        "--config", config_path,
        "--out", out_path,
        "--n-patches", str(n_patches),
    ]
    try:
        build_valset.main()
    finally:
        sys.argv = old_argv


def _write_splits_and_cohorts(did: int, ident: str, val_frac: float, seed: int) -> str:
    pp = preprocessed_dir()
    ds_name = convert_id_to_dataset_name(did)
    pre = join(pp, ds_name)
    pm = Plans(join(pre, ident + ".json"))
    cm = pm.get_configuration("3d_fullres")
    case_dir = join(pre, cm.data_identifier)
    all_ids = Blosc2Folder.get_identifiers(case_dir)
    dj = load_json(join(raw_dir(), ds_name, "dataset.json"))
    ntr = dj.get("numTraining")
    ids = all_ids[: int(ntr)] if ntr is not None else list(all_ids)

    splits = make_balanced_split(ids, val_frac, seed)
    splits_path = join(pre, "splits_final.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)
    cprint(f"[bold green]✓ splits[/bold green] → {splits_path}  ({len(splits[0]['train'])} train / {len(splits[0]['val'])} val)")

    cohorts_path = run_cohorts(did, pre)
    cprint(f"[bold green]✓ cohorts[/bold green] → {cohorts_path}")
    return pre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, nargs="+", required=True)
    ap.add_argument("--merged-id", type=int, default=999)
    ap.add_argument("--merged-name", default="Merged")
    ap.add_argument("--planner", default="nnUNetPlannerResEncL")
    ap.add_argument("-np", "--num_processes", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu-memory-gb", type=float, default=None)
    ap.add_argument(
        "--patch-vol",
        choices=tuple(PATCH_VOL),
        default="large",
        help="target patch volume edge (isotropic equivalent before aniso split); large=256 (nnU-Net default)",
    )
    ap.add_argument("--plans-name", default=None)
    ap.add_argument("--skip-fingerprint", action="store_true")
    ap.add_argument("--skip-plan", action="store_true")
    ap.add_argument(
        "--sidecars-only",
        action="store_true",
        help="regenerate *_centroids.json sidecars only; never touches .b2nd/plans/gt_segmentations",
    )
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of each source cohort held out for val")
    ap.add_argument("--split-seed", type=int, default=12345, help="seed for the balanced train/val split")
    ap.add_argument("--no-splits", action="store_true", help="skip splits_final.json + cohorts.json generation")
    ap.add_argument(
        "--valset-config",
        default=None,
        help="roi config path; when set, also builds a fixed valset_<n>.json manifest via nanounet_build_valset",
    )
    ap.add_argument("--valset-n", type=int, default=1500, help="patch count for --valset-config (default 1500)")
    args = ap.parse_args()
    if len(args.dataset_id) == 1:
        did = args.dataset_id[0]
        nano_header(f"nanoUNet preprocess  Dataset{did:03d}")
    else:
        build_merged_raw(args.dataset_id, args.merged_id, args.merged_name)
        did = args.merged_id
        nano_header(
            "nanoUNet preprocess  merge "
            f"{','.join(str(i) for i in args.dataset_id)} -> Dataset{did:03d}_{args.merged_name}"
        )
    if args.sidecars_only:
        if not args.plans_name:
            ap.error("--sidecars-only needs --plans-name (identifies the existing plans json to read)")
        run_preprocess(did, args.plans_name, args.num_processes, False, sidecars_only=True)
        return
    if not args.skip_fingerprint:
        run_fingerprint(did, args.num_processes)
        nano_rule()
    if args.skip_plan:
        if not args.plans_name:
            ap.error("--skip-plan needs --plans-name (e.g. nnUNetResEncUNetTinyPlans)")
        ident = args.plans_name
    else:
        ident = run_plan(
            did,
            args.planner,
            args.gpu_memory_gb,
            None,
            args.plans_name,
            patch_edge=PATCH_VOL[args.patch_vol],
        )
        nano_rule()
    run_preprocess(did, ident, args.num_processes, args.resume)

    artifacts = [f"preprocessed cases → {join(preprocessed_dir(), convert_id_to_dataset_name(did))}"]
    if not args.no_splits:
        nano_rule()
        pre = _write_splits_and_cohorts(did, ident, args.val_frac, args.split_seed)
        artifacts += [f"splits_final.json → {join(pre, 'splits_final.json')}", f"cohorts.json → {join(pre, 'cohorts.json')}"]
        if args.valset_config:
            val_out = join(pre, f"valset_{args.valset_n}.json")
            _run_build_valset(did, ident, args.valset_config, val_out, args.valset_n)
            artifacts.append(f"valset manifest → {val_out}")

    nano_rule()
    cprint("[bold cyan]preprocess complete[/bold cyan]")
    for a in artifacts:
        cprint(f"  [dim]-[/dim] {a}")
    next_cmd = f"nanounet_train -d {did} -f 0 --plans {ident} --config configs/default.json"
    if not args.no_splits and args.valset_config:
        next_cmd += f" --val-manifest {join(preprocessed_dir(), convert_id_to_dataset_name(did), f'valset_{args.valset_n}.json')}"
    cprint(f"next: {next_cmd}")


if __name__ == "__main__":
    main()
