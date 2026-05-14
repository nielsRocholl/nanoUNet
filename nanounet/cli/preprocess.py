"""Fingerprint → plan (ResEnc) → preprocess 3d_fullres."""

from __future__ import annotations

import argparse

from nanounet.common import nano_header, nano_rule, sync_nnunet_env
from nanounet.plan.fingerprint import run_fingerprint
from nanounet.plan.planner import run_plan
from nanounet.plan.preprocess import run_preprocess

PATCH_VOL = {"small": 128, "medium": 192, "large": 256, "xlarge": 320}


def main() -> None:
    sync_nnunet_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
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
    ap.add_argument("--config-path", default=None)
    ap.add_argument("--skip-fingerprint", action="store_true")
    ap.add_argument("--skip-plan", action="store_true")
    args = ap.parse_args()
    did = args.dataset_id
    nano_header(f"nanoUNet preprocess  Dataset{did:03d}")
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
    run_preprocess(did, ident, args.num_processes, args.resume, args.config_path)


if __name__ == "__main__":
    main()
