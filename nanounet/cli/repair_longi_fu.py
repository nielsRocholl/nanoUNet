"""Fast repair for register_longi cases stuck mid-write by the CIFS copy2 bug.

output.py used shutil.copy2 to copy FU/meta files, whose copystat/utime call raises
PermissionError on the CIFS mount. Cases hit this after the warped BL was already written
(the expensive elastix step succeeded) but before the FU json/mask/meta were copied.

fu_out (the FU click coords written to inputsTrFU/{stem}.json) comes straight from the meta
CSV -- it does not depend on the registration result -- so these cases can be repaired by
re-deriving/copying the missing files, without re-running elastix.

  python -m nanounet.cli.repair_longi_fu --data-root <DIR> --out <DIR>
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

from nanounet.register.landmarks import read_pairs
from nanounet.register.output import write_points

ap = argparse.ArgumentParser(description="Repair FU/meta sidecars for cases whose BL registration already succeeded")
ap.add_argument("--data-root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def _paths(out: str, stem: str, pid: str) -> dict[str, str]:
    return {
        "bl_img": f"{out}/inputsTrBL/{stem}.nii.gz",
        "bl_pts": f"{out}/inputsTrBL/{stem}.json",
        "bl_seg": f"{out}/targetsTrBL/{stem}.nii.gz",
        "fu_img": f"{out}/inputsTrFU/{stem}.nii.gz",
        "fu_pts": f"{out}/inputsTrFU/{stem}.json",
        "fu_seg": f"{out}/targetsTrFU/{stem}.nii.gz",
        "meta": f"{out}/meta/{pid}.csv",
    }


def _repair_case(data_root: str, out: str, pid: str, idx: str) -> str:
    stem = f"{pid}_{idx}"
    p = _paths(out, stem, pid)
    if all(os.path.isfile(v) for v in p.values()):
        return "complete"
    if not (os.path.isfile(p["bl_img"]) and os.path.isfile(p["bl_pts"])
            and os.path.isfile(p["bl_seg"]) and os.path.isfile(p["fu_img"])):
        return "needs_full_registration"

    meta_csv = os.path.join(data_root, "meta", f"{pid}.csv")
    has_csv = os.path.isfile(meta_csv)
    fu_out = None
    if has_csv:
        pairs = read_pairs(meta_csv)
        fu_out = [(lid, cf.tolist()) for lid, (_cb, cf) in pairs.items() if cf is not None]

    if not os.path.isfile(p["fu_pts"]):
        os.makedirs(os.path.dirname(p["fu_pts"]), exist_ok=True)
        if fu_out is not None:
            write_points(fu_out, p["fu_pts"])
        else:
            src = os.path.join(data_root, "inputsTrFU", f"{stem}.json")
            if os.path.isfile(src):
                shutil.copyfile(src, p["fu_pts"])

    if not os.path.isfile(p["fu_seg"]):
        src = os.path.join(data_root, "targetsTrFU", f"{stem}.nii.gz")
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(p["fu_seg"]), exist_ok=True)
            shutil.copyfile(src, p["fu_seg"])

    if not os.path.isfile(p["meta"]) and has_csv:
        os.makedirs(os.path.dirname(p["meta"]), exist_ok=True)
        shutil.copyfile(meta_csv, p["meta"])

    return "repaired" if all(os.path.isfile(v) for v in p.values()) else "still_incomplete"


counts: dict[str, int] = {}
for bl in sorted(glob.glob(os.path.join(args.out, "inputsTrBL", "*.nii.gz"))):
    stem = os.path.basename(bl)[: -len(".nii.gz")]
    pid, idx = stem.rsplit("_", 1)
    status = _repair_case(args.data_root, args.out, pid, idx)
    counts[status] = counts.get(status, 0) + 1
    if status not in ("complete", "repaired"):
        print(f"{stem}\t{status}")

print(counts)
