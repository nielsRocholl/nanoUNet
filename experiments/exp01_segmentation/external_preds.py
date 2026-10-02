"""Ingest of prediction folders from systems that are not run in this repo (ULS+, nnInteractive).

Contract (also in experiments/README.md): exp01 writes one click file per (scenario, case) to `artifacts/prompts/<scenario>/<case_id>.json`
(`{"points": [{"name": "<lesion id>|decoy", "point": [x, y, z]}]}`, native voxels of the case image, S3 = empty list). The owner runs each external
system on those clicks and stores its binary masks as `DIR/<scenario>/<case_id>.nii.gz`: one folder per scenario because the same case is
segmented once per scenario, nnU-Net-style file names inside. Any non-zero voxel is foreground; the mask must sit on the case image's grid.
A system that cannot run without a click (both of them) writes an empty mask for S3. The prompt-drop twin (S1_noprompt) does not exist for
external systems. A missing or misplaced file is a startup error, never a silent drop: a partial folder is rescored after it is completed.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from experiments.common import problem
from experiments.segment import METHOD_OVERLAP_KEY

EXTERNAL_SCENARIOS = ("S1", "S2", "S3", "S4")
EXTERNAL_NAMES = tuple(k for k in METHOD_OVERLAP_KEY if k != "nanounet")  # keys of the manifest's overlap flags: nninteractive, uls_plus
LAYOUT_FIX = "python -m experiments.exp01_segmentation.run --emit-prompts-only ... then run the system on artifacts/prompts/<scenario>/<case>.json into DIR/<scenario>/<case>.nii.gz"


def parse_external(specs: list[str]) -> tuple[dict[str, Path], list[str]]:
    """`NAME=DIR` strings to {name: dir}; problems for a malformed spec, an unknown name, a repeated name or a missing dir."""
    dirs, problems = {}, []
    for spec in specs:
        name, _, folder = spec.partition("=")
        if name not in EXTERNAL_NAMES or not folder:
            problems.append(problem(f"--external {spec!r} is not NAME=DIR with a known system", f"NAME in {list(EXTERNAL_NAMES)} (the manifest's overlap keys), e.g. nninteractive=/data/preds/nninteractive",
                                    "rerun with --external nninteractive=<dir> uls_plus=<dir>"))
        elif name in dirs:
            problems.append(problem(f"--external names {name} twice", "one folder per system", f"keep one {name}=DIR"))
        elif not Path(folder).is_dir():
            problems.append(problem(f"--external {name}: folder {folder} does not exist", "a folder of <scenario>/<case_id>.nii.gz", LAYOUT_FIX))
        else:
            dirs[name] = Path(folder)
    return dirs, problems


def external_path(folder: Path, scenario: str, case_id: str) -> Path:
    return folder / scenario / f"{case_id}.nii.gz"


def inventory_problems(name: str, folder: Path, needed: list[tuple[dict, tuple[int, int, int], tuple[str, ...]]]) -> list[str]:
    """Missing files and wrong-grid masks of one system, one problem per (kind, scenario) with the first offenders named.
    `needed` = [(case, expected (z, y, x) size, scenarios)] for the cases the system is scored on."""
    missing, wrong = {}, {}
    for case, size, scenarios in needed:
        for sc in scenarios:
            f = external_path(folder, sc, case["case_id"])
            if not f.is_file():
                missing.setdefault(sc, []).append(f)
                continue
            rd = sitk.ImageFileReader()
            rd.SetFileName(str(f))
            rd.ReadImageInformation()
            got = tuple(int(v) for v in rd.GetSize()[::-1])
            if got != size:
                wrong.setdefault(sc, []).append(f"{f} has size {got}, expected {size}")
    out = [problem(f"{name}: {len(fs)} prediction(s) missing for {sc}, e.g. {fs[0]}", f"{folder}/{sc}/<case_id>.nii.gz for every case scored", LAYOUT_FIX) for sc, fs in missing.items()]
    return out + [problem(f"{name}: {len(msgs)} mask(s) on the wrong grid for {sc}, e.g. {msgs[0]}", "masks on the case image's grid (z, y, x size equal)", "resample the predictions to the case image and rerun the system") for sc, msgs in wrong.items()]
