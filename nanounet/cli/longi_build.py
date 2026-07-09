"""Build a 2-channel raw longi dataset from register_longi output.

Each FU case becomes a 2-modality nnUNet-raw case: _0000 = FU CT, _0001 = warped BL CT (FU frame),
label = FU seg. `targetsTrFU` masks are instance-labeled (per-lesion ids, not {0,1}), so the seg is
binarized here (mirrors uclp-pro's binarize_mask) before writing labelsTr -- dataset.json only ever
declares {"background":0,"lesion":1}, and preprocessing's foreground-oversampling (class_locations)
only looks for id 1, so an un-binarized copy silently drops every other lesion instance from
oversampling. Warped-BL and FU union clicks (both carry the full lesion-id union, including
"disappeared" lesions with no FU ground truth) are copied to clicksTr/<case>.json and
clicksTrFU/<case>.json for later mapping into preprocessed voxels by nanounet.cli.longi_clicks.
The FU-frame warp gives _0000/_0001 an identical grid, so preprocessing's single nonzero crop
keeps them voxel-aligned (see design doc S12). Cases marked non-"ok" in --register-out's
clickfix_report.csv (unsanitized fallback points) are excluded, not patched.

  python -m nanounet.cli.longi_build --register-out <regout> \
      --template-dj <a dataset.json with labels/file_ending> \
      --out <nnUNet_raw>/DatasetNNN_longi
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil

import SimpleITK as sitk

from nanounet.common import cprint, nano_header


def _ok_cases(register_out: str) -> set[str] | None:
    """Cases marked "ok" in clickfix_report.csv, or None if no report (no filtering).

    "partial" cases carry at least one unsanitized/border-clamped fallback point (registration
    failure recovered via an out-of-range cog_propagated) that can land outside a case's cropped
    preprocessing bbox and crash longi_clicks's in-bounds assert. Same exclude-don't-patch policy
    as the rest of the click-fix pipeline: drop the case, don't silently clamp a bad point.
    """
    path = os.path.join(register_out, "clickfix_report.csv")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {row["case"] for row in csv.DictReader(f) if row["status"] == "ok"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-out", required=True)
    ap.add_argument("--template-dj", required=True, help="dataset.json to copy labels/file_ending from")
    ap.add_argument("--out", required=True, help="raw dataset dir to create, e.g. .../DatasetNNN_longi")
    args = ap.parse_args()

    nano_header("nanoUNet longi-build", color="green")

    fu_dir = os.path.join(args.register_out, "inputsTrFU")
    bl_dir = os.path.join(args.register_out, "inputsTrBL")
    seg_dir = os.path.join(args.register_out, "targetsTrFU")
    stems = sorted(os.path.basename(p)[: -len(".nii.gz")] for p in glob.glob(os.path.join(fu_dir, "*.nii.gz")))
    assert stems, fu_dir

    ok_cases = _ok_cases(args.register_out)
    if ok_cases is not None:
        excluded = [s for s in stems if s not in ok_cases]
        stems = [s for s in stems if s in ok_cases]
        if excluded:
            cprint(f"[yellow]excluding {len(excluded)} case(s) not marked ok in clickfix_report.csv: {excluded}[/yellow]")

    img_out = os.path.join(args.out, "imagesTr")
    lab_out = os.path.join(args.out, "labelsTr")
    clk_out = os.path.join(args.out, "clicksTr")
    clk_fu_out = os.path.join(args.out, "clicksTrFU")
    for d in (img_out, lab_out, clk_out, clk_fu_out):
        os.makedirs(d, exist_ok=True)

    clicks_bl_dir = os.path.join(args.register_out, "clicksBL")
    clicks_fu_dir = os.path.join(args.register_out, "clicksFU")

    n = 0
    for stem in stems:
        fu = os.path.join(fu_dir, f"{stem}.nii.gz")
        bl = os.path.join(bl_dir, f"{stem}.nii.gz")
        seg = os.path.join(seg_dir, f"{stem}.nii.gz")
        clk_bl = os.path.join(clicks_bl_dir, f"{stem}.json")
        clk_fu = os.path.join(clicks_fu_dir, f"{stem}.json")
        assert os.path.isfile(bl), bl
        assert os.path.isfile(seg), seg
        assert os.path.isfile(clk_bl), (
            f"missing {clk_bl}. Expected clicksBL/<case>.json (lesion-union clicks) under --register-out.\n"
            f"Fix: regenerate the click-fix data for {args.register_out}, or check --register-out points at "
            f"the unigradicon-registered dir with clicksBL/clicksFU/lesions siblings."
        )
        assert os.path.isfile(clk_fu), (
            f"missing {clk_fu}. Expected clicksFU/<case>.json (lesion-union clicks) under --register-out.\n"
            f"Fix: regenerate the click-fix data for {args.register_out}, or check --register-out points at "
            f"the unigradicon-registered dir with clicksBL/clicksFU/lesions siblings."
        )
        # FU-frame warp => identical sampling grid; assert so a bad warp fails loudly, not silently misaligned.
        assert sitk.ReadImage(fu).GetSize() == sitk.ReadImage(bl).GetSize(), stem
        # copy2's copystat/utime raises PermissionError on the CIFS mount; copyfile copies data only.
        shutil.copyfile(fu, os.path.join(img_out, f"{stem}_0000.nii.gz"))
        shutil.copyfile(bl, os.path.join(img_out, f"{stem}_0001.nii.gz"))
        seg_img = sitk.ReadImage(seg)
        seg_bin = sitk.Cast(seg_img > 0, sitk.sitkUInt8)  # instance ids -> {0,1}; see module docstring
        sitk.WriteImage(seg_bin, os.path.join(lab_out, f"{stem}.nii.gz"), True)
        shutil.copyfile(clk_bl, os.path.join(clk_out, f"{stem}.json"))
        shutil.copyfile(clk_fu, os.path.join(clk_fu_out, f"{stem}.json"))
        n += 1

    with open(args.template_dj, encoding="utf-8") as f:
        dj = json.load(f)
    dj["channel_names"] = {"0": "CT", "1": "CT"}
    dj.pop("modality", None)  # determine_num_input_channels prefers "modality"; drop it so channel_names (=2) wins
    # A merged template carries a per-case "dataset" block with absolute paths to OTHER datasets;
    # drop it so get_filenames_of_train_images_and_targets scans our imagesTr instead.
    dj.pop("dataset", None)
    dj["numTraining"] = n
    dj["file_ending"] = dj.get("file_ending", ".nii.gz")
    with open(os.path.join(args.out, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dj, f, indent=4)

    cprint(f"[green]wrote {n} cases -> {args.out}[/green]")
    cprint(f"[dim]next: nanounet_preprocess -d <id-for {args.out}>[/dim]")


if __name__ == "__main__":
    main()
