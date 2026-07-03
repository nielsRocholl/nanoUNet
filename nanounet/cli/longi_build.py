"""Build a 2-channel raw longi dataset from register_longi output.

Each FU case becomes a 2-modality nnUNet-raw case: _0000 = FU CT, _0001 = warped BL CT (FU frame),
label = FU seg. Warped BL clicks are copied to clicksTr/<case>.json for later mapping into
preprocessed voxels by nanounet.cli.longi_clicks. The FU-frame warp gives _0000/_0001 an identical
grid, so preprocessing's single nonzero crop keeps them voxel-aligned (see design doc S12).

  python -m nanounet.cli.longi_build --register-out <regout> \
      --template-dj <a dataset.json with labels/file_ending> \
      --out <nnUNet_raw>/DatasetNNN_longi
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import SimpleITK as sitk

from nanounet.common import cprint

ap = argparse.ArgumentParser()
ap.add_argument("--register-out", required=True)
ap.add_argument("--template-dj", required=True, help="dataset.json to copy labels/file_ending from")
ap.add_argument("--out", required=True, help="raw dataset dir to create, e.g. .../DatasetNNN_longi")
args = ap.parse_args()

fu_dir = os.path.join(args.register_out, "inputsTrFU")
bl_dir = os.path.join(args.register_out, "inputsTrBL")
seg_dir = os.path.join(args.register_out, "targetsTrFU")
stems = sorted(os.path.basename(p)[: -len(".nii.gz")] for p in glob.glob(os.path.join(fu_dir, "*.nii.gz")))
assert stems, fu_dir

img_out = os.path.join(args.out, "imagesTr")
lab_out = os.path.join(args.out, "labelsTr")
clk_out = os.path.join(args.out, "clicksTr")
for d in (img_out, lab_out, clk_out):
    os.makedirs(d, exist_ok=True)

n = 0
for stem in stems:
    fu = os.path.join(fu_dir, f"{stem}.nii.gz")
    bl = os.path.join(bl_dir, f"{stem}.nii.gz")
    seg = os.path.join(seg_dir, f"{stem}.nii.gz")
    clk = os.path.join(bl_dir, f"{stem}.json")
    assert os.path.isfile(bl), bl
    assert os.path.isfile(seg), seg
    assert os.path.isfile(clk), clk
    # FU-frame warp => identical sampling grid; assert so a bad warp fails loudly, not silently misaligned.
    assert sitk.ReadImage(fu).GetSize() == sitk.ReadImage(bl).GetSize(), stem
    # copy2's copystat/utime raises PermissionError on the CIFS mount; copyfile copies data only.
    shutil.copyfile(fu, os.path.join(img_out, f"{stem}_0000.nii.gz"))
    shutil.copyfile(bl, os.path.join(img_out, f"{stem}_0001.nii.gz"))
    shutil.copyfile(seg, os.path.join(lab_out, f"{stem}.nii.gz"))
    shutil.copyfile(clk, os.path.join(clk_out, f"{stem}.json"))
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
