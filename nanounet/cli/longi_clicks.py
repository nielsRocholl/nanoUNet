"""Map warped BL clicks (register xyz, FU frame) into preprocessed voxels; write
<case>_bl_clicks.json {bl_clicks_zyx, has_baseline} next to each preprocessed FU case.

Reuses cog_to_preprocessed (same forward geometry as centroid weighting). Crashes loudly if a mapped
click lands outside the preprocessed volume (wrong --cog-axis-order).

  python -m nanounet.cli.longi_clicks -d NNN --plans nnUNetPlans \
      --clicks-dir <nnUNet_raw>/DatasetNNN_longi/clicksTr
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nanounet.common import cprint, nano_header, nano_progress, preprocessed_dir
from nanounet.data.blosc2_dataset import Blosc2Folder, case_spatial_shape, load_case_properties
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.lesion_types import cog_to_preprocessed
from nanounet.plan.plans import Plans

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset_id", type=int, required=True)
ap.add_argument("--plans", required=True)
ap.add_argument("--clicks-dir", required=True, help="raw clicksTr dir with <case>.json warped BL clicks")
ap.add_argument("--cog-axis-order", choices=("xyz", "zyx"), default="xyz")
args = ap.parse_args()

ds = convert_id_to_dataset_name(args.dataset_id)
nano_header(f"nanoUNet longi-clicks  {ds}", color="green")
pp = preprocessed_dir()
pm = Plans(join(pp, ds, args.plans + ".json"))
tf = pm.transpose_forward
cm = pm.get_configuration("3d_fullres")
case_dir = join(pp, ds, cm.data_identifier)

ids = Blosc2Folder.get_identifiers(case_dir)
assert ids, case_dir
n_bl = 0
with nano_progress(len(ids), "longi-clicks") as advance:
    for cid in ids:
        props = load_case_properties(case_dir, cid)
        pre_shape = case_spatial_shape(case_dir, cid)
        bbox = props["bbox_used_for_cropping"]
        shape_ac = props["shape_after_cropping_and_before_resampling"]
        clk_path = join(args.clicks_dir, cid + ".json")
        zyx: list[list[int]] = []
        if isfile(clk_path):
            with open(clk_path, encoding="utf-8") as f:
                pts = [p["point"] for p in json.load(f)["points"]]
            for xyz in pts:
                m = cog_to_preprocessed(np.asarray(xyz, dtype=float), tf, bbox, shape_ac, pre_shape, args.cog_axis_order)
                mz = [int(round(m[i])) for i in range(3)]
                for i in range(3):
                    assert 0 <= mz[i] < pre_shape[i], (cid, xyz, mz, pre_shape)
                zyx.append(mz)
        with open(join(case_dir, cid + "_bl_clicks.json"), "w", encoding="utf-8") as f:
            json.dump({"bl_clicks_zyx": zyx, "has_baseline": bool(zyx)}, f)
        n_bl += 1 if zyx else 0
        advance(1)

cprint(f"cases: {len(ids)}  with baseline clicks: {n_bl}")
