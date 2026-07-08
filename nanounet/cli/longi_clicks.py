"""Map warped-BL and FU union clicks (register xyz, FU frame) into preprocessed voxels; write
<case>_bl_clicks.json {bl_clicks_zyx, has_baseline} and, if --clicks-fu-dir is given,
<case>_fu_clicks.json {fu_clicks_zyx, fu_topology} next to each preprocessed FU case.

Both click sets live in the same FU-image voxel grid (BL clicks are warped into it, FU clicks are
native to it), so both use the identical cog_to_preprocessed geometry. Crashes loudly if a mapped
click lands outside the preprocessed volume (wrong --cog-axis-order).

  python -m nanounet.cli.longi_clicks -d NNN --plans nnUNetPlans \
      --clicks-dir <nnUNet_raw>/DatasetNNN_longi/clicksTr \
      --clicks-fu-dir <nnUNet_raw>/DatasetNNN_longi/clicksTrFU
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


def _map_points(clk_path: str, cid: str, tf, bbox, shape_ac, pre_shape, axis_order: str):
    """Read a clicksXX/<case>.json and forward-map each point into preprocessed voxel zyx."""
    with open(clk_path, encoding="utf-8") as f:
        pts = json.load(f)["points"]
    zyx: list[list[int]] = []
    topo: list[str] = []
    for p in pts:
        m = cog_to_preprocessed(np.asarray(p["point"], dtype=float), tf, bbox, shape_ac, pre_shape, axis_order)
        mz = [int(round(m[i])) for i in range(3)]
        for i in range(3):
            assert 0 <= mz[i] < pre_shape[i], (cid, p["point"], mz, pre_shape)
        zyx.append(mz)
        topo.append(p["topology"])
    return zyx, topo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--clicks-dir", required=True, help="raw clicksTr dir with <case>.json warped BL clicks")
    ap.add_argument("--clicks-fu-dir", default=None, help="raw clicksTrFU dir with <case>.json FU union clicks")
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
    n_fu = 0
    with nano_progress(len(ids), "longi-clicks") as advance:
        for cid in ids:
            props = load_case_properties(case_dir, cid)
            pre_shape = case_spatial_shape(case_dir, cid)
            bbox = props["bbox_used_for_cropping"]
            shape_ac = props["shape_after_cropping_and_before_resampling"]

            clk_path = join(args.clicks_dir, cid + ".json")
            zyx: list[list[int]] = []
            if isfile(clk_path):
                zyx, _ = _map_points(clk_path, cid, tf, bbox, shape_ac, pre_shape, args.cog_axis_order)
            with open(join(case_dir, cid + "_bl_clicks.json"), "w", encoding="utf-8") as f:
                json.dump({"bl_clicks_zyx": zyx, "has_baseline": bool(zyx)}, f)
            n_bl += 1 if zyx else 0

            if args.clicks_fu_dir is not None:
                clk_fu_path = join(args.clicks_fu_dir, cid + ".json")
                assert isfile(clk_fu_path), (
                    f"missing {clk_fu_path}. --clicks-fu-dir was given but has no entry for case {cid}.\n"
                    f"Fix: check --clicks-fu-dir points at the clicksFU (or clicksTrFU) dir matching this dataset."
                )
                fu_zyx, fu_topo = _map_points(clk_fu_path, cid, tf, bbox, shape_ac, pre_shape, args.cog_axis_order)
                with open(join(case_dir, cid + "_fu_clicks.json"), "w", encoding="utf-8") as f:
                    json.dump({"fu_clicks_zyx": fu_zyx, "fu_topology": fu_topo}, f)
                n_fu += 1 if fu_zyx else 0
            advance(1)

    cprint(f"[dim]next: nanounet_train -d {args.dataset_id} --plans {args.plans} --longi …[/dim]")
    cprint(f"cases: {len(ids)}  with baseline clicks: {n_bl}" + (f"  with FU clicks: {n_fu}" if args.clicks_fu_dir else ""))


if __name__ == "__main__":
    main()
