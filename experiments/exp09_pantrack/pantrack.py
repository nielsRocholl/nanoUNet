"""PanTrack dataset adapter for exp09: scan pairs, identity ground truth, lesion types, the validation table, the synthesised
Longitudinal-CT layout and the baselines' per-pair inputs.

Raw layout (`/nnunet_data/raw/PanTrack`): `images/<scan>_0000.nii.gz`, `labels/<scan>.nii.gz` (uint8 instance masks), `totalseg/<scan>.nii.gz`,
`patients.json` (patient -> scans in time order), `tracking.json` (patient -> one dict per consecutive pair: label id -> {bl_point,
fu_point_prop, fu_point, img_bl, img_fu, merged_lesions}), `organ_annotations.json` (scan -> annotation id -> {organ, location, ...}).

Facts this module relies on and CHECKS (`validate`, every run, all problems at once with `Fix:`):
- Label NIfTIs are uint8, so annotation ids wrap modulo 256 (annotation 307 = label value 51). `tracking.json` keys and the organ join use
  the wrapped id. Two annotations of one scan that wrap to the same label are a collision and fail the validation.
- Coordinate frame: `bl_point` and `fu_point` are the label centroids in VOXEL x,y,z of the BL / FU scan's own grid (not mm); `fu_point_prop`
  is the uniGradICON-propagated BL point in the FU voxel grid. The check compares them with the centroids computed from the masks.
- Identity GT = the same label id in BL and FU (UNCHANGED); an id present only in BL is DISAPPEARING (its `fu_point` is NaN/null), an id
  present only in FU is NEWLYAPPEARING; no merges exist (`merged_lesions` always has one entry), no splits.
- Lesion type for the matcher comes from `organ_annotations.json` (join by `annotation id % 256 == label id`): liver -> `Liver`,
  lymph node -> `Lymph node`, pancreas -> `Others` (the matcher's vocabulary has no pancreas). It is cross-checked against the TotalSeg
  derivation (organ with the largest voxel overlap among liver=5 / pancreas=7); lymph nodes are no TotalSeg class and are only reported.

Synthesised layout (`build_layout`, under the run's `artifacts/layout/`, because `lesionglue`'s graph builder and `pipeline.py` read the
Longitudinal-CT layout): one pseudo-patient `<patient>-p<k>` per consecutive pair k, BL = `inputsTrBL|targetsTrBL/<pid>_00.nii.gz` (symlinks
to the raw image/label), FU = `.../<pid>_01.nii.gz`, `meta/<pid>.csv` with img_id_bl 0 and img_id_fu 1 (lesion_id, topology_class, cog_bl,
cog_propagated = fu_point_prop, cog_fu, lesion_type, merged_into, linking_unclear=False; a NEWLYAPPEARING row's cog_fu is the FU label
centroid because `tracking.json` only lists BL lesions), `inputsTrBL/<pid>_00.json` (true BL points, prompts of setting C) and
`inputsTrFU/<pid>_01.json` (propagated points, the FU prompts and the propagated-position file). Voxel x,y,z like Longitudinal-CT.

Baseline inputs (`baseline_inputs`): the same structure as `exp04_baselines/inputs.py` (BL masks translated by `fu_point_prop - bl_point`
in mm, re-gridded to the FU grid; FU centroid; 9x9x9 appearance patches), built from the raw files.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from experiments.common import problem
from experiments.exp04_baselines.diveroli import overlap_table
from experiments.pipeline import Pair

ROOT = Path("/nnunet_data/raw/PanTrack")
ORGAN_TYPE = {"liver": "Liver", "lymph node": "Lymph node", "pancreas": "Others"}  # the vocabulary has no pancreas
TOTALSEG = {"liver": 5, "pancreas": 7}  # TotalSegmentator label values (checked on scan PanTrack_001_20220111: lesion 51 overlaps value 7)
FRAME_TOL = 0.01  # voxels: bl_point / fu_point are the exact mask centroids (float rounding only)
PATCH = 4  # half width of the cubic appearance patch in voxels (9 x 9 x 9), as exp04


def load_index(root: Path) -> dict:
    return {n: json.loads((root / f"{n}.json").read_text()) for n in ("patients", "tracking", "organ_annotations")}


def pair_list(idx: dict, patients: list[str]) -> list[dict]:
    """One record per consecutive pair of the given patients: pseudo patient id `<patient>-p<k>`, BL/FU scan, the tracking dict."""
    return [{"pid": f"{p}-p{k}", "patient": p, "k": k, "bl": s[k], "fu": s[k + 1], "track": idx["tracking"][p][k]}
            for p in patients for s in [idx["patients"][p]] for k in range(len(s) - 1)]


def vanishing(entry: dict) -> bool:
    return entry["fu_point"] is None or (isinstance(entry["fu_point"], float) and math.isnan(entry["fu_point"]))


def read_scan(root: Path, scan: str) -> tuple[np.ndarray, list, np.ndarray]:
    """(label array zyx, find_objects boxes, spacing xyz). The image is held in a variable before GetArrayFromImage (CIFS/SimpleITK rule)."""
    itk = sitk.ReadImage(str(root / "labels" / f"{scan}.nii.gz"))
    arr = sitk.GetArrayFromImage(itk)
    return arr, ndi.find_objects(arr), np.asarray(itk.GetSpacing(), dtype=np.float64)


def voxels_xyz(arr: np.ndarray, boxes: list, lid: int) -> np.ndarray | None:
    """(n, 3) int voxel x,y,z of label `lid`, or None if the label is absent."""
    if lid - 1 >= len(boxes) or boxes[lid - 1] is None:
        return None
    sl = boxes[lid - 1]
    return (np.argwhere(arr[sl] == lid) + np.array([s.start for s in sl]))[:, ::-1]


def scan_stats(root: Path, scan: str) -> dict:
    """{"spacing", "instances": {label id: {"n", "centroid" (voxel xyz), "ts": {organ: voxels overlapping that TotalSeg organ}}}} of one scan."""
    arr, boxes, sp = read_scan(root, scan)
    itk = sitk.ReadImage(str(root / "totalseg" / f"{scan}.nii.gz"))
    ts = sitk.GetArrayFromImage(itk)
    out = {}
    for lid in range(1, len(boxes) + 1):
        if boxes[lid - 1] is None:
            continue
        m = arr[boxes[lid - 1]] == lid
        vals = ts[boxes[lid - 1]][m]
        out[lid] = {"n": int(m.sum()), "centroid": voxels_xyz(arr, boxes, lid).mean(0).tolist(), "ts": {o: int((vals == v).sum()) for o, v in TOTALSEG.items()}}
    return {"spacing": sp.tolist(), "instances": out}


def organs_of(idx: dict, scan: str) -> dict[int, list[str]]:
    """{label id: [organ of every annotation wrapping to it]} for one scan (more than one entry = collision)."""
    out: dict[int, list[str]] = {}
    for ann, v in idx["organ_annotations"].get(scan, {}).items():
        out.setdefault(int(ann) % 256, []).append(v["organ"])
    return out


def derived_organ(ts: dict) -> str:
    """Organ with the largest TotalSeg overlap among liver / pancreas, `none` if the instance touches neither."""
    return "none" if not any(ts.values()) else max(ts, key=lambda o: ts[o])


def validate(idx: dict, pairs: list[dict], stats: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """(one row per scan, problems). Rows: counts of instances / annotations / per-organ counts (annotation vs TotalSeg), frame deviation in voxels
    (bl_point and fu_point vs mask centroids), ids missing in either direction. Problems carry `Fix:` (see `problem`)."""
    rows, problems = [], []
    frame: dict[str, list[float]] = {}
    for p in pairs:
        for lid, e in p["track"].items():
            lid = int(lid)
            bl_c, fu = stats[p["bl"]]["instances"].get(lid), stats[p["fu"]]["instances"].get(lid)
            if bl_c is None:
                problems.append(problem(f"{p['pid']}: tracking.json lesion {lid} is not a label of {p['bl']}", "every tracked lesion id is a BL label value",
                                        "check the `% 256` wrap of tracking keys against labels/"))
                continue
            frame.setdefault(p["bl"], []).append(float(np.abs(np.array(e["bl_point"]) - np.array(bl_c["centroid"])).max()))
            if vanishing(e) != (fu is None):
                problems.append(problem(f"{p['pid']}: lesion {lid} fu_point is {'null/NaN' if vanishing(e) else 'set'} but the label is {'absent' if fu is None else 'present'} in {p['fu']}",
                                        "vanishing (fu_point NaN) iff the label id is absent in FU", "inspect tracking.json and labels/ of this pair"))
            elif fu is not None:
                frame.setdefault(p["fu"], []).append(float(np.abs(np.array(e["fu_point"]) - np.array(fu["centroid"])).max()))
    for scan in sorted({s for p in pairs for s in (p["bl"], p["fu"])}):
        inst, org = stats[scan]["instances"], organs_of(idx, scan)
        coll = sorted(k for k, v in org.items() if len(v) > 1)
        undecided = sorted(set(inst) - set(org))
        ghost = sorted(set(org) - set(inst))
        if coll:
            problems.append(problem(f"{scan}: annotation ids {coll} collide modulo 256", "one annotation per label value", "resolve the collision in organ_annotations.json by hand"))
        if undecided:
            problems.append(problem(f"{scan}: instances {undecided} have no entry in organ_annotations.json (organ undecidable)", "every mask instance has an organ annotation (id % 256)",
                                    "add the annotation or exclude the scan; nothing is guessed"))
        if ghost:
            problems.append(problem(f"{scan}: organ annotations {ghost} have no mask instance", "every annotation has a label value", "check the modulo-256 join"))
        ann_count = {o: sum(v.count(o) for v in org.values()) for o in ORGAN_TYPE}
        ts_count = {o: sum(1 for i in inst.values() if derived_organ(i["ts"]) == o) for o in list(TOTALSEG) + ["none"]}
        agree = all(ts_count[o] == ann_count[o] for o in TOTALSEG) and ann_count["lymph node"] == ts_count["none"]
        rows.append({"scan": scan, "n_instances": len(inst), "n_annotations": sum(ann_count.values()), "collisions": len(coll), "undecided": len(undecided),
                     **{f"ann_{o.replace(' ', '_')}": c for o, c in ann_count.items()}, **{f"totalseg_{o}": c for o, c in ts_count.items()}, "counts_agree": agree,
                     "frame_dev_voxels": max(frame.get(scan, [0.0]))})
        if max(frame.get(scan, [0.0])) > FRAME_TOL:
            problems.append(problem(f"{scan}: bl_point/fu_point differ from the mask centroids by {max(frame[scan]):.3f} voxels (tolerance {FRAME_TOL})",
                                    "bl_point and fu_point = label centroids in voxel x,y,z", "re-verify the coordinate frame before using the points"))
    return rows, problems


def lesion_organ(idx: dict, scan: str, lid: int) -> str:
    orgs = organs_of(idx, scan)[lid]
    return orgs[0]


def pair_rows(idx: dict, p: dict, stats: dict) -> list[dict]:
    """Meta-CSV rows of one pair (see module docstring): UNCHANGED / DISAPPEARING from tracking.json, NEWLYAPPEARING = FU ids not tracked."""
    xyz = lambda v: " ".join(f"{float(c):.6f}" for c in v)
    rows, tracked = [], {int(k) for k in p["track"]}
    base = {"img_id_bl": 0, "img_id_fu": 1, "merged_into": "", "linking_unclear": "False", "cog_backpropagated": ""}
    for lid, e in sorted((int(k), v) for k, v in p["track"].items()):
        gone = vanishing(e)
        rows.append({**base, "lesion_id": lid, "topology_class": "DISAPPEARING" if gone else "UNCHANGED", "cog_bl": xyz(e["bl_point"]), "cog_propagated": xyz(e["fu_point_prop"]),
                     "cog_fu": "" if gone else xyz(e["fu_point"]), "lesion_type": ORGAN_TYPE[lesion_organ(idx, p["bl"], lid)], "organ": lesion_organ(idx, p["bl"], lid)})
    for lid in sorted(set(stats[p["fu"]]["instances"]) - tracked):
        rows.append({**base, "lesion_id": lid, "topology_class": "NEWLYAPPEARING", "cog_bl": "", "cog_propagated": "", "cog_fu": xyz(stats[p["fu"]]["instances"][lid]["centroid"]),
                     "lesion_type": ORGAN_TYPE[lesion_organ(idx, p["fu"], lid)], "organ": lesion_organ(idx, p["fu"], lid)})
    return rows


def write_clicks(path: Path, points: dict[int, list[float]]) -> None:
    path.write_text(json.dumps({"name": "Points of interest", "points": [{"name": str(k), "point": [float(c) for c in v]} for k, v in sorted(points.items())],
                                "type": "Multiple points", "version": {"major": 1, "minor": 0}}, indent=1))


def build_layout(root: Path, idx: dict, pairs: list[dict], stats: dict, layout: Path) -> dict[str, Pair]:
    """Write the synthesised Longitudinal-CT layout (symlinks + meta CSV + click JSONs) and return {pseudo patient: Pair}."""
    cols = ["lesion_id", "cog_bl", "cog_backpropagated", "img_id_bl", "cog_propagated", "cog_fu", "img_id_fu", "lesion_type", "topology_class", "merged_into", "linking_unclear", "organ"]
    for d in ("inputsTrBL", "inputsTrFU", "targetsTrBL", "targetsTrFU", "meta"):
        (layout / d).mkdir(parents=True, exist_ok=True)
    out = {}
    for p in pairs:
        pid, links = p["pid"], {}
        for side, scan, i in (("BL", p["bl"], 0), ("FU", p["fu"], 1)):
            for d, src in ((f"inputsTr{side}", root / "images" / f"{scan}_0000.nii.gz"), (f"targetsTr{side}", root / "labels" / f"{scan}.nii.gz")):
                dst = layout / d / f"{pid}_{i:02d}.nii.gz"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src, dst)
                links[d] = dst
        write_clicks(layout / "inputsTrBL" / f"{pid}_00.json", {int(k): e["bl_point"] for k, e in p["track"].items()})
        write_clicks(layout / "inputsTrFU" / f"{pid}_01.json", {int(k): e["fu_point_prop"] for k, e in p["track"].items()})
        meta = layout / "meta" / f"{pid}.csv"
        with open(meta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(pair_rows(idx, p, stats))
        out[pid] = Pair(pid, f"{pid}_01", 1, links["inputsTrBL"], links["inputsTrFU"], layout / "inputsTrBL" / f"{pid}_00.json", layout / "inputsTrFU" / f"{pid}_01.json",
                        links["targetsTrBL"], links["targetsTrFU"], layout / "inputsTrFU" / f"{pid}_01.json", meta)
    return out


def patch(arr_zyx: np.ndarray, centre_xyz: np.ndarray) -> np.ndarray:
    c = np.round(centre_xyz).astype(int)[::-1]
    ix = [np.clip(np.arange(c[a] - PATCH, c[a] + PATCH + 1), 0, arr_zyx.shape[a] - 1) for a in range(3)]
    return arr_zyx[np.ix_(*ix)].astype(np.float32).ravel()


def baseline_inputs(root: Path, p: dict) -> dict:
    """Inputs of Di Veroli and Qahqaie at Lstar for one pair (structure of exp04 `inputs.build`). BL lesions are all nodes (every BL lesion has a
    propagated point). x_mm in the FU frame, vol in mm^3, pts = int voxels in the FU grid (BL masks translated by `fu_point_prop - bl_point` in mm)."""
    bl_arr, bl_boxes, sp_bl = read_scan(root, p["bl"])
    fu_arr, fu_boxes, sp_fu = read_scan(root, p["fu"])
    itk = sitk.ReadImage(str(root / "images" / f"{p['bl']}_0000.nii.gz"))
    bl_ct = sitk.GetArrayFromImage(itk)
    itk = sitk.ReadImage(str(root / "images" / f"{p['fu']}_0000.nii.gz"))
    fu_ct = sitk.GetArrayFromImage(itk)
    out = {"pid": p["pid"], "missing": [], "bl": [], "fu": []}
    for lid, e in sorted((int(k), v) for k, v in p["track"].items()):
        idx = voxels_xyz(bl_arr, bl_boxes, lid)
        prop, cog = np.asarray(e["fu_point_prop"]), np.asarray(e["bl_point"])
        pts = np.unique(np.round(prop + (idx - cog) * sp_bl / sp_fu).astype(int), axis=0)
        out["bl"].append({"id": lid, "node": True, "source": "uniGradICON", "x_mm": prop * sp_fu, "vol_mm3": float(len(idx) * np.prod(sp_bl)), "pts": pts, "patch": patch(bl_ct, cog)})
    for lid in range(1, len(fu_boxes) + 1):
        idx = voxels_xyz(fu_arr, fu_boxes, lid)
        if idx is None:
            continue
        c = idx.mean(0)
        out["fu"].append({"id": lid, "node": True, "x_mm": c * sp_fu, "vol_mm3": float(len(idx) * np.prod(sp_fu)), "pts": idx, "patch": patch(fu_ct, c)})
    out["overlap"] = overlap_table([b["pts"] for b in out["bl"]], [f["pts"] for f in out["fu"]])
    return out
