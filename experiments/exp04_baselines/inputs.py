"""Per-patient inputs of the two baselines under node supply Lstar: lesion centroids and volumes in the FU frame, BL lesions moved into the FU
voxel grid by point propagation, appearance patches and the tabulated Di Veroli overlaps. One pickle per patient in `artifacts/inputs/`.

Point propagation (the same propagation as ours and Kirchhoff, no deformation field): every voxel of a BL lesion is shifted by
`cog_propagated - cog_bl` in mm and re-gridded to the FU voxel grid. A BL lesion without a propagated point gets one from the
uniGradICON derivative only where `sanity_ok` (`--prop-fill unigradicon`, the same rule as the graph builder's opt-in fill); otherwise it
owns no node and is counted as missed, never guessed. A lesion whose id is absent from its instance mask is reported in `missing`.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from experiments.exp04_baselines.diveroli import overlap_table
from experiments.scoring import PairCase
from lesionglue.data.source.meta import V2Paths

PATCH = 4  # half width of the cubic appearance patch in voxels (9 x 9 x 9)
UG_DIR = "derivatives/unigrad-icon-registration"


def read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(xyz float32 array, spacing xyz): the voxel layout of the meta `cog_*` coordinates."""
    itk = sitk.ReadImage(str(path))
    arr = np.ascontiguousarray(sitk.GetArrayFromImage(itk).transpose(2, 1, 0).astype(np.float32))
    return arr, np.asarray(itk.GetSpacing(), dtype=np.float64)


def xyz(cell: str) -> np.ndarray | None:
    cell = (cell or "").strip()
    return np.asarray([float(t) for t in cell.split()], dtype=np.float64) if cell else None


def fill_points(root: Path, pid: str, fu_id: int) -> dict[int, np.ndarray]:
    """{lesion id: uniGradICON bl_click (FU voxels)} for lesions whose sanity check passed."""
    for split in ("train", "test"):
        f = root / UG_DIR / split / "lesions" / f"{pid}_{fu_id:02d}.json"
        if f.is_file():
            return {int(l["lesion_id"]): np.asarray(l["bl_click"], dtype=np.float64) for l in json.loads(f.read_text())["lesions"]
                    if l.get("bl_click") is not None and l.get("sanity_ok") and int(l["img_id_fu"]) == fu_id}
    return {}


def patch(vol: np.ndarray, centre: np.ndarray) -> np.ndarray:
    c = np.round(centre).astype(int)
    ix = [np.clip(np.arange(c[a] - PATCH, c[a] + PATCH + 1), 0, vol.shape[a] - 1) for a in range(3)]
    return vol[np.ix_(*ix)].astype(np.float32).ravel()


def lesion_voxels(mask: np.ndarray, boxes: list, lid: int) -> np.ndarray | None:
    if lid - 1 >= len(boxes) or boxes[lid - 1] is None:
        return None
    sl = boxes[lid - 1]
    return np.argwhere(mask[sl] == lid) + np.array([s.start for s in sl])


def build(root: Path, pid: str, case: PairCase, prop_fill: str) -> dict:
    with open(root / "meta" / f"{pid}.csv", newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    n_fu = Counter(int(r["img_id_fu"]) for r in raw)
    fu_id = max(n_fu, key=lambda k: (n_fu[k], -k))
    rows = {int(r["lesion_id"]): r for r in raw if int(r["img_id_fu"]) == fu_id}
    vp = V2Paths(root, pid)
    filled = fill_points(root, pid, fu_id) if prop_fill == "unigradicon" else {}
    fu_mask, sp_fu = read_xyz(vp.fu_mask(fu_id))
    fu_ct, _ = read_xyz(vp.fu_img(fu_id))
    fu_int = fu_mask.astype(np.int32)
    fu_boxes = ndi.find_objects(fu_int)
    out = {"pid": pid, "fu_id": fu_id, "missing": [], "bl": [], "fu": [], "spacing_fu": sp_fu}
    scans: dict[int, tuple] = {}
    for lid in case.bl_ids:
        r = rows[lid]
        k = int(r["img_id_bl"])
        if k not in scans:
            m, sp = read_xyz(vp.bl_mask(k))
            ct, _ = read_xyz(vp.bl_img(k))
            scans[k] = (m.astype(np.int32), ndi.find_objects(m.astype(np.int32)), ct, sp)
        m, boxes, ct, sp_bl = scans[k]
        cog_bl, prop = xyz(r["cog_bl"]), xyz(r["cog_propagated"])
        source = "original"
        if prop is None and lid in filled:
            prop, source = filled[lid], "unigradicon"
        idx = lesion_voxels(m, boxes, lid)
        if prop is None or idx is None or cog_bl is None:
            out["missing"].append(("bl", lid, "no propagated point" if prop is None else "id absent from BL mask"))
            out["bl"].append({"id": lid, "node": False})
            continue
        pts = np.unique(np.round(prop + (idx - cog_bl) * sp_bl / sp_fu).astype(int), axis=0)
        out["bl"].append({"id": lid, "node": True, "source": source, "x_mm": prop * sp_fu, "vol_mm3": float(len(idx) * np.prod(sp_bl)),
                          "pts": pts, "patch": patch(ct, cog_bl)})
    for lid in case.fu_ids:
        idx = lesion_voxels(fu_int, fu_boxes, lid)
        if idx is None:
            out["missing"].append(("fu", lid, "id absent from FU mask"))
            out["fu"].append({"id": lid, "node": False})
            continue
        c = idx.mean(axis=0)
        out["fu"].append({"id": lid, "node": True, "x_mm": c * sp_fu, "vol_mm3": float(len(idx) * np.prod(sp_fu)), "pts": idx, "patch": patch(fu_ct, c)})
    out["overlap"] = overlap_table([b.get("pts") for b in out["bl"]], [f.get("pts") for f in out["fu"]])
    return out


def get(root: Path, pid: str, case: PairCase, prop_fill: str, cache_dir: Path, *, write: bool) -> dict:
    """Load `inputs/<pid>.pkl` or build and store it (write=False, used by --rescore, refuses to build)."""
    f = cache_dir / f"{pid}.pkl"
    if f.is_file():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    if not write:
        raise FileNotFoundError(f"{f} is missing in the run being rescored\nExpected the inputs/ artifacts written by the original run.\nFix: rerun without --rescore (or --resume that run)")
    out = build(root, pid, case, prop_fill)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(out, fh)
    return out
