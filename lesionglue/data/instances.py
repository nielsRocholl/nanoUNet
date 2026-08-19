"""Binary FG + nanoUNet click JSON → instance mask (voxel = lesion_id)."""

from __future__ import annotations

import json
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np

from tracking.common import cprint


def load_clicks(path: Path) -> dict[int, tuple[int, int, int]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No clicks JSON at {path}.\n"
            f"Expected {{'points': [{{'name': '<id>', 'point': [x,y,z]}}, ...]}}.\n"
            f"Fix: pass the sibling nanoUNet click JSON for that scan"
        )
    pts = json.loads(path.read_text()).get("points")
    if not isinstance(pts, list):
        raise SystemExit(
            f"'points' missing or not a list in {path}.\n"
            f"Expected {{'points': [{{'name': '<id>', 'point': [x,y,z]}}, ...]}}.\n"
            f"Fix: pass nanoUNet click JSON"
        )
    out: dict[int, tuple[int, int, int]] = {}
    for item in pts:
        raw = item.get("name") if isinstance(item, dict) else None
        try:
            lid = int(raw)
        except (TypeError, ValueError):
            raise SystemExit(
                f"Click in {path} has missing or non-integer name: {item!r}.\n"
                f"Expected points[].name to be the lesion_id integer.\n"
                f"Fix: use sibling JSON from inputsTrFU"
            ) from None
        p = item["point"]
        out.setdefault(lid, (int(round(p[2])), int(round(p[1])), int(round(p[0]))))
    return out


def binary_to_instances(pred: np.ndarray, clicks_zyx: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """pred is bool/0-1, same grid as clicks. Return int32 mask, voxel = lesion_id."""
    out = np.zeros(pred.shape, dtype=np.int32)
    lab = cc3d.connected_components((pred > 0).astype(np.uint8), connectivity=18)
    claimed: dict[int, int] = {}
    conflicts: list[tuple[int, int, int]] = []
    for lid, (z, y, x) in clicks_zyx.items():
        z = min(max(int(z), 0), pred.shape[0] - 1)
        y = min(max(int(y), 0), pred.shape[1] - 1)
        x = min(max(int(x), 0), pred.shape[2] - 1)
        cc = int(lab[z, y, x])
        if cc == 0:
            continue
        if cc in claimed and claimed[cc] != lid:
            conflicts.append((lid, claimed[cc], cc))
            continue
        claimed[cc] = lid
        out[lab == cc] = lid
    if conflicts:
        cprint(f"[yellow]click CC conflicts (later click skipped): {conflicts}[/yellow]")
    return out


def instances_from_nifti(pred_path: Path, clicks_json_path: Path, out_path: Path) -> Path:
    img = nib.load(str(pred_path))
    vol = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
    pred_zyx = np.transpose(vol, (2, 1, 0))
    inst_zyx = binary_to_instances(pred_zyx, load_clicks(clicks_json_path))
    inst = np.ascontiguousarray(np.transpose(inst_zyx, (2, 1, 0)).astype(np.int32))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(inst, img.affine, img.header), str(out_path))
    return out_path
