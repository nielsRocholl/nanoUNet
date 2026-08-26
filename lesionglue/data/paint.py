"""Paint FU instance mask with tracking ids. BL ids stay canonical."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def fu_track_map(
    bl_ids: list[int],
    fu_ids: list[int],
    pairs: list[tuple[int, int]],
) -> dict[int, int]:
    """fu_click_id → tracking id painted on the FU mask."""
    fu_bls: dict[int, list[int]] = {}
    for b, f in pairs:
        fu_bls.setdefault(int(f), []).append(int(b))
    used = {int(x) for x in bl_ids}
    out: dict[int, int] = {}
    for f, bls in fu_bls.items():
        tid = min(bls)
        out[f] = tid
        used.add(tid)
    nxt = (max(used) + 1) if used else 1
    for f in fu_ids:
        f = int(f)
        if f in out:
            continue
        if f not in used:
            out[f] = f
            used.add(f)
            nxt = max(nxt, f + 1)
        else:
            out[f] = nxt
            used.add(nxt)
            nxt += 1
    return out


def paint_fu(fu_mask: np.ndarray, m: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(fu_mask, dtype=np.int32)
    for src, tid in m.items():
        out[fu_mask == src] = tid
    return out


def write_case_masks(bl_inst: Path, fu_inst: Path, out_dir: Path, mapping: dict[int, int]) -> tuple[Path, Path]:
    bl_img = nib.load(str(bl_inst))
    fu_img = nib.load(str(fu_inst))
    bl = np.ascontiguousarray(bl_img.get_fdata(dtype=np.float32)).astype(np.int32)
    fu = np.ascontiguousarray(fu_img.get_fdata(dtype=np.float32)).astype(np.int32)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bl_out, fu_out = out_dir / "bl.nii.gz", out_dir / "fu.nii.gz"
    nib.save(nib.Nifti1Image(bl, bl_img.affine, bl_img.header), str(bl_out))
    nib.save(nib.Nifti1Image(paint_fu(fu, mapping), fu_img.affine, fu_img.header), str(fu_out))
    return bl_out, fu_out


def copy_mask(src: Path, dest: Path, *, zero: bool = False) -> Path:
    img = nib.load(str(src))
    vol = np.ascontiguousarray(img.get_fdata(dtype=np.float32)).astype(np.int32)
    if zero:
        vol = np.zeros_like(vol)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, img.affine, img.header), str(dest))
    return dest


CSV_HEADER = "bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id\n"


def write_empty_csv(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CSV_HEADER)
