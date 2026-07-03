"""On-disk output for a warped BL/FU pair: dataset layout, click sidecars, QC montage.

write_dataset mirrors the Longitudinal_CT_v2 layout (inputsTrBL/targetsTrBL warped, FU + meta copied).
Clicks are written named by lesion_id so BL point N and FU point N are the same physical lesion.
"""

from __future__ import annotations

import json
import os
import shutil

import itk
import numpy as np

CT_LO, CT_HI = -160.0, 240.0  # soft-tissue window for QC display


def write_points(named_pts: list[tuple[int, list]], path: str) -> None:
    points = [{"name": str(lid), "point": [c[0], c[1], c[2]]} for lid, c in named_pts]
    doc = {
        "name": "Points of interest",
        "points": points,
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=4)


def _copy_if_exists(src: str, dst: str) -> str | None:
    if not os.path.isfile(src):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # copy2 (and its copystat/utime call) raises PermissionError on the CIFS mount;
    # copyfile copies only file data, which is all we need here.
    shutil.copyfile(src, dst)
    return dst


def write_dataset(out_root: str, data_root: str, pid: str, idx: str, res) -> list[str]:
    """Warped BL into inputsTrBL/targetsTrBL; FU + meta copied from data_root."""
    stem = f"{pid}_{idx}"
    written: list[str] = []

    os.makedirs(os.path.join(out_root, "inputsTrBL"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "targetsTrBL"), exist_ok=True)
    bl_img = os.path.join(out_root, "inputsTrBL", f"{stem}.nii.gz")
    bl_pts = os.path.join(out_root, "inputsTrBL", f"{stem}.json")
    bl_seg = os.path.join(out_root, "targetsTrBL", f"{stem}.nii.gz")
    itk.imwrite(res.warped_img, bl_img)
    written.append(bl_img)
    write_points(res.bl_clicks, bl_pts)
    written.append(bl_pts)
    mask_itk = itk.image_from_array(res.warped_seg)
    mask_itk.CopyInformation(res.fu_ref)
    itk.imwrite(mask_itk, bl_seg)
    written.append(bl_seg)

    os.makedirs(os.path.join(out_root, "inputsTrFU"), exist_ok=True)
    fu_img = os.path.join(out_root, "inputsTrFU", f"{stem}.nii.gz")
    fu_pts = os.path.join(out_root, "inputsTrFU", f"{stem}.json")
    p = _copy_if_exists(os.path.join(data_root, f"inputsTrFU/{stem}.nii.gz"), fu_img)
    if p:
        written.append(p)
    if res.fu_out is not None:
        write_points(res.fu_out, fu_pts)
        written.append(fu_pts)
    else:
        p = _copy_if_exists(os.path.join(data_root, f"inputsTrFU/{stem}.json"), fu_pts)
        if p:
            written.append(p)

    p = _copy_if_exists(
        os.path.join(data_root, f"targetsTrFU/{stem}.nii.gz"),
        os.path.join(out_root, "targetsTrFU", f"{stem}.nii.gz"),
    )
    if p:
        written.append(p)

    p = _copy_if_exists(
        os.path.join(data_root, "meta", f"{pid}.csv"),
        os.path.join(out_root, "meta", f"{pid}.csv"),
    )
    if p:
        written.append(p)
    return written


def qc_png(fu_arr, bl_arr, seg, clicks_xyz, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz = fu_arr.shape[0]
    zs = sorted({int(round(c[2])) for c in clicks_xyz if 0 <= int(round(c[2])) < nz})
    if not zs:
        zs = [nz // 2]
    if len(zs) > 3:
        zs = [zs[0], zs[len(zs) // 2], zs[-1]]

    fig, axes = plt.subplots(len(zs), 3, figsize=(12, 4 * len(zs)), squeeze=False)
    for r, z in enumerate(zs):
        for c, (img, title) in enumerate(
            ((fu_arr, "FU CT"), (bl_arr, "warped BL CT"), (fu_arr, "FU + warped BL mask + clicks"))
        ):
            ax = axes[r][c]
            ax.imshow(np.clip(img[z], CT_LO, CT_HI), cmap="gray", vmin=CT_LO, vmax=CT_HI)
            ax.set_title(f"{title}  z={z}")
            ax.axis("off")
            if c == 2:
                if seg[z].any():
                    ax.contour(seg[z], levels=[0.5], colors="red", linewidths=0.8)
                pts = [c2 for c2 in clicks_xyz if int(round(c2[2])) == z]
                if pts:
                    ax.scatter([p[0] for p in pts], [p[1] for p in pts], c="yellow", s=18, marker="x")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
