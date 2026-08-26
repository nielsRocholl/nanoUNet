"""SegTrackCase, folder pairing, output paths, load BL instance mask (sitk zyx)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanounet.common import results_dir
from nanounet.data.io import SimpleITKIO

END = ".nii.gz"


@dataclass
class SegTrackCase:
    stem: str
    bl_img: Path
    bl_clicks: Path | None
    fu_img: Path
    fu_clicks: Path
    types_csv: Path | None = None
    bl_mask: Path | None = None


def resolve_ckpt_path(cli: str | None, env_key: str, default: Path) -> tuple[Path, str]:
    if cli:
        return Path(cli), "cli"
    v = os.environ.get(env_key)
    if v:
        return Path(v), "env"
    return default, "default"


def resolve_out(stem: str, *, fu_dir_name: str | None, out: Path | None, single: bool) -> Path:
    if out is not None:
        return Path(out) if single else Path(out) / stem
    root = Path(results_dir()) / "segtrack"
    return root / "single" / stem if single else root / str(fu_dir_name) / stem


def _mask_file(folder: Path, stem: str) -> Path | None:
    for ext in (".nii.gz", ".mha"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _stems(folder: Path, *, clicks: bool) -> dict[str, tuple[Path, Path | None]]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"No input folder at {folder}.\n"
            f"Expected a folder of {{stem}}{END} + sibling {{stem}}.json.\n"
            f"Fix: --bl-dir / --fu-dir like inputsTrBL / inputsTrFU  (see docs/steps/track.md)"
        )
    out, missing = {}, []
    for p in sorted(folder.glob(f"*{END}")):
        stem = p.name[: -len(END)]
        js = folder / f"{stem}.json"
        if clicks:
            out[stem] = (p, js)
            if not js.is_file():
                missing.append(stem)
        else:
            out[stem] = (p, None)
    if missing:
        raise SystemExit(
            f"missing points JSON for: {', '.join(missing[:12])}.\n"
            f"Expected sibling <case>.json next to each scan in {folder}.\n"
            f"Fix: add the JSON  (see docs/steps/track.md)"
        )
    if not out:
        extra = (
            "Expected .nii.gz CTs (BL clicks not required with --bl-mask-dir)."
            if not clicks else
            "Expected sibling .nii.gz + .json like inputsTrFU."
        )
        raise SystemExit(
            f"No {END} scans in {folder}.\n{extra}\n"
            f"Fix: pass --bl-dir / --fu-dir  (see docs/steps/track.md)"
        )
    return out


def pair_folder(bl_dir: Path, fu_dir: Path, *, bl_mask_dir: Path | None = None) -> list[SegTrackCase]:
    bl, fu = _stems(Path(bl_dir), clicks=bl_mask_dir is None), _stems(Path(fu_dir), clicks=True)
    only_bl, only_fu = sorted(set(bl) - set(fu)), sorted(set(fu) - set(bl))
    if only_bl or only_fu:
        raise SystemExit(
            "BL/FU folders do not share the same case names.\n"
            f"--bl-dir has {len(only_bl)} stems not in --fu-dir (e.g. {', '.join(only_bl[:12]) or 'none'}).\n"
            f"--fu-dir has {len(only_fu)} stems not in --bl-dir (e.g. {', '.join(only_fu[:12]) or 'none'}).\n"
            "Fix: pass matching inputsTrBL and inputsTrFU, or --patients-csv to select a subset\n"
            "(see docs/steps/track.md)"
        )
    if bl_mask_dir is None:
        return [SegTrackCase(s, bl[s][0], bl[s][1], fu[s][0], fu[s][1]) for s in sorted(bl)]
    md = Path(bl_mask_dir)
    if not md.is_dir():
        raise SystemExit(
            f"No BL mask folder at {md}.\n"
            "Expected a folder of {stem}.nii.gz instance masks.\n"
            "Fix: --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL  (see docs/steps/track.md)"
        )
    miss = [s for s in sorted(bl) if _mask_file(md, s) is None]
    if miss:
        raise SystemExit(
            f"No BL instance mask for: {', '.join(miss[:12])} ({len(miss)} missing).\n"
            "Expected {stem}.nii.gz or {stem}.mha under --bl-mask-dir.\n"
            "Fix: --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL  (see docs/steps/track.md)"
        )
    return [
        SegTrackCase(s, bl[s][0], None, fu[s][0], fu[s][1], None, _mask_file(md, s))
        for s in sorted(bl)
    ]


def load_instance_zyx(path: Path) -> tuple[np.ndarray, dict]:
    """Read instance mask via SimpleITKIO. Return (Z,Y,X) int32 + sitk props."""
    vol, props = SimpleITKIO().read_seg(str(path))
    assert vol.ndim == 4 and vol.shape[0] == 1, vol.shape
    return np.ascontiguousarray(np.rint(vol[0]).astype(np.int32)), props
