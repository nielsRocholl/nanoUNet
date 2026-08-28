"""SegTrackCase, folder pairing by stem intersection (skip list), output paths, load BL instance mask (sitk zyx)."""
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
    meta_csv: Path | None = None


def stem_pid_region(stem: str) -> tuple[str, int]:
    pid, rr = stem.rsplit("_", 1)
    return pid, int(rr)


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


def _stems(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"No input folder at {folder}.\n"
            f"Expected a folder of {{stem}}{END} + sibling {{stem}}.json.\n"
            f"Fix: --bl-dir / --fu-dir like inputsTrBL / inputsTrFU  (see docs/steps/track.md)"
        )
    out = {p.name[: -len(END)]: p for p in sorted(folder.glob(f"*{END}"))}
    if not out:
        raise SystemExit(
            f"No {END} scans in {folder}.\n"
            f"Expected .nii.gz CTs (sibling .json per case).\n"
            f"Fix: pass --bl-dir / --fu-dir  (see docs/steps/track.md)"
        )
    return out


def pair_folder(
    bl_dir: Path, fu_dir: Path, *, bl_mask_dir: Path | None = None, pids: set[str] | None = None,
) -> tuple[list[SegTrackCase], list[tuple[str, str]]]:
    bl_dir, fu_dir = Path(bl_dir), Path(fu_dir)
    bl, fu = _stems(bl_dir), _stems(fu_dir)
    if pids is not None:
        bl = {s: p for s, p in bl.items() if s.split("_", 1)[0] in pids}
        fu = {s: p for s, p in fu.items() if s.split("_", 1)[0] in pids}
        if not bl and not fu:
            raise SystemExit(
                "no cases match --patients-csv.\n"
                "Expected CSV column 'patient' matching stem prefixes.\n"
                "Fix: --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv  (see docs/steps/track.md)"
            )
    skipped: list[tuple[str, str]] = [(s, "no FU scan") for s in sorted(set(bl) - set(fu))]
    skipped += [(s, "no BL scan") for s in sorted(set(fu) - set(bl))]
    both = sorted(set(bl) & set(fu))
    if not both:
        raise SystemExit(
            "BL/FU folders do not share any case names.\n"
            f"--bl-dir has {len(bl)} stems, --fu-dir has {len(fu)} "
            f"(e.g. {', '.join(sorted(bl)[:6]) or 'none'}).\n"
            "Expected matching {stem}.nii.gz in both folders.\n"
            "Fix: pass matching inputsTrBL and inputsTrFU  (see docs/steps/track.md)"
        )
    md = Path(bl_mask_dir) if bl_mask_dir else None
    if md is not None and not md.is_dir():
        raise SystemExit(
            f"No BL mask folder at {md}.\n"
            "Expected a folder of {stem}.nii.gz instance masks.\n"
            "Fix: --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL  (see docs/steps/track.md)"
        )
    cases: list[SegTrackCase] = []
    for s in both:
        fu_js = fu_dir / f"{s}.json"
        if not fu_js.is_file():
            skipped.append((s, "no FU json"))
            continue
        if md is None:
            bl_js = bl_dir / f"{s}.json"
            if not bl_js.is_file():
                skipped.append((s, "no BL json"))
                continue
            cases.append(SegTrackCase(s, bl[s], bl_js, fu[s], fu_js))
            continue
        m = _mask_file(md, s)
        if m is None:
            skipped.append((s, "no BL mask"))
            continue
        cases.append(SegTrackCase(s, bl[s], None, fu[s], fu_js, None, m))
    return cases, skipped


def load_instance_zyx(path: Path) -> tuple[np.ndarray, dict]:
    """Read instance mask via SimpleITKIO. Return (Z,Y,X) int32 + sitk props."""
    vol, props = SimpleITKIO().read_seg(str(path))
    assert vol.ndim == 4 and vol.shape[0] == 1, vol.shape
    return np.ascontiguousarray(np.rint(vol[0]).astype(np.int32)), props


def load_ct(path: Path):
    """One SimpleITK read: (C,Z,Y,X) for preprocess + XYZ/RAS for the matcher."""
    from tracking.data.graph import vol_from_zyx

    data, props = SimpleITKIO().read_images((str(path),))
    return data, props, vol_from_zyx(data[0], props["sitk_stuff"])
