"""Longitudinal_CT_v2 paths, lesion rows parsed from meta CSV, and split JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tracking.common import LESION_TYPES, load_json

_VALID_TOPO = frozenset({"UNCHANGED", "DISAPPEARED", "NEWLYAPPEARING", "MERGED", "SPLIT"})
_ALIAS_TOPO = {"DISAPPEARING": "DISAPPEARED", "MERGING": "MERGED"}


def _norm_topo(raw: str) -> str:
    t = str(raw).strip()
    t = _ALIAS_TOPO.get(t, t)
    if t not in _VALID_TOPO:
        raise ValueError(f"unknown topology_class {raw!r}")
    return t


def parse_xyz(s: object) -> tuple[float, float, float] | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).strip()
    if not t:
        return None
    parts = t.split()
    if len(parts) != 3:
        raise ValueError(f"expected x y z triple, got {s!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


@dataclass(frozen=True)
class V2Paths:
    root: Path
    pid: str

    @property
    def meta(self) -> Path:
        return self.root / "meta" / f"{self.pid}.csv"

    def bl_img(self, idx: int) -> Path:
        return self.root / "inputsTrBL" / f"{self.pid}_{idx:02d}.nii.gz"

    def bl_mask(self, idx: int) -> Path:
        return self.root / "targetsTrBL" / f"{self.pid}_{idx:02d}.nii.gz"

    def fu_img(self, idx: int) -> Path:
        return self.root / "inputsTrFU" / f"{self.pid}_{idx:02d}.nii.gz"

    def fu_mask(self, idx: int) -> Path:
        return self.root / "targetsTrFU" / f"{self.pid}_{idx:02d}.nii.gz"


@dataclass(frozen=True)
class TrackCase:
    pid: str
    bl_img: Path
    bl_mask: Path
    fu_img: Path
    fu_mask: Path
    propagated: Path


def _img_ids(meta: Path) -> tuple[int, int]:
    df = pd.read_csv(meta)
    fu = int(df["img_id_fu"].value_counts().idxmax())
    bl = int(df.loc[df["img_id_fu"] == fu, "img_id_bl"].value_counts().idxmax())
    return bl, fu


def resolve_track_case(
    root: Path,
    pid: str,
    *,
    prop_dir: Path | None = None,
    bl_mask_dir: Path | None = None,
    fu_mask_dir: Path | None = None,
) -> TrackCase | None:
    root = Path(root)
    vp = V2Paths(root, pid)
    bl_i, fu_i = _img_ids(vp.meta) if vp.meta.is_file() else (0, 0)
    pdir = Path(prop_dir) if prop_dir is not None else root / "meta"
    prop = pdir / f"{pid}.csv"
    if not prop.is_file():
        prop = pdir / f"{pid}_{fu_i:02d}.json"
    bl_mask = Path(bl_mask_dir) / f"{pid}_{bl_i:02d}.nii.gz" if bl_mask_dir else vp.bl_mask(bl_i)
    fu_mask = Path(fu_mask_dir) / f"{pid}_{fu_i:02d}.nii.gz" if fu_mask_dir else vp.fu_mask(fu_i)
    case = TrackCase(pid, vp.bl_img(bl_i), bl_mask, vp.fu_img(fu_i), fu_mask, prop)
    if not all(p.is_file() for p in (case.bl_img, case.bl_mask, case.fu_img, case.fu_mask, case.propagated)):
        return None
    return case


@dataclass(frozen=True)
class LesionRow:
    lesion_id: int
    topology: str
    cog_bl: tuple[float, float, float] | None
    cog_propagated: tuple[float, float, float] | None
    cog_backpropagated: tuple[float, float, float] | None
    cog_fu: tuple[float, float, float] | None
    img_id_bl: int
    img_id_fu: int
    lesion_type: str
    merged_into: int | None


def parse_meta_csv(path: Path) -> list[LesionRow]:
    df = pd.read_csv(path)
    rows: list[LesionRow] = []
    for _, r in df.iterrows():
        if bool(r.get("linking_unclear", False)):
            continue
        topo = _norm_topo(r["topology_class"])
        lt = str(r["lesion_type"]).strip()
        if lt not in LESION_TYPES:
            raise ValueError(f"unknown lesion_type {lt!r}")
        mi = r.get("merged_into")
        merged = None if pd.isna(mi) or str(mi).strip() == "" else int(mi)
        rows.append(
            LesionRow(
                lesion_id=int(r["lesion_id"]),
                topology=topo,
                cog_bl=parse_xyz(r["cog_bl"]),
                cog_propagated=parse_xyz(r["cog_propagated"]),
                cog_backpropagated=parse_xyz(r.get("cog_backpropagated")),
                cog_fu=parse_xyz(r["cog_fu"]),
                img_id_bl=int(r["img_id_bl"]),
                img_id_fu=int(r["img_id_fu"]),
                lesion_type=lt,
                merged_into=merged,
            )
        )
    return rows


def load_split_json(path: Path) -> dict[str, list[str]]:
    obj = load_json(path)
    assert "train" in obj and "val" in obj
    return obj
