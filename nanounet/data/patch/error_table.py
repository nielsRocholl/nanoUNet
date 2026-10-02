"""Registration-error offset table: load-once-per-process cache, validation, and the empirical draw.

Schema (see nanounet/docs/reference/config.md): {frame, spacing_zyx, size_bins_mm,
backends: {name: {offsets_zyx: [[dz,dy,dx], ...] per size bin]}}, excluded, provenance}. Offsets are
in the TABLE's resampled voxels (table spacing_zyx), so the draw goes voxels -> mm with the table
spacing, then mm -> voxels with the training data's spacing (PropagatedConfig.data_spacing_zyx,
bound from the plans, or per case, by nanounet/data/patch/spacing.py); lesion volume -> diameter also uses the data spacing.
Shared by nanounet/config.py (startup validation) and
nanounet/data/patch/sampling.py (the actual draw), so the JSON is parsed exactly once per process.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

from nanounet.prompt.centroids import apply_propagation_offset

if TYPE_CHECKING:
    from nanounet.config import PropagatedConfig

_CACHE: Dict[str, dict] = {}

DEFAULT_ERROR_TABLE = "/nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json"
DEFAULT_BACKENDS = ("original", "unigradicon")
DEFAULT_SIGMA = (5.95, 6.39, 5.93)  # corrected through-plane axis, see nanounet/docs/reference/config.md


def parse_propagated(d: dict | None) -> dict:
    """Parse+validate the `propagated` config block; returns kwargs for PropagatedConfig."""
    d = d if isinstance(d, dict) else {}
    mode = str(d.get("mode", "empirical"))
    if mode not in ("gaussian", "empirical"):
        raise ValueError(
            f"propagated.mode must be 'gaussian' or 'empirical', got {mode!r}\n"
            f"Expected config sampling.propagated.mode to be one of 'gaussian' or 'empirical'.\n"
            f"Fix: set sampling.propagated.mode to 'gaussian' or 'empirical' in the training config JSON. See nanounet/docs/reference/config.md"
        )
    sg = d.get("sigma_per_axis", DEFAULT_SIGMA)
    assert isinstance(sg, (list, tuple)) and len(sg) == 3
    backends_raw = d.get("backends", DEFAULT_BACKENDS)
    assert isinstance(backends_raw, (list, tuple)) and len(backends_raw) > 0
    backends = tuple(str(b) for b in backends_raw)
    error_table = str(d.get("error_table", DEFAULT_ERROR_TABLE))
    if mode == "empirical":
        validate_table(error_table, backends)
    return dict(
        mode=mode,
        error_table=error_table,
        backends=backends,
        sigma_per_axis=tuple(float(x) for x in sg),
        max_vox=float(d.get("max_vox", 34.0)),
    )


def load_table(path: str) -> dict:
    if path not in _CACHE:
        _CACHE[path] = json.loads(Path(path).read_text(encoding="utf-8"))
    return _CACHE[path]


def validate_table(path: str, backends: Tuple[str, ...]) -> None:
    p = Path(path)
    fix = (
        'Fix: point propagated.error_table at a table matching the schema in '
        'nanounet/docs/reference/config.md, or set propagated.mode to "gaussian" (no table needed).'
    )
    if not p.is_file():
        raise FileNotFoundError(
            f"propagated.error_table {path!r} does not exist (mode=empirical requires it).\n{fix}"
        )
    try:
        table = load_table(path)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"propagated.error_table {path!r} is not valid JSON ({e}).\n{fix}"
        ) from e
    if table.get("frame") != "resampled_voxels_zyx" or len(table.get("spacing_zyx", [])) != 3:
        raise ValueError(  # nanochat-style: allow E1 (Fix: line is in the `fix` variable)
            f"propagated.error_table {path!r} has frame={table.get('frame')!r} and "
            f"spacing_zyx={table.get('spacing_zyx')!r}; expected frame 'resampled_voxels_zyx' with a 3-value spacing_zyx.\n{fix}"
        )
    size_bins = table.get("size_bins_mm")
    if not size_bins:
        raise ValueError(f"propagated.error_table {path!r} has no size_bins_mm.\n{fix}")
    table_backends = table.get("backends", {})
    for b in backends:
        if b not in table_backends:
            raise ValueError(  # nanochat-style: allow E1 (Fix: line is in the `fix` variable)
                f"propagated.backends requests {b!r} but {path!r} only has "
                f"{list(table_backends)}.\n{fix}"
            )
        offsets = table_backends[b].get("offsets_zyx", [])
        if len(offsets) != len(size_bins):
            raise ValueError(  # nanochat-style: allow E1 (Fix: line is in the `fix` variable)
                f"propagated.error_table {path!r} backend {b!r} has {len(offsets)} size-bin "
                f"entries, expected {len(size_bins)}.\n{fix}"
            )
        for i, bin_offsets in enumerate(offsets):
            if len(bin_offsets) == 0:
                raise ValueError(  # nanochat-style: allow E1 (Fix: line is in the `fix` variable)
                    f"propagated.error_table {path!r} backend {b!r} size bin {size_bins[i]} "
                    f"is empty.\n{fix}"
                )


def volume_vox_to_diam_mm(volume_vox: float, spacing_zyx: Tuple[float, float, float]) -> float:
    vol_mm3 = volume_vox * spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2]
    return 2.0 * (3.0 * vol_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def _bin_index(diam_mm: float, size_bins_mm: list) -> int:
    for i, (lo, hi) in enumerate(size_bins_mm):
        if lo <= diam_mm < hi:
            return i
    return len(size_bins_mm) - 1 if diam_mm >= size_bins_mm[-1][0] else 0


def _draw_from_bin(table: dict, backends: Tuple[str, ...], binidx: int, rng: np.random.Generator):
    b = backends[int(rng.integers(len(backends)))]
    pool = table["backends"][b]["offsets_zyx"][binidx]
    off = pool[int(rng.integers(len(pool)))]
    return float(off[0]), float(off[1]), float(off[2])


def sample_offset_mm(
    volume_vox: float,
    path: str,
    backends: Tuple[str, ...],
    data_spacing_zyx: Tuple[float, ...],
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """One offset (dz,dy,dx) in MILLIMETRES, drawn from the measured table, size-matched to the
    lesion's equivalent-sphere diameter (volume_vox is in DATA voxels)."""
    table = load_table(path)
    diam_mm = volume_vox_to_diam_mm(float(volume_vox), data_spacing_zyx)
    binidx = _bin_index(diam_mm, table["size_bins_mm"])
    return _table_vox_to_mm(_draw_from_bin(table, backends, binidx, rng), table)


def sample_offset_mm_pooled(
    path: str, backends: Tuple[str, ...], rng: np.random.Generator
) -> Tuple[float, float, float]:
    """Offset (mm) drawn from a uniformly-random size bin -- used when no lesion volume is known
    (e.g. a follow-up click with no matching segmentation component)."""
    table = load_table(path)
    binidx = int(rng.integers(len(table["size_bins_mm"])))
    return _table_vox_to_mm(_draw_from_bin(table, backends, binidx, rng), table)


def _table_vox_to_mm(off: Tuple[float, float, float], table: dict) -> Tuple[float, float, float]:
    sp = table["spacing_zyx"]
    return (off[0] * float(sp[0]), off[1] * float(sp[1]), off[2] * float(sp[2]))


def draw_propagated_offset(
    centroid_zyx: Tuple[int, int, int],
    volume_vox: float | None,
    prop: "PropagatedConfig",
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    """Displace a GLOBAL centroid by one draw from cfg.sampling.propagated. mode='empirical' draws
    a real measured registration offset (mm -> data voxels via prop.data_spacing_zyx), size-matched via volume_vox (pooled across bins if the
    volume is unknown, e.g. an unmatched follow-up click); mode='gaussian' keeps the legacy
    Gaussian jitter. No magnitude clip for empirical -- the table is already outlier-filtered."""
    if prop.mode == "gaussian":
        return apply_propagation_offset(centroid_zyx, prop.sigma_per_axis, prop.max_vox, rng)
    sp = prop.data_spacing_zyx
    if sp is None:
        raise ValueError(
            "propagated.mode='empirical' needs a voxel spacing to convert the table's mm offsets to voxels, "
            "but none was bound (z-only plans: the case has no spacing_after_resampling).\n"
            "Expected the plans' 3d_fullres spacing (bind_plan_spacing) or the case's spacing_after_resampling.\n"
            "Fix: re-run nanounet_preprocess for this plan, or set propagated.mode to 'gaussian'."
        )
    if volume_vox is None:
        dmm = sample_offset_mm_pooled(prop.error_table, prop.backends, rng)
    else:
        dmm = sample_offset_mm(float(volume_vox), prop.error_table, prop.backends, sp, rng)
    return tuple(int(round(c + d / s)) for c, d, s in zip(centroid_zyx, dmm, sp))
