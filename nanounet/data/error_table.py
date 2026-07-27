"""Registration-error offset table: load-once-per-process cache, validation, and the empirical draw.

Schema (see scripts/measure_registration_error.py): {frame, spacing_zyx, size_bins_mm,
backends: {name: {offsets_zyx: [[dz,dy,dx], ...] per size bin]}}, excluded, provenance}. Offsets are
in RESAMPLED voxels. Shared by nanounet/config.py (startup validation) and
nanounet/data/sampling.py (the actual draw), so the JSON is parsed exactly once per process.
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

_MEASURE_CMD = "python3 scripts/measure_registration_error.py"
_CACHE: Dict[str, dict] = {}

DEFAULT_ERROR_TABLE = "/nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json"
DEFAULT_BACKENDS = ("original", "unigradicon")
DEFAULT_SIGMA = (5.95, 6.39, 5.93)  # corrected through-plane axis, see docs/reference/config.md


def parse_propagated(d: dict | None) -> dict:
    """Parse+validate the `propagated` config block; returns kwargs for PropagatedConfig."""
    d = d if isinstance(d, dict) else {}
    mode = str(d.get("mode", "empirical"))
    if mode not in ("gaussian", "empirical"):
        raise ValueError(f"propagated.mode must be 'gaussian' or 'empirical', got {mode!r}")
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
    if not p.is_file():
        raise FileNotFoundError(
            f"propagated.error_table {path!r} does not exist (mode=empirical requires it).\n"
            f"Fix: {_MEASURE_CMD}   (writes {path})"
        )
    try:
        table = load_table(path)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"propagated.error_table {path!r} is not valid JSON ({e}).\nFix: {_MEASURE_CMD}"
        ) from e
    size_bins = table.get("size_bins_mm")
    if not size_bins:
        raise ValueError(f"propagated.error_table {path!r} has no size_bins_mm.\nFix: {_MEASURE_CMD}")
    table_backends = table.get("backends", {})
    for b in backends:
        if b not in table_backends:
            raise ValueError(
                f"propagated.backends requests {b!r} but {path!r} only has "
                f"{list(table_backends)}.\nFix: {_MEASURE_CMD}"
            )
        offsets = table_backends[b].get("offsets_zyx", [])
        if len(offsets) != len(size_bins):
            raise ValueError(
                f"propagated.error_table {path!r} backend {b!r} has {len(offsets)} size-bin "
                f"entries, expected {len(size_bins)}.\nFix: {_MEASURE_CMD}"
            )
        for i, bin_offsets in enumerate(offsets):
            if len(bin_offsets) == 0:
                raise ValueError(
                    f"propagated.error_table {path!r} backend {b!r} size bin {size_bins[i]} "
                    f"is empty.\nFix: {_MEASURE_CMD}"
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


def sample_offset_vox(
    volume_vox: float,
    path: str,
    backends: Tuple[str, ...],
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """One offset (dz,dy,dx) in RESAMPLED voxels, drawn from the measured table, size-matched to
    the lesion's equivalent-sphere diameter."""
    table = load_table(path)
    spacing = tuple(float(x) for x in table["spacing_zyx"])
    diam_mm = volume_vox_to_diam_mm(float(volume_vox), spacing)
    binidx = _bin_index(diam_mm, table["size_bins_mm"])
    return _draw_from_bin(table, backends, binidx, rng)


def sample_offset_vox_pooled(
    path: str, backends: Tuple[str, ...], rng: np.random.Generator
) -> Tuple[float, float, float]:
    """Offset drawn from a uniformly-random size bin -- used when no lesion volume is known
    (e.g. a follow-up click with no matching segmentation component)."""
    table = load_table(path)
    binidx = int(rng.integers(len(table["size_bins_mm"])))
    return _draw_from_bin(table, backends, binidx, rng)


def draw_propagated_offset(
    centroid_zyx: Tuple[int, int, int],
    volume_vox: float | None,
    prop: "PropagatedConfig",
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    """Displace a GLOBAL centroid by one draw from cfg.sampling.propagated. mode='empirical' draws
    a real measured registration offset, size-matched via volume_vox (pooled across bins if the
    volume is unknown, e.g. an unmatched follow-up click); mode='gaussian' keeps the legacy
    Gaussian jitter. No magnitude clip for empirical -- the table is already outlier-filtered."""
    if prop.mode == "gaussian":
        return apply_propagation_offset(centroid_zyx, prop.sigma_per_axis, prop.max_vox, rng)
    if volume_vox is None:
        dz, dy, dx = sample_offset_vox_pooled(prop.error_table, prop.backends, rng)
    else:
        dz, dy, dx = sample_offset_vox(float(volume_vox), prop.error_table, prop.backends, rng)
    cz, cy, cx = centroid_zyx
    return (int(round(cz + dz)), int(round(cy + dy)), int(round(cx + dx)))
