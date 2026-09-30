"""Per-lesion registration provenance: was this position observed, or a registration guess?

Every graph-eligible node (BL node for UNCHANGED/DISAPPEARED/MERGED/SPLIT, FU node for
UNCHANGED/NEWLYAPPEARING/SPLIT, per lesionglue/data/graph/dense.py::node_rows) always has its own
cog_bl / cog_fu populated in meta/*.csv -- verified across all 300 patients (4,079 BL-eligible
rows, 0 empty cog_bl; 3,065 FU-eligible rows, 0 empty cog_fu). So the plan's original rule
("imputed if cog_bl/cog_fu is empty") is a no-op at the graph-node level: it never fires for a
node the model actually sees.

What clickfix_report.csv's n_bl_filled / n_fu_filled actually count (reconstructed and verified:
291/307 and 298/307 cases match exactly) is different: the number of lesions on the *missing*
side of a NEWLYAPPEARING (no real BL) or DISAPPEARED (no real FU) row that received a
registration-projected guess coordinate for QC bookkeeping. Those guesses are never graph nodes.

So `imputed` here means "this coordinate is a registration guess, not an annotation" -- true only
for a NEWLYAPPEARING's bl-side guess or a DISAPPEARED's fu-side guess. Every entry that
corresponds to a real graph node is `imputed=False` by construction. Reported for stratified
audit ONLY; never fed to the model (that would leak the label-generating process).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lesionglue.data.source.meta import LesionRow, V2Paths, parse_meta_csv

CLICKFIX_REL = "derivatives/unigrad-icon-registration/clickfix_report.csv"


@dataclass(frozen=True)
class Provenance:
    pid: str
    lesion_id: int
    side: str  # "bl" | "fu"
    imputed: bool  # coordinate is a registration guess, not an annotation
    sanity_bad: bool  # case-level QC flag from clickfix_report.csv


def _row_provenance(pid: str, r: LesionRow, flags: dict[str, dict]) -> list[Provenance]:
    # clickfix keys by "{pid}_{img_id_fu:02d}", one row per BL-FU image transition, not per patient.
    case = f"{pid}_{r.img_id_fu:02d}"
    sanity_bad = bool(flags.get(case, {}).get("n_sanity_bad", 0))
    out = []
    if r.cog_bl is not None:
        out.append(Provenance(pid, r.lesion_id, "bl", imputed=False, sanity_bad=sanity_bad))
    elif r.cog_backpropagated is not None:
        out.append(Provenance(pid, r.lesion_id, "bl", imputed=True, sanity_bad=sanity_bad))
    if r.cog_fu is not None:
        out.append(Provenance(pid, r.lesion_id, "fu", imputed=False, sanity_bad=sanity_bad))
    elif r.cog_propagated is not None:
        out.append(Provenance(pid, r.lesion_id, "fu", imputed=True, sanity_bad=sanity_bad))
    return out


def lesion_provenance(pid: str, root: Path) -> list[Provenance]:
    rows = parse_meta_csv(V2Paths(Path(root), pid).meta)
    flags = case_flags(root)
    out: list[Provenance] = []
    for r in rows:
        out.extend(_row_provenance(pid, r, flags))
    return out


_CASE_FLAGS_CACHE: dict[Path, dict[str, dict]] = {}


def case_flags(root: Path) -> dict[str, dict]:
    # Parsed once per root and cached at module level (R9: detected facts live at module top).
    root = Path(root)
    hit = _CASE_FLAGS_CACHE.get(root)
    if hit is not None:
        return hit
    df = pd.read_csv(root / CLICKFIX_REL)
    out = {str(row["case"]): row.to_dict() for _, row in df.iterrows()}
    _CASE_FLAGS_CACHE[root] = out
    return out
