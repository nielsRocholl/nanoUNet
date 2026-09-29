"""Paint FU instance mask with tracking ids. BL ids stay canonical."""

from __future__ import annotations

from pathlib import Path

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
    n = int(fu_mask.max()) + 1
    lut = np.zeros(n, dtype=np.int32)
    for src, tid in m.items():
        s = int(src)
        assert 0 <= s < n, (s, n)
        lut[s] = int(tid)
    return lut[fu_mask]


CSV_HEADER = "bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id\n"


def write_empty_csv(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CSV_HEADER)
