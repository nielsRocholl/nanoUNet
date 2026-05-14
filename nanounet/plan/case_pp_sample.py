"""Uniform sample coords per foreground class for nnU-Net oversampling (vectorized, nnU-Net core)."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from nanounet.common import cprint


def sample_foreground_locations(
    seg: np.ndarray,
    classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
    seed: int = 1234,
    verbose: bool = False,
    min_num_samples: int = 10000,
    min_percent_coverage: float = 0.01,
) -> dict:
    rnd = np.random.RandomState(seed)
    req_labs = set()
    for c in classes_or_regions:
        if isinstance(c, (tuple, list)):
            req_labs.update(int(x) for x in c)
        else:
            req_labs.add(int(c))
    req_arr = np.fromiter(req_labs, dtype=np.int32)
    vm = np.isin(seg, req_arr)
    coords = np.argwhere(vm)
    seg_sel = seg[vm]
    del vm
    n = seg_sel.size
    out: dict = {}
    if n == 0:
        for c in classes_or_regions:
            k = tuple(c) if isinstance(c, (tuple, list)) else int(c)
            out[k] = []
        return out
    order = np.argsort(seg_sel, kind="stable")
    lab_sorted = seg_sel[order]
    coords_sorted = coords[order]
    chg = np.flatnonzero(lab_sorted[1:] != lab_sorted[:-1]) + 1
    starts = np.r_[0, chg]
    ends = np.r_[chg, n]
    labels_present = lab_sorted[starts]
    l2r = {int(l): (int(s), int(e)) for l, s, e in zip(labels_present, starts, ends)}
    present = set(l2r.keys())
    for c in classes_or_regions:
        is_reg = isinstance(c, (tuple, list))
        labs = tuple(int(x) for x in c) if is_reg else (int(c),)
        k = labs if is_reg else labs[0]
        if not any(lab in present for lab in labs):
            out[k] = []
            continue
        ranges, counts = [], []
        for lab in labs:
            r = l2r.get(lab)
            if r is None:
                continue
            s, e = r
            cnt = e - s
            if cnt > 0:
                ranges.append((s, e))
                counts.append(cnt)
        if not counts:
            out[k] = []
            continue
        total = int(np.sum(counts))
        tgt = min(min_num_samples, total)
        tgt = max(tgt, int(np.ceil(total * min_percent_coverage)))
        offsets = rnd.choice(total, tgt, replace=False)
        cum = np.cumsum(counts)
        which = np.searchsorted(cum, offsets, side="right")
        prev = np.concatenate(([0], cum[:-1]))
        in_range = offsets - prev[which]
        starts_for = np.fromiter((ranges[i][0] for i in which), dtype=np.int64, count=which.size)
        picked = starts_for + in_range.astype(np.int64)
        out[k] = coords_sorted[picked]
        if verbose:
            cprint(f"[dim]{c} {tgt}[/dim]")
    return out
