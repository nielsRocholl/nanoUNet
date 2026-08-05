"""Offline per-patch scenario search for nanounet_build_valset.

Split out of nanounet/cli/build_valset.py (and separate from the pure-arithmetic
valset_alloc.py) to keep every file under the 200-LOC ceiling. Everything here runs ONCE,
offline: cc3d connected components, the clicked-subset target, and rejection sampling all live
here so nanounet/data/valset.py never touches any of it at validation time."""

from __future__ import annotations

from dataclasses import dataclass

import cc3d
import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from rich.table import Table

from nanounet.common import cprint
from nanounet.data.blosc2_dataset import case_spatial_shape, load_case_properties
from nanounet.data.error_table import draw_propagated_offset
from nanounet.data.patch_bbox import _sample_bbox
from nanounet.data.sampling import _sample_false_pos
from nanounet.data.valset import SCENARIOS, SIZE_BUCKETS
from nanounet.prompt.centroids import filter_centroids_in_patch

# Matches the connectivity nanounet/prompt/centroids.py uses to build the centroid sidecars
# (cc3d.connected_components default). A mismatch would desync instance ids from centroids_zyx.
CC3D_CONNECTIVITY = 26


@dataclass
class CaseInfo:
    cid: str
    shape: np.ndarray
    cts_global: list[tuple[int, int, int]]
    vols: list[float]


def case_info(case_dir: str, cid: str, cache: dict[str, "CaseInfo"]) -> "CaseInfo":
    """Cached per-case centroid/volume properties (from the sidecar, no cc3d) + spatial shape."""
    if cid in cache:
        return cache[cid]
    props = load_case_properties(case_dir, cid)
    shape = np.array(case_spatial_shape(case_dir, cid))
    cts = [tuple(int(x) for x in c) for c in props["centroids_zyx"]]
    vols = [float(v) for v in props["volume_vox"]]
    info = CaseInfo(cid, shape, cts, vols)
    cache[cid] = info
    return info


class LabelCache:
    """LRU cache of cc3d label volumes, capped so a 1500-patch build doesn't hold every val
    case's full-resolution label volume in memory at once."""

    def __init__(self, cap: int = 24):
        self._cap = cap
        self._d: dict[str, np.ndarray] = {}

    def get(self, ds, cid: str, n_expected: int) -> np.ndarray:
        if cid in self._d:
            return self._d[cid]
        with ds.open_case(cid, need_seg=True) as (_data, seg, _sp, _props):
            seg_full = np.asarray(seg[:])
        labels, n_inst = cc3d.connected_components(
            (seg_full[0] > 0).astype(np.uint8), connectivity=CC3D_CONNECTIVITY, return_N=True
        )
        assert n_inst == n_expected, (cid, n_inst, n_expected)
        if len(self._d) >= self._cap:
            self._d.pop(next(iter(self._d)))
        self._d[cid] = labels
        return labels


def _in_patch_idxs(cts_global: list[tuple[int, int, int]], bbox: list[list[int]]) -> list[int]:
    z, y, x = (slice(a, b) for a, b in bbox)
    return [
        j
        for j, (cz, cy, cx) in enumerate(cts_global)
        if z.start <= cz < z.stop and y.start <= cy < y.stop and x.start <= cx < x.stop
    ]


def _size_bucket(vols: list[float], idxs: list[int], small_max: int) -> str:
    biggest = max((vols[j] for j in idxs), default=0.0)
    return "small" if biggest <= small_max else "large"


def draw_bbox(shape, cts_global, patch_size, fg_prob: float, rng) -> list[list[int]]:
    need_to_pad = np.zeros(3, dtype=int)
    lbs, ubs, _anchor = _sample_bbox(shape, cts_global, None, fg_prob, patch_size, need_to_pad, rng)
    # _sample_bbox mixes plain int with np.int64 (patch_size is an ndarray) -- cast now so every
    # downstream consumer (clicks, JSON) only ever sees plain Python ints.
    return [[int(a), int(b)] for a, b in zip(lbs, ubs)]


def draw_lesion_clicks(idxs, case: CaseInfo, bbox, prop_cfg, rng1, rng2):
    """Displace the chosen centroids independently for draw 1/2, then filter into the patch --
    filter_centroids_in_patch converts GLOBAL displaced coords to PATCH-LOCAL in one step."""
    pslc = tuple(slice(a, b) for a, b in bbox)
    d1 = [draw_propagated_offset(case.cts_global[j], case.vols[j], prop_cfg, rng1) for j in idxs]
    d2 = [draw_propagated_offset(case.cts_global[j], case.vols[j], prop_cfg, rng2) for j in idxs]
    return filter_centroids_in_patch(d1, pslc), filter_centroids_in_patch(d2, pslc)


def build_subset_target(labels: np.ndarray, bbox, chosen_instance_ids: list[int]) -> np.ndarray:
    """packbits-ready uint8 mask, patch-shaped: 1 where the voxel belongs to one of the CLICKED
    instances. Padded with 0 (background), never -1 -- this target never sees RemoveLabelTansform."""
    lab_crop = crop_and_pad_nd(labels[None], bbox, 0)[0]
    mask = np.isin(lab_crop, chosen_instance_ids).astype(np.uint8)
    return np.packbits(mask.ravel())


def _crop_seg(ds, cid: str, bbox) -> np.ndarray:
    # Only `seg` is ever sliced here -- the builder never needs the image, that is re-cropped by
    # ValPatchDataset at load time. Pad -1 (RemoveLabelTansform territory), matching crop_patch.
    with ds.open_case(cid, need_seg=True) as (_data, seg, _sp, _props):
        return np.asarray(crop_and_pad_nd(seg, bbox, -1))


def try_lesion_free_decoy(ds, cid: str, case: CaseInfo, patch_size, rng_bbox, rng_decoy):
    bbox = draw_bbox(case.shape, case.cts_global, patch_size, 0.0, rng_bbox)
    seg_crop = _crop_seg(ds, cid, bbox)
    if seg_crop.max() > 0:
        return None
    decoy = _sample_false_pos(seg_crop, rng_decoy)
    if not decoy:
        return None
    return {
        "bbox": bbox, "clicks": list(decoy), "clicks2": list(decoy), "n_fp": 1,
        "size_bucket": "large", "instance_ids": None, "seg_crop": seg_crop,
    }


def try_foreground(scenario: str, ds, cid: str, case: CaseInfo, patch_size, prop_cfg, small_max, rng_bbox, rng_choice, rng1, rng2):
    bbox = draw_bbox(case.shape, case.cts_global, patch_size, 1.0, rng_bbox)
    idxs = _in_patch_idxs(case.cts_global, bbox)
    min_needed = 2 if scenario == "subset_clicked" else 1
    if len(idxs) < min_needed:
        return None
    chosen = idxs
    instance_ids = None
    if scenario == "subset_clicked":
        k = int(rng_choice.integers(1, len(idxs)))  # strict subset: 1 <= k < len(idxs)
        chosen = list(rng_choice.choice(idxs, size=k, replace=False))
        instance_ids = [j + 1 for j in chosen]  # cc3d labels are 1-indexed, in centroids_zyx order
    elif scenario == "none_clicked":
        chosen = []
    c1, c2 = draw_lesion_clicks(chosen, case, bbox, prop_cfg, rng1, rng2)
    if scenario in ("all_clicked", "subset_clicked") and (not c1 or not c2):
        return "reject_zero_clicks"
    size_bucket = _size_bucket(case.vols, idxs, small_max)
    return {
        "bbox": bbox,
        "clicks": c1,
        "clicks2": c2,
        "n_fp": 0,
        "size_bucket": size_bucket,
        "instance_ids": instance_ids,
        "seg_crop": _crop_seg(ds, cid, bbox),
    }


def report_composition(entries: list[dict], cohorts: list[str]) -> None:
    """Realised composition tables. These are the numbers to quote, not the requested --mix:
    per-cohort largest-remainder rounding drifts the global scenario totals by <1 per cohort."""
    t = Table(title="valset composition", box=None, padding=(0, 2))
    t.add_column("cohort", style="cyan")
    for s in SCENARIOS:
        t.add_column(s, justify="right")
    t.add_column("total", justify="right")
    for cohort in cohorts:
        row = [sum(1 for e in entries if e["cohort"] == cohort and e["scenario"] == s) for s in SCENARIOS]
        t.add_row(cohort, *[str(x) for x in row], str(sum(row)))
    cprint(t)

    t2 = Table(title="tag counts", box=None, padding=(0, 2))
    t2.add_column("tag", style="cyan")
    t2.add_column("count", justify="right")
    for v in (1, 0, -1):
        t2.add_row(f"click_inside={v}", str(sum(1 for e in entries if e["click_inside"] == v)))
    for b in SIZE_BUCKETS:
        t2.add_row(f"size={b}", str(sum(1 for e in entries if e["size_bucket"] == b)))
    # Displacement can push a lesion click out of the patch in one draw but not the other, so the
    # pair differs in click COUNT, not only placement. Production behaves the same way; the count
    # is surfaced here (and tagged per row) so val_prompt_agreement can be read with it in mind.
    n_un = sum(1 for e in entries if len(e["clicks_zyx"]) != len(e["clicks2_zyx"]))
    t2.add_row("draws_unmatched", str(n_un))
    cprint(t2)
