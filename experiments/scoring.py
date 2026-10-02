"""Shared metrics for the paper experiments: identity counts, identity ceiling, edge F1, node matching, per-lesion segmentation scores, patient-level bootstrap.

Pure numpy/scipy (no torch, no Lightning). Decoding is NOT done here: predicted pairs come from
`lesionglue.model.decode.decode_pairs`; `pairs_to_links` only renames its (row, column) output into annotated ids.

DEFINITIONS (paper Sec. "What we measure"; the same words are in `DEFINITIONS` for every results.json)

Node supply. The lesion set each side hands the matcher: `Lstar` (the annotation) or `Lhat` (the segmenter's output). A predicted
lesion is *identified with* an annotated lesion when their IoU > `IOU_HIT` (0.1, `nanounet.score`); the annotated lesion with the
highest IoU wins (exact ties: the smaller id), and `node_overlaps` lists every overlapping pair so one-to-many cases are recorded, not
hidden. A predicted lesion identified with nothing is an *extra node*: it may steal links but is never scored as a lesion. Annotated
lesions that own a node are `found_bl` / `found_fu` in a `PairCase` (all of them under Lstar only if the graph builder made a node for
every lesion: set them from the actual node lists, never assume).

Per-lesion, per-class recall (one decision per lesion; ids are the annotated lesion ids of the scan pair):
  unchanged    BL lesion is correct iff the decoded links contain (its id, its annotated FU partner);
  disappeared  BL lesion is correct iff its node exists and has no decoded link at all;
  new          FU lesion is correct iff its node exists and no decoded link enters it;
  merged       BL lesion is correct iff the decoded links contain (its id, the FU lesion it merged into): per contributing lesion, not
               per event (a k-way merge is k decisions).
A lesion whose required node is missing (Lhat) is counted WRONG in its class. Extra links of a linked lesion are not a recall error;
they cost edge precision. `recall_<class> = ok / tot`.

Identity ceiling per class = the fraction of that class's lesions whose required nodes both exist (unchanged/merged: BL node and FU
partner node; disappeared: BL node; new: FU node). It is 1 only if every annotated lesion owns a node; `matcher_error_<class> =
ceiling - recall` is the matcher's share of the error, `1 - ceiling` the node supply's.

Edge P/R/F1. The decoded links `pred_links` are a set of (annotated BL id, annotated FU id) edges over the whole bipartite graph and
`links` the annotated edges. tp = |pred and true|, fp = |pred minus true| (this includes every edge that touches an extra node: an
endpoint < 0 in `pred_links` means an extra node, `pairs_to_links` numbers them -1, -2, ...), fn = |true minus pred| (an edge whose
lesion has no node is a fn). Precision = tp/(tp+fp), recall = tp/(tp+fn), F1 = 2tp/(2tp+fp+fn). Never accuracy over candidate pairs.

Graph builder gap (checked 2026-09-30, `lesionglue.data.graph.dense.node_rows`): a BL lesion without `cog_propagated` gets no node (127
of 4396 headline lesions: 41 unchanged, 27 disappeared, 59 merged) and a merged FU lesion never gets a node (merge targets are MERGING
rows, which `node_rows` does not turn into FU nodes), so on the current caches the "Lstar" ceiling is 0.983 unchanged, 0.978 disappeared,
1.0 new and 0.0 merged, and `_add_graph` credits a merged lesion for "no link". That is why `found_*` must come from the real node lists.

Reporting rule. Never quote one class alone (a matcher that links nothing scores 1.0 on disappeared): `bootstrap`/`pooled` always give
all four classes, `recall_macro` and the edge metrics; an undefined ratio (no lesion of that class) is NaN, never 0.

linking_unclear. `load_pairs(include_unclear=False)` (headline) drops every meta row flagged `linking_unclear` (and merge contributors
whose target is flagged), exactly as the graph cache built by `lesionglue.data.source.meta.parse_meta_csv` does. `include_unclear=True`
keeps all rows (the paper's counts: 300 patients, 4530 lesions = 2424 UNCHANGED + 1382 DISAPPEARING + 558 NEWLYAPPEARING + 166
MERGING) and lists the flagged ids in `PairCase.unclear`. `score_pair(..., exclude_unclear=True)` treats the flagged nodes as absent:
a lesion is dropped from tot/ok/ceil if it or any annotated partner is flagged (counted in `n_unclear`), and every true or predicted
edge touching a flagged id is dropped from tp/fp/fn (a disappeared lesion linked only to a flagged node therefore still counts as
unlinked). With `exclude_unclear=False` everything is scored: that is the with-unclear sensitivity row.

Patient pair. One scan pair per patient = the meta rows whose `img_id_fu` equals the patient's most frequent `img_id_fu` (ties: the
smallest id, which is what pandas' idxmax gives on the two tied patients).

Segmentation per lesion (`score_lesion`): Dice, NSD at 1 mm (`NSD_TOL_MM`), detection = IoU > `IOU_HIT`, using
`nanounet.score.dice/iou/nsd`. The prediction credited to a GT lesion is the 18-connected component of the predicted foreground with the
largest overlap (`lesion_prediction`; the same rule as `nanounet.score.score_case`, protocol longiseg_lesion_v1); no overlap means an
empty prediction (Dice 0). Prompt drop = Dice with the prompt channel on minus zeroed (two `score_lesion` calls, subtracted by the caller).

Bootstrap. Resample PATIENTS (never lesions) with replacement, `np.random.default_rng(BOOTSTRAP_SEED)`, B = `BOOTSTRAP_B` draws of n
patients (one `rng.integers(0, n, n)` per replicate, patients in sorted id order), percentile 95 interval [2.5, 97.5]; the point estimate is
the statistic on the full sample. A statistic that is undefined in a resample (class absent) is skipped by the percentile. Paired deltas
reuse the same draws for both arms. For case-mean metrics (Dice per case, then mean over cases) give each patient ONE item, its own mean.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import csv
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi

from experiments.common import BOOTSTRAP_B, BOOTSTRAP_SEED
from nanounet.score import IOU_HIT, NSD_TOL_MM, dice, iou, nsd

CLASSES = ("unchanged", "disappeared", "new", "merged")
TOPOLOGY = {"UNCHANGED": "UNCHANGED", "DISAPPEARING": "DISAPPEARED", "DISAPPEARED": "DISAPPEARED", "NEWLYAPPEARING": "NEWLYAPPEARING", "MERGING": "MERGED", "MERGED": "MERGED"}
CLASS_OF = {"UNCHANGED": "unchanged", "DISAPPEARED": "disappeared", "NEWLYAPPEARING": "new", "MERGED": "merged"}
COUNT_KEYS = tuple(f"{c}_{k}" for c in CLASSES for k in ("ok", "tot", "ceil")) + ("tp", "fp", "fn", "fp_extra", "n_unclear")
METRICS = (tuple(f"recall_{c}" for c in CLASSES) + ("recall_macro",) + tuple(f"ceiling_{c}" for c in CLASSES)
           + tuple(f"matcher_error_{c}" for c in CLASSES) + ("edge_precision", "edge_recall", "edge_f1"))
DEFINITIONS = {"iou_hit": IOU_HIT, "nsd_tol_mm": NSD_TOL_MM, "node_supply": "Lstar"}  # override node_supply per run ("Lhat", "Lstar/Lhat")
CI_PERCENTILES = (2.5, 97.5)
META_COLUMNS = ("lesion_id", "img_id_fu", "topology_class", "merged_into")


@dataclass
class PairCase:
    """One patient's scan pair: everything needed to score identity (ids are annotated lesion ids)."""

    pid: str
    bl_ids: list[int]                       # annotated BL lesions (UNCHANGED, DISAPPEARED, MERGED)
    fu_ids: list[int]                       # annotated FU lesions (UNCHANGED partners, NEWLYAPPEARING, merge targets)
    topology: dict[int, str]                # BL id -> UNCHANGED|DISAPPEARED|MERGED ; FU-only ids -> NEWLYAPPEARING
    links: set[tuple[int, int]]             # annotated (bl_id, fu_id) edges: (i, i) if unchanged, (i, merged_into) if merged
    found_bl: set[int]                      # annotated ids that own a node on the BL side (== all only if every lesion got a node)
    found_fu: set[int]
    pred_links: set[tuple[int, int]]        # decoded links in annotated-id space; endpoints < 0 are extra nodes
    unclear: set[int] = frozenset()         # linking_unclear ids (with include_unclear=True); empty in the headline pairs


def score_pair(c: PairCase, *, exclude_unclear: bool = True) -> dict[str, int]:
    """Per class `<class>_{ok,tot,ceil}` plus `tp, fp, fn, fp_extra` (fp that touch an extra node) and `n_unclear` (lesions dropped)."""
    skip = frozenset(c.unclear) if exclude_unclear else frozenset()
    partners: dict[int, list[int]] = {}
    for b, f in c.links:
        partners.setdefault(b, []).append(f)
    true = {(b, f) for b, f in c.links if b not in skip and f not in skip}
    pred = {(b, f) for b, f in c.pred_links if b not in skip and f not in skip}
    for b, f in pred:
        assert (b < 0 or b in c.found_bl) and (f < 0 or f in c.found_fu), f"{c.pid}: decoded link {(b, f)} uses an annotated lesion without a node (extra nodes need negative ids)"
    linked_bl, linked_fu = {b for b, _ in pred}, {f for _, f in pred}
    out = dict.fromkeys(COUNT_KEYS, 0)

    def add(cls: str, ok: bool, ceil: bool) -> None:
        out[f"{cls}_tot"] += 1
        out[f"{cls}_ok"] += int(ok)
        out[f"{cls}_ceil"] += int(ceil)

    for b in c.bl_ids:
        t = c.topology[b]
        parts = partners.get(b, [])
        assert t == "DISAPPEARED" or parts, f"{c.pid}: {t} lesion {b} has no annotated partner in links"
        if b in skip or any(f in skip for f in parts):
            out["n_unclear"] += 1
        elif t == "DISAPPEARED":
            add("disappeared", b in c.found_bl and b not in linked_bl, b in c.found_bl)
        else:
            add(CLASS_OF[t], all((b, f) in pred for f in parts), b in c.found_bl and all(f in c.found_fu for f in parts))
    for f in c.fu_ids:
        if c.topology.get(f) == "NEWLYAPPEARING":
            if f in skip:
                out["n_unclear"] += 1
            else:
                add("new", f in c.found_fu and f not in linked_fu, f in c.found_fu)
    out["tp"], out["fp"], out["fn"] = len(pred & true), len(pred - true), len(true - pred)
    out["fp_extra"] = sum(1 for b, f in pred if b < 0 or f < 0)
    assert all(out[f"{k}_ok"] <= out[f"{k}_ceil"] <= out[f"{k}_tot"] for k in CLASSES), (c.pid, out)
    return out


def pairs_to_links(pairs: np.ndarray, bl_nodes: Sequence[int | None], fu_nodes: Sequence[int | None]) -> set[tuple[int, int]]:
    """`decode_pairs` output (m, 2) of (row, column) -> annotated-id edges. bl_nodes[i] / fu_nodes[j] = annotated id of matcher row i /
    column j, or None for an extra node, which gets its own negative id (-(i+1)) so each edge touching an extra stays a separate fp."""
    bl = [-(i + 1) if a is None else int(a) for i, a in enumerate(bl_nodes)]
    fu = [-(j + 1) if a is None else int(a) for j, a in enumerate(fu_nodes)]
    return {(bl[i], fu[j]) for i, j in np.asarray(pairs, dtype=np.int64).reshape(-1, 2).tolist()}


def node_overlaps(pred_inst: np.ndarray, gt_inst: np.ndarray, iou_hit: float = IOU_HIT) -> list[tuple[int, int, float]]:
    """Every (predicted id, annotated id, IoU) with IoU > iou_hit between two instance-label volumes on the same grid (0 = background),
    sorted by predicted id, then IoU descending, then annotated id. This is the full table behind `match_nodes`."""
    assert pred_inst.shape == gt_inst.shape, f"instance volumes differ in shape: {pred_inst.shape} vs {gt_inst.shape}"
    area_p, area_g = np.bincount(pred_inst[pred_inst > 0].astype(np.int64)), np.bincount(gt_inst[gt_inst > 0].astype(np.int64))
    both = (pred_inst > 0) & (gt_inst > 0)
    if not both.any():
        return []
    width = max(len(area_g), 1)
    key, inter = np.unique(pred_inst[both].astype(np.int64) * width + gt_inst[both].astype(np.int64), return_counts=True)
    p, g = key // width, key % width
    j = inter / (area_p[p] + area_g[g] - inter)
    order = np.lexsort((g, -j, p))
    return [(int(p[i]), int(g[i]), float(j[i])) for i in order if j[i] > iou_hit]


def match_nodes(pred_inst: np.ndarray, gt_inst: np.ndarray, iou_hit: float = IOU_HIT) -> dict[int, int]:
    """predicted id -> annotated id for every predicted lesion with IoU > iou_hit against some annotated lesion (best IoU wins; exact
    ties: smaller annotated id). Predicted ids that are absent are extra nodes. Several predicted ids may map to one annotated id:
    inspect `node_overlaps` for the whole table."""
    best: dict[int, int] = {}
    for p, g, _ in node_overlaps(pred_inst, gt_inst, iou_hit):
        best.setdefault(p, g)
    return best


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def _metrics(v: np.ndarray) -> np.ndarray:
    """Metric vector (order METRICS) from a summed count vector (order COUNT_KEYS); the one place the formulas live."""
    n = dict(zip(COUNT_KEYS, v.tolist()))
    recall = [_ratio(n[f"{k}_ok"], n[f"{k}_tot"]) for k in CLASSES]
    ceiling = [_ratio(n[f"{k}_ceil"], n[f"{k}_tot"]) for k in CLASSES]
    gap = [_ratio(n[f"{k}_ceil"] - n[f"{k}_ok"], n[f"{k}_tot"]) for k in CLASSES]
    defined = [r for r in recall if r == r]
    tp, fp, fn = n["tp"], n["fp"], n["fn"]
    return np.array(recall + [sum(defined) / len(defined) if defined else float("nan")] + ceiling + gap
                    + [_ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(2 * tp, 2 * tp + fp + fn)])


def _vector(counts: dict) -> np.ndarray:
    return np.array([counts[k] for k in COUNT_KEYS], dtype=float)


def pooled(counts: list[dict]) -> dict:
    """Sum the per-pair counts of `score_pair` and derive `METRICS` (recall_/ceiling_/matcher_error_ per class, recall_macro, edge_*)."""
    v = np.sum([_vector(c) for c in counts], axis=0) if counts else np.zeros(len(COUNT_KEYS))
    return {**{k: int(x) for k, x in zip(COUNT_KEYS, v)}, **dict(zip(METRICS, _metrics(v).tolist()))}


def _draws(n: int, b: int, seed: int) -> Iterable[np.ndarray]:
    """B index draws of n patients each: replicate r is the r-th `rng.integers(0, n, n)` of one seeded generator."""
    rng = np.random.default_rng(seed)
    for _ in range(b):
        yield rng.integers(0, n, size=n)


def _interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column: the statistic was undefined in every resample
        lo, hi = np.nanpercentile(values, CI_PERCENTILES, axis=0)
    return lo, hi


def _boot(by_patient: dict[str, list], stat: Callable, b: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared core: (point, lo, hi); `stat` maps the concatenated items of a patient sample to a float or a 1-D array."""
    assert by_patient, "bootstrap needs at least one patient"
    pids = sorted(by_patient)
    items = [list(by_patient[p]) for p in pids]
    point = np.asarray(stat([x for it in items for x in it]), dtype=float)
    values = np.empty((b,) + point.shape)
    for r, idx in enumerate(_draws(len(pids), b, seed)):
        values[r] = stat([x for i in idx for x in items[i]])
    lo, hi = _interval(values)
    return point, lo, hi


def _boot_delta(a: dict[str, list], c: dict[str, list], stat: Callable, b: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert set(a) == set(c), f"paired delta needs the same patients in both arms: only in a {sorted(set(a) - set(c))[:5]}, only in b {sorted(set(c) - set(a))[:5]}"
    pids = sorted(a)
    ia, ic = [list(a[p]) for p in pids], [list(c[p]) for p in pids]
    point = np.asarray(stat([x for it in ia for x in it]), dtype=float) - np.asarray(stat([x for it in ic for x in it]), dtype=float)
    values = np.empty((b,) + point.shape)
    for r, idx in enumerate(_draws(len(pids), b, seed)):
        values[r] = np.asarray(stat([x for i in idx for x in ia[i]]), dtype=float) - np.asarray(stat([x for i in idx for x in ic[i]]), dtype=float)
    lo, hi = _interval(values)
    return point, lo, hi


def bootstrap_stat(by_patient: dict[str, list], stat: Callable[[list], float], b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(point, lo, hi) of `stat` under patient-level resampling; `stat` receives the concatenated item lists of the resampled patients."""
    point, lo, hi = _boot(by_patient, stat, b, seed)
    return float(point), float(lo), float(hi)


def paired_delta_stat(by_patient_a: dict[str, list], by_patient_b: dict[str, list], stat: Callable[[list], float], b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(point, lo, hi) of stat(a) - stat(b) on the same patients with the same resamples (both arms must hold the same patient ids)."""
    point, lo, hi = _boot_delta(by_patient_a, by_patient_b, stat, b, seed)
    return float(point), float(lo), float(hi)


def _metric_stat(items: list) -> np.ndarray:
    return _metrics(np.sum(items, axis=0))


def bootstrap(counts_by_patient: dict[str, dict], b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, tuple[float, float, float]]:
    """{metric in METRICS: (point, lo, hi)} from per-patient `score_pair` counts, patient-level percentile 95. Same resampling core as
    `bootstrap_stat`, evaluated for all metrics in one pass per replicate."""
    point, lo, hi = _boot({p: [_vector(c)] for p, c in counts_by_patient.items()}, _metric_stat, b, seed)
    return {m: (float(point[i]), float(lo[i]), float(hi[i])) for i, m in enumerate(METRICS)}


def paired_delta(a: dict[str, dict], b: dict[str, dict], n_boot: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, tuple[float, float, float]]:
    """{metric: (delta, lo, hi)} of metric(a) - metric(b) from per-patient counts of two systems on the same patients (same resamples)."""
    point, lo, hi = _boot_delta({p: [_vector(c)] for p, c in a.items()}, {p: [_vector(c)] for p, c in b.items()}, _metric_stat, n_boot, seed)
    return {m: (float(point[i]), float(lo[i]), float(hi[i])) for i, m in enumerate(METRICS)}


def _meta_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"meta CSV not found: {path}\nExpected <root>/meta/<patient>.csv as in Longitudinal-CT.\nFix: pass root=/nnunet_data/Longitudinal-CT (or drop the patient from the list)")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [k for k in META_COLUMNS if k not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks column(s) {missing}\nExpected the Longitudinal-CT meta columns {list(META_COLUMNS)} (+ optional linking_unclear).\nFix: regenerate the meta CSV in the Longitudinal-CT layout")
        return list(reader)


def load_pairs(root: str | Path, pids: Iterable[str], *, include_unclear: bool = False) -> dict[str, PairCase]:
    """{pid: PairCase} in the annotated-id space, one pair per patient (rows of the dominant `img_id_fu`) from `<root>/meta/<pid>.csv`.

    found_bl/found_fu are set to ALL annotated ids and pred_links is empty: the caller overwrites them with `dataclasses.replace(case,
    found_bl=..., found_fu=..., pred_links=...)` from the real node lists and decoded links. include_unclear=False drops flagged rows
    (headline); True keeps them and records their ids in `unclear` (see the module docstring)."""
    out: dict[str, PairCase] = {}
    for pid in pids:
        path = Path(root) / "meta" / f"{pid}.csv"
        raw = _meta_rows(path)
        n_fu = Counter(int(r["img_id_fu"]) for r in raw)
        dominant = max(n_fu, key=lambda k: (n_fu[k], -k))
        rows = []
        for r in raw:
            if int(r["img_id_fu"]) != dominant:
                continue
            topo = TOPOLOGY.get(r["topology_class"].strip())
            if topo is None:
                raise ValueError(f"{path}: unknown topology_class {r['topology_class']!r} (lesion {r['lesion_id']})\nExpected one of {sorted(TOPOLOGY)}.\nFix: correct the topology_class cell in the meta CSV")
            into = r["merged_into"].strip()
            if topo == "MERGED" and not into:
                raise ValueError(f"{path}: MERGING lesion {r['lesion_id']} has no merged_into\nExpected the id of the follow-up lesion it merged into.\nFix: fill the merged_into cell in the meta CSV")
            rows.append({"id": int(r["lesion_id"]), "topo": topo, "into": int(float(into)) if topo == "MERGED" else None,
                         "flag": r.get("linking_unclear", "").strip().lower() in ("true", "1")})
        flagged = {r["id"] for r in rows if r["flag"]}
        drop = flagged | {r["id"] for r in rows if r["into"] in flagged}  # a merge whose target is unclear is unscorable too
        keep = rows if include_unclear else [r for r in rows if r["id"] not in drop]
        topology = {r["id"]: r["topo"] for r in keep}
        bl_ids = sorted(i for i, t in topology.items() if t != "NEWLYAPPEARING")
        fu_ids = sorted({i for i, t in topology.items() if t in ("UNCHANGED", "NEWLYAPPEARING")} | {r["into"] for r in keep if r["into"] is not None})
        links = {(i, i) for i, t in topology.items() if t == "UNCHANGED"} | {(r["id"], r["into"]) for r in keep if r["into"] is not None}
        out[pid] = PairCase(pid, bl_ids, fu_ids, topology, links, set(bl_ids), set(fu_ids), set(), frozenset(drop) if include_unclear else frozenset())
    return out


def label_components(pred_fg: np.ndarray) -> np.ndarray:
    """18-connected component labels of a predicted foreground (call once per volume, then `lesion_prediction` per GT lesion)."""
    return ndi.label(pred_fg > 0, structure=ndi.generate_binary_structure(3, 2))[0]


def lesion_prediction(components: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """The predicted component credited to one GT lesion (bool mask, same grid): largest overlap with `gt`, empty if none."""
    hit = components[gt > 0]
    hit = hit[hit > 0]
    return components == np.bincount(hit).argmax() if hit.size else np.zeros(gt.shape, dtype=bool)


def score_lesion(pred: np.ndarray, gt: np.ndarray, spacing_zyx: tuple[float, float, float]) -> dict[str, float]:
    """{dsc, nsd, iou, hit} for ONE lesion: `pred` = the prediction credited to it, `gt` = its annotated mask, same grid, spacing in mm (z, y, x).
    nsd is NSD at NSD_TOL_MM; hit = 1.0 when iou > IOU_HIT. Empty gt gives NaN iou/nsd (drop it, as nanounet.score does)."""
    pred, gt = pred > 0, gt > 0
    if (pred | gt).any():
        z, y, x = np.where(pred | gt)
        box = (slice(z.min(), z.max() + 1), slice(y.min(), y.max() + 1), slice(x.min(), x.max() + 1))
        pred, gt = pred[box], gt[box]  # every metric only sees the union, so cropping is exact and avoids full-volume passes
    j = iou(gt, pred)
    return {"dsc": dice(gt, pred), "nsd": nsd(gt, pred, spacing_zyx), "iou": j, "hit": float(j > IOU_HIT)}


def score_lesions(pred_fg: np.ndarray, gt_inst: np.ndarray, spacing_zyx: tuple[float, float, float], ids: Iterable[int] | None = None) -> list[dict]:
    """`score_lesion` for every annotated lesion of an instance-label volume (or only `ids`): rows {id, dsc, nsd, iou, hit}."""
    comp = label_components(pred_fg)
    rows = []
    for lid in sorted(int(i) for i in (np.unique(gt_inst[gt_inst > 0]) if ids is None else ids)):
        gt = gt_inst == lid
        if gt.any():
            rows.append({"id": lid, **score_lesion(lesion_prediction(comp, gt), gt, spacing_zyx)})
    return rows
