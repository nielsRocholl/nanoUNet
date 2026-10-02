"""The deployment pipeline as a function (segment -> nodes -> matcher -> decode), shared by exp05 and later exp07, exp08, exp09.

One scan pair in, one `Scores` record out. `Scores` holds the matcher's RAW output (pair logits and dustbin scores), the node lists
and, for every node, the annotated lesion it is identified with (IoU > 0.1, `scoring.match_nodes`); decoding is a separate pure step
(`decode_links`), so any decoder or tau can be rerun offline from a saved `.npz` and scored with `scoring.score_pair` (`score_scores`).

Three node supplies (`setting`):
  A  Lstar on both sides: nodes from the annotation via `lesionglue.data.graph.dense.build_hetero_data` (the cache builder, so it
     follows the graph-builder fix: merge-target nodes, opt-in prop-fill); node ids ARE annotated ids.
  B  protocol setting: BL = annotated instance masks, FU = the segmenter prompted with the propagated points (`fu_clicks`).
  C  points only: BL segmented from `bl_clicks`, FU from `fu_clicks`; predicted nodes are tied to annotation by IoU > 0.1 on both sides.
B and C go through the same public calls as `segtrack.track.run_case` (`preprocess_loaded` + `segment_native`, `binary_to_instances`,
`lesionglue.infer.track(..., volumes=...)`); `run_case` itself only writes CSV/masks and returns no scores, so its steps are recomposed
here to keep the raw scores.

Non-obvious choices:
- Matching uses the nanoUNet click names as instance ids (as the deployed pipeline does); an FU component no click claims gets a fresh id
  and is an EXTRA node that can steal links but is never scored as a lesion (negative id in `pairs_to_links`).
- `found_bl`/`found_fu` (what the ceiling is computed from) come from the real node lists the matcher saw, never from the annotation:
  the builder drops BL lesions without a propagated point and the degenerate case "one side has no node" is scored as all-new /
  all-disappeared instead of being skipped.
- Lesion types and propagated BL positions come from the meta CSV (`Pair.meta` and `Pair.types`, rows of the pair's FU region), exactly as segtrack does.
  `load_propagated` falls back to `cog_fu` when a row has no `cog_propagated`: that affects only lesions the cache builder drops.
- The matcher runs with its EMA weights (`use_ema=True`); tau is the checkpoint's own `dust_tau`, never tuned on scored data.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from experiments.common import MATCHER_FINAL, abort_if, problem
from experiments.scoring import PairCase, match_nodes, pairs_to_links, score_pair
from experiments.segment import Segmenter, load_segmenter
from lesionglue.data.graph.dense import build_hetero_data, node_rows
from lesionglue.data.instances.build import binary_to_instances, load_clicks
from lesionglue.data.source.meta import V2Paths, parse_meta_csv
from lesionglue.data.source.propagate import fill_propagated, load_propagated
from lesionglue.infer import graph_cfg_from_ckpt, load_matcher, track
from lesionglue.model.decode import decode_pairs
from nanounet.data.store.io import SimpleITKIO
from nanounet.infer.predict.io import preprocess_loaded
from segtrack.case import load_ct, load_instance_zyx
from segtrack.track import segment_native

SETTINGS = ("A", "B", "C")
DECODERS = ("hungarian", "sinkhorn")
SINKHORN_ITERS = 20  # the training config's sinkhorn_iters
SCORE_KEYS = ("bl_ids", "fu_ids", "pair", "dust_bl", "dust_fu", "bl_ann", "fu_ann")


@dataclass
class Pipeline:
    """Everything loaded once per process: the matcher (EMA weights used at inference), the segmenter (None if only setting A runs), tau."""

    matcher: object
    segmenter: Segmenter | None
    tau: float  # the matcher checkpoint's dust_tau
    device: str
    ckpt: Path


@dataclass(frozen=True)
class Pair:
    """Explicit paths of one scan pair (any dataset layout can build one; `longitudinal_pair` does it for Longitudinal-CT)."""

    pid: str
    stem: str
    region: int  # img_id_fu of the pair: selects the meta rows and names the files
    bl_img: Path
    fu_img: Path
    bl_clicks: Path  # true BL centroids (prompts of setting C)
    fu_clicks: Path  # propagated points (prompts of B and C)
    bl_mask: Path  # annotated instance masks (Lstar nodes, matching targets)
    fu_mask: Path
    meta: Path  # propagated BL positions in the FU frame: meta CSV (rows of `region`), slim CSV `lesion_id,z,y,x` or FU-frame JSON
    types: Path | None = None  # CSV with lesion_id, lesion_type (the meta CSV for Longitudinal-CT); None = every lesion gets type `unclear`


@dataclass
class Scores:
    """One pair's matcher output. `*_ann` = annotated lesion id a node is identified with, 0 = extra node (under A every node is annotated)."""

    bl_ids: np.ndarray
    fu_ids: np.ndarray
    pair: np.ndarray  # (n_bl, n_fu) raw pair logits
    dust_bl: np.ndarray
    dust_fu: np.ndarray
    bl_ann: np.ndarray
    fu_ann: np.ndarray
    t_seg: float = 0.0
    t_track: float = 0.0


def dominant_region(meta_csv: Path) -> int:
    """img_id_fu with the most meta rows (ties: the smallest id), the scan pair the paper scores for this patient."""
    n = pd.read_csv(meta_csv, usecols=["img_id_fu"])["img_id_fu"].value_counts()
    return int(n[n == n.max()].index.min())


def longitudinal_pair(root: Path, pid: str) -> Pair:
    """Pair of the dominant FU region in the Longitudinal-CT layout (inputsTrBL|FU, targetsTrBL|FU, meta)."""
    root = Path(root)
    meta = root / "meta" / f"{pid}.csv"
    region = dominant_region(meta)
    stem = f"{pid}_{region:02d}"
    return Pair(pid, stem, region, root / "inputsTrBL" / f"{stem}.nii.gz", root / "inputsTrFU" / f"{stem}.nii.gz", root / "inputsTrBL" / f"{stem}.json",
                root / "inputsTrFU" / f"{stem}.json", root / "targetsTrBL" / f"{stem}.nii.gz", root / "targetsTrFU" / f"{stem}.nii.gz", meta, meta)


def load_pipeline(device: str, *, matcher_ckpt: Path = MATCHER_FINAL, segmenter: bool = True) -> Pipeline:
    """Load the matcher and (unless segmenter=False) the segmenter once. tau = the matcher checkpoint's own `dust_tau`."""
    mod = load_matcher(matcher_ckpt, device)
    tau = getattr(mod.hparams, "dust_tau", None)
    abort_if([problem(f"matcher checkpoint {matcher_ckpt} has no dust_tau in its hyper-parameters", "a LesionGlue Lightning checkpoint trained with dust_tau",
                      "point --matcher-ckpt at a checkpoint written by lesionglue_train")] if tau is None else [])
    return Pipeline(mod, load_segmenter(device) if segmenter else None, float(tau), device, Path(matcher_ckpt))


def segment_scan(pl: Pipeline, img: Path, clicks: Path) -> dict:
    """Prompted segmentation of one scan: {inst (native zyx int32, ids = click names), vol (ct, affine, spacing for the matcher), props, t_seg}."""
    sg = pl.segmenter
    assert sg is not None, "this Pipeline was loaded with segmenter=False"
    data, props, vol = load_ct(img)
    t0 = time.perf_counter()
    pack = preprocess_loaded(data, props, str(clicks), sg.pl, sg.cm, sg.dj)
    pred, _ = segment_native(sg.net, sg.lm, sg.cfg, sg.pl, sg.cm, sg.dev, pack, use_tta=sg.use_tta, batch_size=sg.batch_size)
    return {"inst": binary_to_instances(pred, load_clicks(clicks)), "vol": vol, "props": props, "t_seg": time.perf_counter() - t0}


def annotated_scan(img: Path, mask: Path) -> dict:
    """The Lstar side of setting B: annotated instance mask + CT (same dict shape as `segment_scan`)."""
    _, props, vol = load_ct(img)
    inst, _ = load_instance_zyx(mask)
    return {"inst": inst, "vol": vol, "props": props, "t_seg": 0.0}


def tie(node_ids: np.ndarray, pred_inst: np.ndarray, gt_mask: Path) -> np.ndarray:
    """Annotated lesion id of every node (IoU > 0.1 against the annotated instance mask, best overlap wins), 0 for an extra node."""
    hit = match_nodes(pred_inst, load_instance_zyx(gt_mask)[0])
    return np.array([hit.get(int(i), 0) for i in node_ids], dtype=np.int64)


def _raw(out, n_bl: int, n_fu: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return out.pair.reshape(n_bl, n_fu).cpu().numpy(), out.dust_bl.cpu().numpy(), out.dust_fu.cpu().numpy()


def run_graph(pl: Pipeline, root: Path, pair: Pair, *, keep_unclear: bool = False, prop_fill: str = "none") -> Scores:
    """Setting A: annotated nodes from the dataset's own graph builder (the FU region of `pair`), matcher on them, raw scores."""
    graphs = [g for g in build_hetero_data(pair.pid, root, graph_cfg_from_ckpt(pl.matcher, 8), keep_unclear, prop_fill) if int(g.img_id_fu_used) == pair.region]
    if not graphs:  # the builder makes no graph when one side has no node: nothing can be linked, but the nodes of the other side exist (all new / all disappeared)
        rows = [r for r in parse_meta_csv(V2Paths(Path(root), pair.pid).meta, keep_unclear) if r.img_id_fu == pair.region]
        rows = fill_propagated(rows, Path(root), pair.pid, pair.region)[0] if prop_fill == "unigradicon" else rows
        bl, fu = node_rows(rows, pair.pid)
        bl_ids, fu_ids = np.array(sorted(bl), dtype=np.int64), np.array(sorted(fu), dtype=np.int64)
        return Scores(bl_ids, fu_ids, np.zeros((len(bl_ids), len(fu_ids)), np.float32), np.zeros(len(bl_ids), np.float32), np.zeros(len(fu_ids), np.float32), bl_ids.copy(), fu_ids.copy())
    data, t0 = graphs[0], time.perf_counter()
    bl_ids, fu_ids = data["bl"].lesion_id.cpu().numpy(), data["fu"].lesion_id.cpu().numpy()
    with torch.no_grad():
        out = pl.matcher.predict_batch(Batch.from_data_list([data.to(next(pl.matcher.parameters()).device)]), use_ema=True)
    pair_l, d_bl, d_fu = _raw(out, len(bl_ids), len(fu_ids))
    return Scores(bl_ids, fu_ids, pair_l, d_bl, d_fu, bl_ids.copy(), fu_ids.copy(), 0.0, time.perf_counter() - t0)


def run_masks(pl: Pipeline, pair: Pair, setting: str) -> tuple[Scores, dict]:
    """Settings B and C: nodes from masks (annotated or segmented), matcher through `lesionglue.infer.track(volumes=...)`, raw scores.
    Returns (Scores, {"bl": scan dict, "fu": scan dict}) so the caller can write the instance masks."""
    assert setting in ("B", "C"), f"run_masks handles B and C, got {setting!r}"
    fu = segment_scan(pl, pair.fu_img, pair.fu_clicks)
    bl = annotated_scan(pair.bl_img, pair.bl_mask) if setting == "B" else segment_scan(pl, pair.bl_img, pair.bl_clicks)
    t0 = time.perf_counter()
    if bl["inst"].any() and fu["inst"].any():
        r = track(pair.bl_img, pair.bl_img, pair.fu_img, pair.fu_img, pair.meta, pl.ckpt, decode="hungarian", device=pl.device, matcher=pl.matcher,
                  sinkhorn_tau=pl.tau, use_ema=True, types_csv=pair.types, img_id=pair.region,
                  volumes=(*bl["vol"], np.ascontiguousarray(bl["inst"].transpose(2, 1, 0)), *fu["vol"], np.ascontiguousarray(fu["inst"].transpose(2, 1, 0))))
        bl_ids, fu_ids, pair_l, d_bl, d_fu = r.bl_ids, r.fu_ids, r.pair, r.dust_bl, r.dust_fu
    else:  # one side is empty: the builder returns no graph; nodes are whatever each side holds, nothing can be linked
        bl_all = np.unique(bl["inst"][bl["inst"] > 0])
        prop, _ = load_propagated(pair.meta, bl_all.tolist(), img_id=pair.region)
        bl_ids, fu_ids = np.array([i for i in bl_all if int(i) in prop], dtype=np.int64), np.unique(fu["inst"][fu["inst"] > 0]).astype(np.int64)
        pair_l, d_bl, d_fu = np.zeros((len(bl_ids), len(fu_ids)), np.float32), np.zeros(len(bl_ids), np.float32), np.zeros(len(fu_ids), np.float32)
    bl_ann = np.asarray(bl_ids, dtype=np.int64) if setting == "B" else tie(bl_ids, bl["inst"], pair.bl_mask)
    return Scores(np.asarray(bl_ids), np.asarray(fu_ids), pair_l, d_bl, d_fu, bl_ann, tie(fu_ids, fu["inst"], pair.fu_mask), bl["t_seg"] + fu["t_seg"],
                  time.perf_counter() - t0), {"bl": bl, "fu": fu}


def run_setting(pl: Pipeline, root: Path, pair: Pair, setting: str, *, keep_unclear: bool = False, prop_fill: str = "none") -> tuple[Scores, dict | None]:
    """One scan pair under setting A, B or C -> (Scores, scans or None). `root`, `keep_unclear` and `prop_fill` are only read by A (the graph builder)."""
    if setting == "A":
        return run_graph(pl, root, pair, keep_unclear=keep_unclear, prop_fill=prop_fill), None
    return run_masks(pl, pair, setting)


def decode_links(s: Scores, decoder: str, tau: float) -> set[tuple[int, int]]:
    """Decoded links in annotated-id space (endpoints < 0 = extra nodes), from the stored raw scores; `decode_pairs` does the decoding."""
    n_bl, n_fu = len(s.bl_ids), len(s.fu_ids)
    if n_bl == 0 or n_fu == 0:
        return set()
    pairs = decode_pairs(decoder, torch.from_numpy(s.pair.reshape(-1)), torch.from_numpy(s.dust_bl), torch.from_numpy(s.dust_fu), n_bl, n_fu,
                         thresh=0.5, sinkhorn_iters=SINKHORN_ITERS, sinkhorn_tau=tau)
    return pairs_to_links(pairs, [int(a) or None for a in s.bl_ann], [int(a) or None for a in s.fu_ann])


def score_scores(case: PairCase, s: Scores | None, decoder: str, tau: float, *, exclude_unclear: bool) -> dict[str, int]:
    """Identity counts of one pair (`scoring.score_pair`). `s=None` = the pipeline failed on this patient: no node, no link, all lesions missed."""
    if s is None:
        return score_pair(replace(case, found_bl=set(), found_fu=set(), pred_links=set()), exclude_unclear=exclude_unclear)
    found_bl, found_fu = {int(a) for a in s.bl_ann if a}, {int(a) for a in s.fu_ann if a}
    return score_pair(replace(case, found_bl=found_bl, found_fu=found_fu, pred_links=decode_links(s, decoder, tau)), exclude_unclear=exclude_unclear)


def save_scores(path: Path, s: Scores) -> None:
    """Raw scores + node tables as one .npz (atomic), enough to decode and score again offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("part_" + path.name)
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **{k: getattr(s, k) for k in SCORE_KEYS}, t_seg=s.t_seg, t_track=s.t_track)
    tmp.replace(path)


def load_scores(path: Path) -> Scores:
    z = np.load(path)
    return Scores(*(z[k] for k in SCORE_KEYS), float(z["t_seg"]), float(z["t_track"]))


def write_links_csv(path: Path, s: Scores, tau: float) -> None:
    """matches.csv: every decoded link of both decoders as node ids and annotated ids (0 = extra node)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "bl_node", "fu_node", "bl_ann", "fu_ann"])
        for dec in DECODERS:
            if len(s.bl_ids) and len(s.fu_ids):
                pairs = decode_pairs(dec, torch.from_numpy(s.pair.reshape(-1)), torch.from_numpy(s.dust_bl), torch.from_numpy(s.dust_fu), len(s.bl_ids), len(s.fu_ids),
                                     thresh=0.5, sinkhorn_iters=SINKHORN_ITERS, sinkhorn_tau=tau)
                for i, j in pairs.tolist():
                    w.writerow([dec, int(s.bl_ids[i]), int(s.fu_ids[j]), int(s.bl_ann[i]), int(s.fu_ann[j])])


def write_instances(path: Path, inst: np.ndarray, props: dict) -> None:
    """Instance mask (int32, click names as ids) on the scan's native grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleITKIO().write_seg(inst.astype(np.int32), str(path), props)
