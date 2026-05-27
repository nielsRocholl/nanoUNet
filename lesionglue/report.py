"""Report metrics: decoded GNN rows, nearest-mask baseline, dense edge counts."""

from __future__ import annotations

from pathlib import Path

import gc
import numpy as np
import torch
from rich.progress import track
from torch_geometric.loader import DataLoader as PyGDataLoader

from baselines.nearest_mask.baseline import NearestMaskIndex
from baselines.nearest_mask.io import load_fu_mask
from tracking.data.dataset import LesionDataset
from tracking.data.features import FeatConfig
from tracking.data.meta import V2Paths, parse_meta_csv
from tracking.decode import decode_sinkhorn_hungarian
from tracking.train.match_utils import split_per_graph
from tracking.train.module import MatcherModule


def _zero() -> dict[str, object]:
    return {
        "merge_correct": 0, "merge_total": 0, "split_correct": 0, "split_total": 0,
        "newly_appeared_correct": 0, "newly_appeared_total": 0,
        "disappeared_correct": 0, "disappeared_total": 0,
        "row_correct": 0, "row_total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "patient_row_acc": [], "patient_edge_acc": [], "external_claim_total": 0,
    }


def _acc(ok: int, total: int) -> float | None:
    return None if total == 0 else ok / total


def _finish(c: dict[str, object]) -> dict[str, object]:
    tp, fp, tn, fn = (int(c[k]) for k in ("tp", "fp", "tn", "fn"))
    pred_pos, true_pos = tp + fp, tp + fn
    prec, rec = _acc(tp, pred_pos), _acc(tp, true_pos)
    f1 = None if not prec or not rec else 2 * prec * rec / (prec + rec)
    row_acc = list(c["patient_row_acc"])
    edge_acc = list(c["patient_edge_acc"])
    return {
        "merge_acc": _acc(int(c["merge_correct"]), int(c["merge_total"])),
        "merge_correct": c["merge_correct"], "merge_total": c["merge_total"],
        "split_acc": _acc(int(c["split_correct"]), int(c["split_total"])),
        "split_correct": c["split_correct"], "split_total": c["split_total"],
        "newly_appeared_acc": _acc(int(c["newly_appeared_correct"]), int(c["newly_appeared_total"])),
        "newly_appeared_correct": c["newly_appeared_correct"],
        "newly_appeared_total": c["newly_appeared_total"],
        "disappeared_acc": _acc(int(c["disappeared_correct"]), int(c["disappeared_total"])),
        "disappeared_correct": c["disappeared_correct"], "disappeared_total": c["disappeared_total"],
        "edge_acc_micro": _acc(tp + tn, tp + fp + tn + fn),
        "patient_edge_acc_macro": None if not edge_acc else float(np.mean(edge_acc)),
        "row_acc_micro": _acc(int(c["row_correct"]), int(c["row_total"])),
        "patient_row_acc_macro": None if not row_acc else float(np.mean(row_acc)),
        "positive_edge_recall": rec, "positive_edge_precision": prec, "positive_edge_f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "external_claim_total": c["external_claim_total"],
    }


def _topology(root: Path, g) -> dict[int, str]:
    pid = str(g.pid)
    img = int(torch.as_tensor(g.img_id_fu_used).reshape(-1)[0].item())
    rows = parse_meta_csv(V2Paths(root, pid).meta)
    return {r.lesion_id: r.topology for r in rows if r.img_id_fu == img}


def _add_graph(root: Path, g, dec: np.ndarray, c: dict[str, object]) -> None:
    n_bl, n_fu = len(g["bl"].lesion_id), len(g["fu"].lesion_id)
    lab = g["bl", "cross", "fu"].edge_label.reshape(n_bl, n_fu).cpu().numpy() > 0.5
    bl_ids = g["bl"].lesion_id.cpu().numpy().astype(int)
    topo = _topology(root, g)
    pred = np.zeros((n_bl, n_fu), dtype=bool)
    for i, j in enumerate(dec):
        if int(j) >= 0:
            pred[i, int(j)] = True
        elif int(j) < -1:
            c["external_claim_total"] = int(c["external_claim_total"]) + 1
    tp = int(np.logical_and(pred, lab).sum())
    fp = int(np.logical_and(pred, ~lab).sum())
    fn = int(np.logical_and(~pred, lab).sum())
    tn = int(np.logical_and(~pred, ~lab).sum())
    for k, v in zip(("tp", "fp", "tn", "fn"), (tp, fp, tn, fn)):
        c[k] = int(c[k]) + v
    row_ok = 0
    for i in range(n_bl):
        pos = set(map(int, np.where(lab[i])[0]))
        ok = int(int(dec[i]) in pos) if pos else int(int(dec[i]) == -1)
        row_ok += ok
        t = topo[int(bl_ids[i])]
        if t == "MERGED":
            c["merge_total"] = int(c["merge_total"]) + 1
            c["merge_correct"] = int(c["merge_correct"]) + ok
        if t == "SPLIT":
            c["split_total"] = int(c["split_total"]) + 1
            c["split_correct"] = int(c["split_correct"]) + ok
        if float(g["bl"].no_match_label[i]) > 0.5:
            c["disappeared_total"] = int(c["disappeared_total"]) + 1
            c["disappeared_correct"] = int(c["disappeared_correct"]) + int(int(dec[i]) == -1)
    claimed = {int(j) for j in dec if int(j) >= 0}
    for j in range(n_fu):
        if float(g["fu"].no_match_label[j]) > 0.5:
            c["newly_appeared_total"] = int(c["newly_appeared_total"]) + 1
            c["newly_appeared_correct"] = int(c["newly_appeared_correct"]) + int(j not in claimed)
    c["row_correct"] = int(c["row_correct"]) + row_ok
    c["row_total"] = int(c["row_total"]) + n_bl
    c["patient_row_acc"].append(row_ok / max(n_bl, 1))
    c["patient_edge_acc"].append((tp + tn) / max(tp + fp + tn + fn, 1))


def _eval_device(preference: str) -> torch.device:
    p = preference.lower().strip()
    if p == "cpu":
        return torch.device("cpu")
    if p == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--eval-device cuda but CUDA not available")
        return torch.device("cuda")
    if p == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--eval-device mps but MPS not available")
        return torch.device("mps")
    if p == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"unknown device preference: {preference!r}")


def eval_gnn(
    ckpt: Path,
    root: Path,
    cache: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    tta_n: int,
    use_ema: bool,
    *,
    eval_device: str = "auto",
    show_progress: bool = True,
    cuda_gc_each_batch: bool = True,
    feat: FeatConfig | None = None,
) -> dict[str, object]:
    dev = _eval_device(eval_device)
    fc = feat or FeatConfig()
    ds = LesionDataset(root=cache, split=split, dataset_root=root, feat=fc)
    dl = PyGDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    mod = MatcherModule.load_from_checkpoint(str(ckpt), map_location=dev).to(dev).eval()
    c = _zero()
    it = dl
    if show_progress:
        it = track(dl, description=f"GNN {split}", total=len(dl))
    with torch.no_grad():
        for batch in it:
            batch = batch.to(dev)
            out = mod.predict_batch(batch, tta_n=tta_n, use_ema=use_ema)
            graphs, pp, db, df = split_per_graph(batch, out)
            for g, p, b, f in zip(graphs, pp, db, df):
                n_bl, n_fu = len(g["bl"].lesion_id), len(g["fu"].lesion_id)
                dec = decode_sinkhorn_hungarian(p.detach(), b.detach(), f.detach(), n_bl, n_fu, iters=mod.hparams.sinkhorn_iters, tau=float(mod.hparams.dust_tau))
                _add_graph(root, g.cpu(), dec, c)
            del batch, out, graphs, pp, db, df
            if cuda_gc_each_batch and dev.type == "cuda":
                torch.cuda.empty_cache()
    del mod
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return _finish(c)


def eval_baseline(root: Path, cache: Path, split: str, *, feat: FeatConfig | None = None, show_progress: bool = True, one_mask_cache: bool = True) -> dict[str, object]:
    fc = feat or FeatConfig()
    ds = LesionDataset(root=cache, split=split, dataset_root=root, feat=fc)
    idx_cache: dict[tuple[str, int], NearestMaskIndex] = {}
    c = _zero()
    graphs = ds
    if show_progress:
        graphs = track(ds, description=f"Baseline {split}", total=len(ds))
    for g in graphs:
        pid = str(g.pid)
        img = int(torch.as_tensor(g.img_id_fu_used).reshape(-1)[0].item())
        key = (pid, img)
        if key not in idx_cache:
            if one_mask_cache and idx_cache:
                idx_cache.clear()
            mask, sp = load_fu_mask(V2Paths(root, pid).fu_mask(img))
            idx_cache[key] = NearestMaskIndex(mask, sp)
        rows = {r.lesion_id: r for r in parse_meta_csv(V2Paths(root, pid).meta) if r.img_id_fu == img}
        fu = {int(x): i for i, x in enumerate(g["fu"].lesion_id.cpu().numpy())}
        dec = np.full(len(g["bl"].lesion_id), -1, dtype=np.int64)
        for i, lid in enumerate(g["bl"].lesion_id.cpu().numpy().astype(int)):
            r = rows[int(lid)]
            assert r.cog_propagated is not None
            pred = idx_cache[key].query(r.cog_propagated).pred_fu_lesion_id
            dec[i] = fu[pred] if pred in fu else (-1 if pred < 0 else -2)
        _add_graph(root, g, dec, c)
    return _finish(c)
