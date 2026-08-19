"""CSV-free tracking: CT + instance masks + propagated centroids → pair logits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from tracking.common import eval_device
from tracking.data.graph import GraphConfig, _load_vol
from tracking.data.masks import build_mask_graph
from tracking.decode import DECODE_CHOICES, decode_pairs
from tracking.train.module import MatcherModule


@dataclass
class TrackResult:
    bl_ids: np.ndarray
    fu_ids: np.ndarray
    pair: np.ndarray
    pair_prob: np.ndarray
    dust_bl: np.ndarray
    dust_fu: np.ndarray
    pairs: np.ndarray
    decode: str


def load_matcher(ckpt: Path, device: str) -> MatcherModule:
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}.\n"
            f"Expected a Lightning .ckpt from lesion_track_train.\n"
            f"Fix: --ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt"
        )
    dev = eval_device(device)
    mod = MatcherModule.load_from_checkpoint(str(ckpt), map_location=dev)
    return mod.to(dev).eval()


def mask_has_lesions(path: Path) -> bool:
    vol, _, _ = _load_vol(path)
    return bool(np.any(vol.astype(np.int64) != 0))


def write_match_csv(path: Path, r: TrackResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bl_lesion_id", "fu_lesion_id", "pair_prob", "decode"])
        for i, j in r.pairs:
            w.writerow([int(r.bl_ids[i]), int(r.fu_ids[j]), float(r.pair_prob[i, j]), r.decode])


def track(
    bl_img: Path,
    bl_mask: Path,
    fu_img: Path,
    fu_mask: Path,
    propagated: Path,
    ckpt: Path,
    *,
    decode: str,
    device: str = "cuda",
    default_lesion_type: str | None = "unclear",
    k_intra: int = 8,
    thresh: float = 0.5,
    sinkhorn_iters: int = 20,
    sinkhorn_tau: float = 0.2,
    use_ema: bool = True,
    matcher: MatcherModule | None = None,
) -> TrackResult:
    assert decode in DECODE_CHOICES
    for label, p in (("bl-img", bl_img), ("bl-mask", bl_mask), ("fu-img", fu_img), ("fu-mask", fu_mask), ("propagated", propagated)):
        if not Path(p).is_file():
            raise FileNotFoundError(
                f"No {label} at {p}.\n"
                f"Expected NIfTI, meta CSV, slim CSV, or FU-frame JSON on disk.\n"
                f"Fix: pass an existing --{label} path"
            )
    mod = matcher if matcher is not None else load_matcher(ckpt, device)
    dev = next(mod.parameters()).device
    data = build_mask_graph(
        Path(bl_img), Path(bl_mask), Path(fu_img), Path(fu_mask),
        Path(propagated), GraphConfig(k_intra=k_intra), default_lesion_type,
    )
    n_bl, n_fu = int(data["bl"].num_nodes), int(data["fu"].num_nodes)
    bat = Batch.from_data_list([data.to(dev)])
    with torch.no_grad():
        outp = mod.predict_batch(bat, use_ema=use_ema)
        pair = outp.pair.reshape(n_bl, n_fu)
        dust_bl, dust_fu = outp.dust_bl, outp.dust_fu
        pairs = decode_pairs(
            decode, outp.pair, dust_bl, dust_fu, n_bl, n_fu,
            thresh=thresh, sinkhorn_iters=sinkhorn_iters, sinkhorn_tau=sinkhorn_tau,
        )
    return TrackResult(
        bl_ids=data["bl"].lesion_id.cpu().numpy(),
        fu_ids=data["fu"].lesion_id.cpu().numpy(),
        pair=pair.cpu().numpy(),
        pair_prob=torch.sigmoid(pair).cpu().numpy(),
        dust_bl=dust_bl.cpu().numpy(),
        dust_fu=dust_fu.cpu().numpy(),
        pairs=pairs,
        decode=decode,
    )
