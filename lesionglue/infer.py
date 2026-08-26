"""CSV-free tracking: CT + instance masks + optional propagated centroids → pair logits.

drop_dp checkpoints omit --propagated and use native mask centroids.
"""

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
from tracking.data.paint import fu_track_map
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


def _empty_result(bl_ids: list[int], fu_ids: list[int], decode: str) -> TrackResult:
    n_bl, n_fu = len(bl_ids), len(fu_ids)
    return TrackResult(
        bl_ids=np.asarray(bl_ids, dtype=np.int64),
        fu_ids=np.asarray(fu_ids, dtype=np.int64),
        pair=np.zeros((n_bl, n_fu), dtype=np.float32),
        pair_prob=np.zeros((n_bl, n_fu), dtype=np.float32),
        dust_bl=np.zeros(n_bl, dtype=np.float32),
        dust_fu=np.zeros(n_fu, dtype=np.float32),
        pairs=np.zeros((0, 2), dtype=np.int64),
        decode=decode,
    )


def write_match_csv(path: Path, r: TrackResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pair_ids = [(int(r.bl_ids[i]), int(r.fu_ids[j])) for i, j in r.pairs]
    m = fu_track_map(list(map(int, r.bl_ids)), list(map(int, r.fu_ids)), pair_ids)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bl_lesion_id", "fu_lesion_id", "pair_prob", "decode", "track_id"])
        for i, j in r.pairs:
            fid = int(r.fu_ids[j])
            w.writerow([int(r.bl_ids[i]), fid, float(r.pair_prob[i, j]), r.decode, m[fid]])


def graph_cfg_from_ckpt(mod: MatcherModule, k_intra: int) -> GraphConfig:
    hp = mod.hparams
    ckpt_k = int(getattr(hp, "k_intra", k_intra))
    if hasattr(hp, "k_intra") and ckpt_k != k_intra:
        raise SystemExit(
            f"--k-intra {k_intra} does not match checkpoint k_intra={ckpt_k}.\n"
            f"Expected the train-time GraphConfig stored on the Lightning ckpt.\n"
            f"Fix: omit --k-intra (ckpt wins) or pass --k-intra {ckpt_k}"
        )
    return GraphConfig(
        k_intra=ckpt_k,
        drop_dp=bool(getattr(hp, "drop_dp", False)),
        intra=str(getattr(hp, "intra", "knn")),
        type_mask=bool(getattr(hp, "type_mask", False)),
    )


def track(
    bl_img: Path,
    bl_mask: Path,
    fu_img: Path,
    fu_mask: Path,
    propagated: Path | None,
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
    types_csv: Path | None = None,
    volumes: tuple | None = None,
    img_id: int | None = None,
) -> TrackResult:
    assert decode in DECODE_CHOICES
    mod = matcher if matcher is not None else load_matcher(ckpt, device)
    gcfg = graph_cfg_from_ckpt(mod, k_intra)
    if gcfg.type_mask and types_csv is None and default_lesion_type == "unclear":
        raise SystemExit(
            "type_mask checkpoint needs lesion types, got default_lesion_type='unclear' and no types CSV.\n"
            "Expected --meta / --types-csv or a real --default-lesion-type.\n"
            "Fix: --meta /nnunet_data/Longitudinal-CT/meta/<pid>.csv"
        )
    if gcfg.drop_dp:
        if propagated is not None:
            raise SystemExit(
                f"drop_dp checkpoint does not use propagated coords, got {propagated!s}.\n"
                f"Expected CT + instance masks only.\n"
                f"Fix: omit --propagated"
            )
    scans = (("bl-img", bl_img), ("bl-mask", bl_mask), ("fu-img", fu_img), ("fu-mask", fu_mask))
    if volumes is None:
        need = scans if gcfg.drop_dp else (*scans, ("propagated", propagated))
        for label, p in need:
            if p is None or not Path(p).is_file():
                raise FileNotFoundError(
                    f"No {label} at {p}.\n"
                    f"Expected NIfTI, meta CSV, slim CSV, or FU-frame JSON on disk.\n"
                    f"Fix: pass an existing --{label} path"
                )
        ct_bl, aff_bl, sp_bl = _load_vol(Path(bl_img))
        mk_bl = _load_vol(Path(bl_mask))[0]
        ct_fu, aff_fu, sp_fu = _load_vol(Path(fu_img))
        mk_fu = _load_vol(Path(fu_mask))[0]
    else:
        if not gcfg.drop_dp and (propagated is None or not Path(propagated).is_file()):
            raise FileNotFoundError(
                f"No propagated at {propagated}.\n"
                f"Expected NIfTI, meta CSV, slim CSV, or FU-frame JSON on disk.\n"
                f"Fix: pass an existing --propagated path"
            )
        ct_bl, aff_bl, sp_bl, mk_bl, ct_fu, aff_fu, sp_fu, mk_fu = volumes
        assert mk_bl.shape == ct_bl.shape, (mk_bl.shape, ct_bl.shape)
        assert mk_fu.shape == ct_fu.shape, (mk_fu.shape, ct_fu.shape)
    data = build_mask_graph(
        ct_bl, aff_bl, sp_bl, mk_bl, ct_fu, aff_fu, sp_fu, mk_fu,
        None if gcfg.drop_dp else Path(propagated), gcfg, default_lesion_type,
        types_csv=Path(types_csv) if types_csv is not None else None, img_id=img_id,
    )
    if data is None:
        fu_ids = sorted(int(x) for x in np.unique(mk_fu.astype(np.int64)) if int(x) != 0)
        return _empty_result([], fu_ids, decode)
    dev = next(mod.parameters()).device
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
