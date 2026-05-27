"""Node feature layout: l0 / mae / yerebakan modes, pack_node, cache tag v5_{mode}."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.data.appearance import MaskFeats

L0_DIM = 1372
YEREBAKAN_LEVELS = 5
YEREBAKAN_DIM = L0_DIM * YEREBAKAN_LEVELS
MAE_DIM = 320
STAT_DIM = 14
FEAT_MODES = ("l0", "mae", "yerebakan")
DEFAULT_MAE_CKPT = (
    "/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/"
    "mae_pretrain/checkpoints/last.ckpt"
)
DEFAULT_MAE_PLANS = (
    "/nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/nnUNetResEncUNetLPlans_h200_smallpv.json"
)


@dataclass
class FeatConfig:
    mode: str = "l0"
    mae_ckpt: str = ""
    mae_plans: str = ""
    mae_skip: int = 4
    mae_batch: int = 2


def desc_dim(cfg: FeatConfig) -> int:
    if cfg.mode == "mae":
        return MAE_DIM
    if cfg.mode == "yerebakan":
        return YEREBAKAN_DIM
    return L0_DIM


def feat_dim(cfg: FeatConfig) -> int:
    return desc_dim(cfg) + STAT_DIM + 1


def feat_layout(cfg: FeatConfig) -> tuple[int, int, int, int, int]:
    d = desc_dim(cfg)
    return 0, d, d, d + STAT_DIM, d + STAT_DIM


def cache_tag(cfg: FeatConfig) -> str:
    assert cfg.mode in FEAT_MODES
    return f"v5_{cfg.mode}"


def assert_graph_feat(g, cfg: FeatConfig) -> None:
    from tracking.data.pairs import CROSS_DIM

    gm = getattr(g, "feat_mode", "l0")
    assert gm == cfg.mode, f"graph feat_mode {gm!r} != {cfg.mode!r}"
    fd = feat_dim(cfg)
    assert g["bl"].x.shape[1] == fd, f"bl.x dim {g['bl'].x.shape[1]} != {fd}"
    assert g["fu"].x.shape[1] == fd
    assert g["bl", "cross", "fu"].edge_attr.shape[1] == CROSS_DIM


def pack_node(desc: np.ndarray, mf: MaskFeats, lt_i: int, cfg: FeatConfig) -> np.ndarray:
    d = desc_dim(cfg)
    assert desc.shape == (d,)
    x = np.zeros(feat_dim(cfg), np.float32)
    if cfg.mode == "mae":
        x[:d] = desc.astype(np.float32)
    else:
        x[:d] = np.clip(desc, -1000.0, 1000.0) / 1000.0
    o = d
    x[o] = mf.log_volume / 10.0
    x[o + 1] = np.clip(mf.mean_hu, -1000.0, 1000.0) / 1000.0
    x[o + 2] = np.clip(mf.sphericity, 0.0, 2.0)
    x[o + 3] = mf.hu_std / 500.0
    x[o + 4] = np.clip(mf.hu_p10, -1000.0, 1000.0) / 1000.0
    x[o + 5] = np.clip(mf.hu_p50, -1000.0, 1000.0) / 1000.0
    x[o + 6] = np.clip(mf.hu_p90, -1000.0, 1000.0) / 1000.0
    x[o + 7] = np.clip(mf.hu_min, -1000.0, 1000.0) / 1000.0
    x[o + 8] = np.clip(mf.hu_max, -1000.0, 1000.0) / 1000.0
    x[o + 9] = mf.bbox_e0_mm / 100.0
    x[o + 10] = mf.bbox_e1_mm / 100.0
    x[o + 11] = mf.bbox_e2_mm / 100.0
    x[o + 12] = mf.pca_l1_mm / 100.0
    x[o + 13] = mf.pca_l2_mm / 100.0
    x[o + 14] = float(lt_i)
    return x


def add_feat_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--feat", choices=FEAT_MODES, default="l0")
    ap.add_argument("--mae-ckpt", default=DEFAULT_MAE_CKPT)
    ap.add_argument("--mae-plans", default=DEFAULT_MAE_PLANS)
    ap.add_argument("--mae-skip", type=int, default=4)
    ap.add_argument("--mae-batch", type=int, default=2, help="MAE encoder ROI batch size (use 2 on 11GB GPUs)")


def feat_from_args(args) -> FeatConfig:
    mode = args.feat
    fc = FeatConfig(
        mode=mode,
        mae_ckpt=args.mae_ckpt,
        mae_plans=args.mae_plans,
        mae_skip=args.mae_skip,
        mae_batch=max(1, args.mae_batch),
    )
    if mode == "mae":
        assert Path(fc.mae_ckpt).is_file(), f"missing MAE ckpt: {fc.mae_ckpt}"
        assert Path(fc.mae_plans).is_file(), f"missing MAE plans: {fc.mae_plans}"
    return fc
