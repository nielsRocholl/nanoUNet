"""L0 node feature layout: pack_node, cache tag v7_native."""

from __future__ import annotations

import numpy as np

from lesionglue.data.appearance import MaskFeats

DESC_DIM = 1372
STAT_DIM = 14
FEAT_DIM = DESC_DIM + STAT_DIM + 1
CACHE_TAG = "v7_native"


def feat_layout() -> tuple[int, int, int, int, int]:
    return 0, DESC_DIM, DESC_DIM, DESC_DIM + STAT_DIM, DESC_DIM + STAT_DIM


def assert_graph_feat(g) -> None:
    from lesionglue.data.pairs import CROSS_DIM

    n_bl = g["bl"].num_nodes
    assert getattr(g, "feat_mode", "l0") == "l0"
    assert g["bl"].x.shape[1] == FEAT_DIM
    assert g["fu"].x.shape[1] == FEAT_DIM
    assert g["bl", "cross", "fu"].edge_attr.shape[1] == CROSS_DIM
    assert g["bl"].pos_native.shape == (n_bl, 3)
    assert g["bl"].img_bl.shape == (n_bl,)
    assert g["bl"].sp_bl.shape == (n_bl, 3)


def pack_node(desc: np.ndarray, mf: MaskFeats, lt_i: int) -> np.ndarray:
    assert desc.shape == (DESC_DIM,)
    x = np.zeros(FEAT_DIM, np.float32)
    x[:DESC_DIM] = np.clip(desc, -1000.0, 1000.0) / 1000.0
    o = DESC_DIM
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
