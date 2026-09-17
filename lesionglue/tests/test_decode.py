from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from tracking import report
from tracking.decode import decode_pairs, decode_sinkhorn, decode_sinkhorn_hungarian

HI, LO = 8.0, -8.0


def merge_case(k: int, n_other: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Rows 0..k-1 all merge into column 0; the remaining rows match 1:1 onto columns 1..n_other."""
    n_bl, n_fu = k + n_other, 1 + n_other
    L = torch.full((n_bl, n_fu), LO)
    L[:k, 0] = HI
    for t in range(n_other):
        L[k + t, 1 + t] = HI
    return L.reshape(-1), torch.zeros(n_bl), torch.zeros(n_fu), n_bl, n_fu


class SinkhornDecodeTests(unittest.TestCase):
    def test_merge_kept_by_threshold_dropped_by_hungarian(self) -> None:
        for k in (2, 3, 5):
            L, db, df, n_bl, n_fu = merge_case(k)
            pairs = {tuple(p) for p in decode_sinkhorn(L, db, df, n_bl, n_fu, tau=0.125).tolist()}
            want = {(i, 0) for i in range(k)} | {(k + t, 1 + t) for t in range(n_fu - 1)}
            self.assertEqual(pairs, want, k)
            hung = decode_sinkhorn_hungarian(L, db, df, n_bl, n_fu, tau=0.125)
            self.assertLessEqual(int((hung[:k] == 0).sum()), 1, k)

    def test_merge_beyond_one_over_tau_is_lost(self) -> None:
        L, db, df, n_bl, n_fu = merge_case(10)
        pairs = decode_sinkhorn(L, db, df, n_bl, n_fu, tau=0.125)
        self.assertFalse((pairs[:, 1] == 0).any())

    def test_split_emits_every_column(self) -> None:
        L = torch.full((3, 4), LO)
        L[0, 0] = L[0, 1] = HI
        L[1, 2] = HI
        pairs = decode_sinkhorn(L.reshape(-1), torch.zeros(3), torch.zeros(4), 3, 4, tau=0.125)
        self.assertEqual(pairs.tolist(), [[0, 0], [0, 1], [1, 2]])

    def test_dustbin_row_emits_nothing(self) -> None:
        L = torch.full((2, 2), LO)
        L[0, 0] = HI
        db = torch.tensor([0.0, HI])
        pairs = decode_sinkhorn(L.reshape(-1), db, torch.zeros(2), 2, 2, tau=0.125)
        self.assertEqual(pairs.tolist(), [[0, 0]])

    def test_decode_pairs_shape(self) -> None:
        L = torch.full((2, 2), LO)
        for m in ("dense", "sinkhorn", "hungarian"):
            out = decode_pairs(m, L.reshape(-1), torch.full((2,), HI), torch.full((2,), HI), 2, 2, thresh=0.5, sinkhorn_iters=20, sinkhorn_tau=0.125)
            self.assertEqual((out.shape, out.dtype), ((0, 2), np.int64), m)


def graph(lab: np.ndarray) -> dict:
    g = {
        "bl": SimpleNamespace(lesion_id=torch.arange(lab.shape[0]), no_match_label=torch.tensor((~lab.any(1)).astype(np.float32))),
        "fu": SimpleNamespace(lesion_id=torch.arange(lab.shape[1]), no_match_label=torch.tensor((~lab.any(0)).astype(np.float32))),
    }
    g["bl", "cross", "fu"] = SimpleNamespace(edge_label=torch.tensor(lab.reshape(-1).astype(np.float32)))
    return g


class ReportScoringTests(unittest.TestCase):
    def score(self, lab: np.ndarray, pred: np.ndarray, topo: dict[int, str], external: np.ndarray | None = None) -> dict:
        c = report._zero()
        with mock.patch.object(report, "_topology", return_value=topo):
            report._add_graph(None, graph(lab), pred, c, external=external)
        return c

    def test_merge_rows_all_correct(self) -> None:
        lab = np.array([[1, 0], [1, 0], [0, 0]], bool)
        c = self.score(lab, lab.copy(), {0: "MERGED", 1: "MERGED", 2: "DISAPPEARED"})
        self.assertEqual((c["merge_correct"], c["merge_total"], c["row_correct"], c["disappeared_correct"]), (2, 2, 3, 1))

    def test_extra_link_is_wrong_row(self) -> None:
        lab = np.array([[1, 0]], bool)
        c = self.score(lab, np.array([[1, 1]], bool), {0: "SPLIT"})
        self.assertEqual((c["split_correct"], c["row_correct"], c["fp"]), (0, 0, 1))

    def test_external_claim_is_not_a_disappearance(self) -> None:
        lab = np.array([[0, 0]], bool)
        c = self.score(lab, np.zeros((1, 2), bool), {0: "DISAPPEARED"}, external=np.array([True]))
        self.assertEqual((c["disappeared_correct"], c["row_correct"], c["external_claim_total"]), (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
