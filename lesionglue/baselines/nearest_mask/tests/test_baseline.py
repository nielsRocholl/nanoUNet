from __future__ import annotations

import unittest

import numpy as np

from baselines.nearest_mask.baseline import NearestMaskIndex


class NearestMaskIndexTests(unittest.TestCase):
    def test_point_inside_mask_returns_zero_distance(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=np.int64)
        mask[2, 2, 2] = 7
        index = NearestMaskIndex(mask, np.array([1.0, 1.0, 1.0]))

        match = index.query(np.array([2.2, 2.8, 2.4]))

        self.assertEqual(match.pred_fu_lesion_id, 7)
        self.assertEqual(match.distance_mm, 0.0)
        self.assertEqual(match.status, "inside_mask")

    def test_anisotropic_spacing_uses_physical_distance(self) -> None:
        mask = np.zeros((10, 10, 3), dtype=np.int64)
        mask[0, 8, 0] = 1
        mask[1, 0, 0] = 2
        index = NearestMaskIndex(mask, np.array([10.0, 1.0, 1.0]))

        match = index.query(np.array([0.5, 0.5, 0.5]))

        self.assertEqual(match.pred_fu_lesion_id, 1)
        self.assertEqual(match.status, "nearest_mask")

    def test_empty_mask_returns_no_candidate(self) -> None:
        mask = np.zeros((4, 4, 4), dtype=np.int64)
        index = NearestMaskIndex(mask, np.array([1.0, 1.0, 1.0]))

        match = index.query(np.array([2.0, 2.0, 2.0]))

        self.assertEqual(match.pred_fu_lesion_id, -1)
        self.assertIsNone(match.distance_mm)
        self.assertEqual(match.status, "no_fu_candidate")

    def test_multiple_queries_can_claim_same_label(self) -> None:
        mask = np.zeros((6, 6, 6), dtype=np.int64)
        mask[2, 2, 2] = 4
        index = NearestMaskIndex(mask, np.array([1.0, 1.0, 1.0]))

        first = index.query(np.array([1.0, 2.0, 2.0]))
        second = index.query(np.array([3.0, 2.0, 2.0]))

        self.assertEqual(first.pred_fu_lesion_id, 4)
        self.assertEqual(second.pred_fu_lesion_id, 4)


if __name__ == "__main__":
    unittest.main()
