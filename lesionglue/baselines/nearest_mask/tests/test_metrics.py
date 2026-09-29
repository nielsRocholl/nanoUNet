from __future__ import annotations

import unittest

from baselines.nearest_mask.baseline import PredictionRow
from baselines.nearest_mask.metrics import summarize
from tracking.data.meta import LesionRow


def row(lid: int, topo: str, img_id_fu: int = 0, merged_into: int | None = None) -> LesionRow:
    return LesionRow(
        lesion_id=lid,
        topology=topo,
        cog_bl=(0.0, 0.0, 0.0),
        cog_propagated=(0.0, 0.0, 0.0),
        cog_fu=(0.0, 0.0, 0.0),
        img_id_bl=0,
        img_id_fu=img_id_fu,
        lesion_type="Liver",
        merged_into=merged_into,
    )


class MetricsTests(unittest.TestCase):
    def test_topology_metrics(self) -> None:
        preds = [
            PredictionRow("p", 0, 1, "UNCHANGED", 1, 1, 0.0, True, "inside_mask"),
            PredictionRow("p", 0, 2, "SPLIT", 2, 3, 2.0, False, "nearest_mask"),
            PredictionRow("p", 0, 3, "MERGED", 10, 10, 1.0, True, "nearest_mask"),
            PredictionRow("p", 0, 4, "DISAPPEARED", -1, 8, 4.0, False, "nearest_mask"),
        ]
        rows_by_pid = {
            "p": [
                row(1, "UNCHANGED"),
                row(2, "SPLIT"),
                row(3, "MERGED", merged_into=10),
                row(4, "DISAPPEARED"),
                row(8, "NEWLYAPPEARING"),
                row(9, "NEWLYAPPEARING"),
            ]
        }

        summary = summarize("val", preds, rows_by_pid)

        self.assertEqual(summary["linkable_total"], 3)
        self.assertAlmostEqual(summary["linkable_acc"], 2 / 3)
        self.assertEqual(summary["disappeared_total"], 1)
        self.assertEqual(summary["disappeared_acc"], 0.0)
        self.assertEqual(summary["newly_appearing_total"], 2)
        self.assertEqual(summary["newly_appearing_correct"], 1)
        self.assertEqual(summary["newly_appearing_acc"], 0.5)
        self.assertAlmostEqual(summary["row_acc_all_bl"], 0.5)


if __name__ == "__main__":
    unittest.main()
