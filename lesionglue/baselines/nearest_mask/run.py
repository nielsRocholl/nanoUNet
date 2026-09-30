"""CLI for the isolated nearest-mask distance baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lesionglue.baselines.nearest_mask.baseline import PatientBaselineResult, run_patient
from lesionglue.baselines.nearest_mask.io import (
    load_split_ids,
    write_prediction_rows,
    write_summary_csv,
    write_summary_json,
)
from lesionglue.baselines.nearest_mask.metrics import summarize
from lesionglue.common import cprint


def run_split(root: Path, split: str, pids: list[str], out_dir: Path, graph_compatible: bool) -> dict[str, object]:
    results: list[PatientBaselineResult] = []
    for idx, pid in enumerate(pids, start=1):
        cprint(f"[{split}] {idx}/{len(pids)} {pid}", markup=False)
        results.append(run_patient(root, pid, graph_compatible=graph_compatible))

    predictions = [pred for result in results for pred in result.predictions]
    rows_by_pid = {result.pid: result.rows for result in results}
    skipped_missing_cog = sum(result.skipped_missing_cog for result in results)
    summary = summarize(split, predictions, rows_by_pid, skipped_missing_cog=skipped_missing_cog)
    write_prediction_rows(out_dir / "rows.csv", predictions)
    write_summary_json(out_dir / "summary.json", summary)
    write_summary_csv(out_dir / "summary.csv", [summary])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest follow-up mask baseline for Longitudinal_CT_v2.")
    parser.add_argument("--root", required=True, help="Longitudinal_CT_v2 root")
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"), help="split from data_split.json to run; 'all' runs train, val and test into per-split subdirs of --out")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split patient limit for smoke tests")
    parser.add_argument(
        "--graph-compatible",
        action="store_true",
        help="Filter each patient to the dominant img_id_fu, matching the current graph preprocessing.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    split_map = load_split_ids(root, args.split)
    summaries = []
    for split, pids in split_map.items():
        use_pids = pids[: args.limit] if args.limit and args.limit > 0 else pids
        split_out = out / split if args.split == "all" else out
        summaries.append(run_split(root, split, use_pids, split_out, args.graph_compatible))

    if args.split == "all":
        write_summary_csv(out / "summary_all.csv", summaries)


if __name__ == "__main__":
    main()
