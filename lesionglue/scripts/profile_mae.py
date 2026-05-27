"""Micro-benchmark MAE preprocess bottlenecks on 1-2 patients."""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

from tracking.common import DATASET_ROOT
from tracking.data.features import DEFAULT_MAE_CKPT, DEFAULT_MAE_PLANS, FeatConfig
from tracking.data.graph import GraphConfig, build_hetero_data
from tracking.data.mae import MaeExtractor
from tracking.data.meta import load_split_json


def profile_patient(pid: str, mae: MaeExtractor) -> float:
    t0 = time.perf_counter()
    g = build_hetero_data(pid, DATASET_ROOT, GraphConfig(feat=mae.cfg), mae=mae)
    dt = time.perf_counter() - t0
    if g is None:
        print(f"{pid}: skipped")
    else:
        print(f"{pid}: bl={g['bl'].num_nodes} fu={g['fu'].num_nodes} time={dt:.2f}s")
    return dt


def main() -> None:
    pids = load_split_json(DATASET_ROOT / "data_split.json")["train"][:2]
    feat = FeatConfig(mode="mae", mae_ckpt=DEFAULT_MAE_CKPT, mae_plans=DEFAULT_MAE_PLANS)
    mae = MaeExtractor(feat)

    pr = cProfile.Profile()
    pr.enable()
    total = 0.0
    for pid in pids:
        total += profile_patient(pid, mae)
    pr.disable()

    print(f"\ntotal={total:.2f}s for {len(pids)} patients")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(40)
    print(s.getvalue())


if __name__ == "__main__":
    main()
