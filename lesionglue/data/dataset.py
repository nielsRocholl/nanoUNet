"""Cached v7_native per-split PyG InMemoryDataset from dense Longitudinal_CT_v2 graphs.

Workers pin BLAS/OpenMP to 1 thread: 16 jobs × default torch threads starves a
cgroup and turns CIFS NIfTI reads into ~1 patient/min."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from torch_geometric.data import HeteroData, InMemoryDataset

from tracking.common import DATASET_ROOT, print0
from tracking.data import staging as stg
from tracking.data.augment import drop_nodes, jitter_both
from tracking.data.features import CACHE_TAG, DESC_DIM, FEAT_DIM, assert_graph_feat
from tracking.data.graph import GraphConfig, build_hetero_data
from tracking.data.intra import refresh_edges
from tracking.data.splits import load_tracking_split


def _limit_threads() -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    torch.set_num_threads(1)


def _build_one(pid: str, root_s: str, cfg: GraphConfig) -> tuple[str, list]:
    _limit_threads()
    return pid, build_hetero_data(pid, Path(root_s), cfg)


class LesionDataset(InMemoryDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        dataset_root: Path | str = DATASET_ROOT,
        cfg: GraphConfig | None = None,
        num_workers: int = 1,
        resume: bool = False,
        augment: bool = False,
        fu_jitter_scale: float = 0.3,
        p_drop_fu: float = 0.1,
        p_drop_bl: float = 0.1,
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.cfg = cfg or GraphConfig()
        self.num_workers = max(1, num_workers)
        self.resume = resume
        self.augment = augment
        self.fu_jitter_scale = fu_jitter_scale
        self.p_drop_fu = p_drop_fu
        self.p_drop_bl = p_drop_bl
        super().__init__(root)
        self.load(self.processed_paths[0])
        if len(self):
            assert_graph_feat(super().get(0))

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.split}_{CACHE_TAG}.pt"]

    def get(self, idx: int) -> HeteroData:
        d = super().get(idx).clone()
        if self.augment:
            if self.p_drop_fu > 0.0 or self.p_drop_bl > 0.0:
                drop_nodes(d, self.p_drop_fu, self.p_drop_bl, self.cfg, rng=np.random.default_rng())
            if not self.cfg.drop_dp:
                jitter_both(d, self.cfg, sigma_fu_scale=self.fu_jitter_scale, rng=np.random.default_rng())
        return refresh_edges(d, self.cfg)

    def _log_graph(self, pid: str, g: HeteroData, console: Console) -> None:
        el = g["bl", "cross", "fu"].edge_label
        console.print(
            f"[{self.split}] {pid} bl={g['bl'].num_nodes} fu={g['fu'].num_nodes} "
            f"cross={g['bl','cross','fu'].num_edges} pos={int(el.sum())}"
        )

    def process(self) -> None:
        sp = load_tracking_split()
        if self.split not in sp:
            raise KeyError(
                f"split {self.split!r} missing from tracking split.\n"
                f"Expected keys train/val/test in {sp.keys() if isinstance(sp, dict) else 'configs/split.json'}.\n"
                f"Fix: python3 tracking/cli/split.py --root /nnunet_data/Longitudinal-CT"
            )
        pids = list(sp[self.split])
        root_s = str(self.dataset_root)
        cfg = self.cfg
        staging = stg.dir(Path(self.processed_dir), self.split)
        staging.mkdir(parents=True, exist_ok=True)
        if not self.resume:
            stg.clear(staging)

        todo = stg.todo(pids, staging, self.resume)
        if self.resume and len(todo) < len(pids):
            print0(f"resume {self.split}: skip {len(pids) - len(todo)}/{len(pids)} staged, build {len(todo)}")

        cols = (
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
            TextColumn("[cyan]{task.fields[patient]}[/cyan]"),
        )

        _limit_threads()
        if self.num_workers == 1:
            with Progress(*cols) as prog:
                task = prog.add_task(self.split, total=len(todo), patient="")
                for pid in todo:
                    prog.update(task, patient=str(pid))
                    graphs = build_hetero_data(pid, self.dataset_root, cfg)
                    stg.save(staging, pid, graphs)
                    for g in graphs:
                        self._log_graph(pid, g, prog.console)
                    prog.advance(task)
        else:
            with Progress(*cols) as prog:
                task = prog.add_task(self.split, total=len(todo), patient="")
                with ProcessPoolExecutor(max_workers=self.num_workers, initializer=_limit_threads) as ex:
                    futs = [ex.submit(_build_one, pid, root_s, cfg) for pid in todo]
                    for fut in as_completed(futs):
                        pid, graphs = fut.result()
                        prog.update(task, patient=str(pid))
                        prog.advance(task)
                        stg.save(staging, pid, graphs)
                        for g in graphs:
                            self._log_graph(pid, g, prog.console)

        graphs = stg.load_all(staging, pids)
        if not graphs:
            raise RuntimeError(f"zero graphs for split={self.split}")
        tp = sum(int(g["bl", "cross", "fu"].edge_label.sum()) for g in graphs)
        te = sum(g["bl", "cross", "fu"].num_edges for g in graphs)
        meta = {"edges": te, "positives": tp, "feat_mode": "l0", "desc_dim": DESC_DIM, "feat_dim": FEAT_DIM}
        torch.save(meta, Path(self.processed_dir) / f"{self.split}_{CACHE_TAG}_meta.pt")
        self.save(graphs, self.processed_paths[0])
        stg.rm(staging)
