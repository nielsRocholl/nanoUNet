"""Cached v2 per-split PyG InMemoryDataset built from dense Longitudinal_CT_v2 graphs.

Process builds graphs per patient behind a Rich progress bar. Optional ProcessPoolExecutor
(--jobs > 1) for CPU parallelism; each worker loads full NIfTIs — watch RAM on large --jobs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from torch_geometric.data import HeteroData, InMemoryDataset

from tracking.common import DATASET_ROOT
from tracking.data.graph import GraphConfig, build_hetero_data
from tracking.data.meta import load_split_json


def _build_one(pid: str, root_s: str, k_intra: int) -> tuple[str, HeteroData | None]:
    cfg = GraphConfig(k_intra=k_intra)
    g = build_hetero_data(pid, Path(root_s), cfg)
    return pid, g


class LesionDataset(InMemoryDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        dataset_root: Path | str = DATASET_ROOT,
        cfg: GraphConfig | None = None,
        num_workers: int = 1,
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.cfg = cfg or GraphConfig()
        self.num_workers = max(1, num_workers)
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.split}_v2.pt"]

    def _log_graph(self, pid: str, g: HeteroData, console: Console) -> None:
        el = g["bl", "cross", "fu"].edge_label
        console.print(
            f"[{self.split}] {pid} bl={g['bl'].num_nodes} fu={g['fu'].num_nodes} "
            f"cross={g['bl','cross','fu'].num_edges} pos={int(el.sum())}"
        )

    def process(self) -> None:
        sp = load_split_json(self.dataset_root / "data_split.json")
        if self.split not in sp:
            raise KeyError(self.split)
        pids = list(sp[self.split])
        root_s = str(self.dataset_root)
        k_intra = self.cfg.k_intra
        cols = (
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
            TextColumn("[cyan]{task.fields[patient]}[/cyan]"),
        )

        if self.num_workers == 1:
            graphs = []
            with Progress(*cols) as prog:
                task = prog.add_task(self.split, total=len(pids), patient="")
                for pid in pids:
                    prog.update(task, patient=str(pid))
                    g = build_hetero_data(pid, self.dataset_root, self.cfg)
                    if g is None:
                        prog.advance(task)
                        continue
                    graphs.append(g)
                    self._log_graph(pid, g, prog.console)
                    prog.advance(task)
        else:
            by_pid: dict[str, HeteroData] = {}
            with Progress(*cols) as prog:
                task = prog.add_task(self.split, total=len(pids), patient="")
                with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                    futs = [ex.submit(_build_one, pid, root_s, k_intra) for pid in pids]
                    for fut in as_completed(futs):
                        pid, g = fut.result()
                        prog.update(task, patient=str(pid))
                        prog.advance(task)
                        if g is None:
                            continue
                        by_pid[pid] = g
                        self._log_graph(pid, g, prog.console)
            graphs = [by_pid[p] for p in pids if p in by_pid]

        if not graphs:
            raise RuntimeError(f"zero graphs for split={self.split}")
        tp = sum(int(g["bl", "cross", "fu"].edge_label.sum()) for g in graphs)
        te = sum(g["bl", "cross", "fu"].num_edges for g in graphs)
        pw = max(1.0, min(100.0, ((te - tp) / max(tp, 1))))
        torch.save({"pos_weight": pw, "edges": te, "positives": tp}, Path(self.processed_dir) / f"{self.split}_v2_meta.pt")
        self.save(graphs, self.processed_paths[0])
