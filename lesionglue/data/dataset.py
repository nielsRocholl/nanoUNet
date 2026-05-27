"""Cached v5 per-split PyG InMemoryDataset from dense Longitudinal_CT_v2 graphs.

Workers load NIfTIs; optional ProcessPoolExecutor (--jobs > 1). MAE mode requires jobs=1
(intra-patient resample/IO uses threads; do not fork CUDA across patients).
Per-patient staging under processed/staging/; --resume skips staged patients.
Train split can apply propagation-noise jitter in __getitem__ when augment=True."""

from __future__ import annotations

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
from tracking.data.features import FeatConfig, assert_graph_feat, cache_tag, desc_dim, feat_dim
from tracking.data.graph import GraphConfig, build_hetero_data
from tracking.data.meta import load_split_json


def _build_one(pid: str, root_s: str, k_intra: int, feat: FeatConfig) -> tuple[str, HeteroData | None]:
    cfg = GraphConfig(k_intra=k_intra, feat=feat)
    g = build_hetero_data(pid, Path(root_s), cfg, mae=None)
    return pid, g


class LesionDataset(InMemoryDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        dataset_root: Path | str = DATASET_ROOT,
        cfg: GraphConfig | None = None,
        feat: FeatConfig | None = None,
        num_workers: int = 1,
        resume: bool = False,
        augment: bool = False,
        fu_jitter_scale: float = 0.3,
        p_drop_fu: float = 0.1,
        p_drop_bl: float = 0.1,
        desc_jitter_frac: float = 0.0,
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.feat = feat or FeatConfig()
        self.cfg = cfg or GraphConfig(feat=self.feat)
        self.cfg.feat = self.feat
        self.num_workers = max(1, num_workers)
        self.resume = resume
        if self.feat.mode == "mae":
            assert self.num_workers == 1, "MAE preprocess requires --jobs 1"
        self.augment = augment
        self.fu_jitter_scale = fu_jitter_scale
        self.p_drop_fu = p_drop_fu
        self.p_drop_bl = p_drop_bl
        self.desc_jitter_frac = desc_jitter_frac
        super().__init__(root)
        self.load(self.processed_paths[0])
        if len(self):
            assert_graph_feat(super().get(0), self.feat)

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.split}_{cache_tag(self.feat)}.pt"]

    def get(self, idx: int) -> HeteroData:
        d = super().get(idx).clone()
        if self.augment:
            if self.p_drop_fu > 0.0 or self.p_drop_bl > 0.0:
                drop_nodes(
                    d,
                    self.p_drop_fu,
                    self.p_drop_bl,
                    self.cfg.k_intra,
                    self.feat,
                    rng=np.random.default_rng(),
                )
            jitter_both(
                d,
                k_intra=self.cfg.k_intra,
                sigma_fu_scale=self.fu_jitter_scale,
                rng=np.random.default_rng(),
                desc_jitter_frac=self.desc_jitter_frac,
                feat=self.feat,
            )
        return d

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
        feat = self.feat
        staging = stg.dir(Path(self.processed_dir), self.split, feat)
        staging.mkdir(parents=True, exist_ok=True)
        if not self.resume:
            stg.clear(staging)

        todo = stg.todo(pids, staging, self.resume)
        if self.resume and len(todo) < len(pids):
            print0(f"resume {self.split}: skip {len(pids) - len(todo)}/{len(pids)} staged, build {len(todo)}")

        mae = None
        if feat.mode == "mae":
            from tracking.data.mae import MaeExtractor

            mae = MaeExtractor(feat)
        cols = (
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
            TextColumn("[cyan]{task.fields[patient]}[/cyan]"),
        )

        if self.num_workers == 1:
            with Progress(*cols) as prog:
                task = prog.add_task(self.split, total=len(todo), patient="")
                for pid in todo:
                    prog.update(task, patient=str(pid))
                    g = build_hetero_data(pid, self.dataset_root, self.cfg, mae=mae)
                    if g is None:
                        prog.advance(task)
                        continue
                    stg.save(staging, pid, g)
                    self._log_graph(pid, g, prog.console)
                    prog.advance(task)
        else:
            with Progress(*cols) as prog:
                task = prog.add_task(self.split, total=len(todo), patient="")
                with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                    futs = [ex.submit(_build_one, pid, root_s, k_intra, feat) for pid in todo]
                    for fut in as_completed(futs):
                        pid, g = fut.result()
                        prog.update(task, patient=str(pid))
                        prog.advance(task)
                        if g is None:
                            continue
                        stg.save(staging, pid, g)
                        self._log_graph(pid, g, prog.console)

        graphs = stg.load_all(staging, pids)
        if not graphs:
            raise RuntimeError(f"zero graphs for split={self.split}")
        tp = sum(int(g["bl", "cross", "fu"].edge_label.sum()) for g in graphs)
        te = sum(g["bl", "cross", "fu"].num_edges for g in graphs)
        tag = cache_tag(feat)
        meta = {"edges": te, "positives": tp, "feat_mode": feat.mode, "desc_dim": desc_dim(feat), "feat_dim": feat_dim(feat)}
        torch.save(meta, Path(self.processed_dir) / f"{self.split}_{tag}_meta.pt")
        self.save(graphs, self.processed_paths[0])
        stg.rm(staging)
