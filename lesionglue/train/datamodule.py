"""LightningDataModule: cached v7_native LesionDataset + PyG DataLoader.

No-fold fit set is train∪val caches (everyone except test_patients.csv).
No Lightning val: holdout is scored once after training. CV folds still have a val pool.
pool="all" adds the test cache: the fit set and the CV folds then cover all 300 patients (the holdout is part of the pool).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader

from lesionglue.common import CACHE_ROOT, DATASET_ROOT, HOLDOUT_CSV
from lesionglue.data.graph.augment import drop_nodes, jitter_both
from lesionglue.data.cache.dataset import LesionDataset
from lesionglue.data.features.layout import CACHE_TAG
from lesionglue.data.graph.dense import GraphConfig
from lesionglue.data.graph.intra import refresh_edges
from lesionglue.data.source.splits import POOL_SPLITS, POOLS, fold_patient_sets, load_holdout, load_tracking_split, pool_patient_ids


class MatcherDataModule(LightningDataModule):
    def __init__(
        self,
        cache_root: Path | str = CACHE_ROOT,
        dataset_root: Path | str = DATASET_ROOT,
        batch_size: int = 8,
        val_batch_size: int = 1,
        num_workers: int = 2,
        fu_jitter_scale: float = 0.3,
        p_drop_fu: float = 0.1,
        p_drop_bl: float = 0.1,
        graph: GraphConfig | None = None,
        fold: int | None = None,
        n_folds: int = 5,
        cv_seed: int = 0,
        pool: str = "train-val",
    ):
        super().__init__()
        self.cache_root = Path(cache_root)
        self.dataset_root = Path(dataset_root)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.fu_jitter_scale = fu_jitter_scale
        self.p_drop_fu = p_drop_fu
        self.p_drop_bl = p_drop_bl
        self.graph = graph or GraphConfig()
        self.fold = fold
        self.n_folds = n_folds
        self.cv_seed = cv_seed
        assert pool in POOLS, f"pool {pool!r}: expected one of {POOLS}"
        self.pool = pool
        if fold is not None:
            assert fold in range(n_folds)
            assert n_folds >= 2

    def prepare_data(self) -> None:
        if self.pool == "train-val":
            sp = load_tracking_split()
            holdout = set(load_holdout(HOLDOUT_CSV))
            fit = set(map(str, sp["train"])) | set(map(str, sp["val"]))
            leak = fit & holdout
            if leak:
                raise SystemExit(
                    f"{len(leak)} holdout ids in train/val: {sorted(leak)[:8]}...\n"
                    f"Expected train/val disjoint from {HOLDOUT_CSV}.\n"
                    f"Fix: python3 lesionglue/cli/split.py --root {self.dataset_root}"
                )
            if self.fold is None and set(map(str, sp["test"])) != holdout:
                raise SystemExit(
                    f"split.json test ({len(sp['test'])}) != {HOLDOUT_CSV} ({len(holdout)}).\n"
                    f"Expected the holdout CSV to be the only eval split.\n"
                    f"Fix: python3 lesionglue/cli/split.py --root {self.dataset_root}"
                )
        for spn in POOL_SPLITS[self.pool]:
            p = self.cache_root / "processed" / f"{spn}_{CACHE_TAG}.pt"
            if not p.is_file():
                raise FileNotFoundError(
                    f"No graph cache at {p}.\n"
                    f"Expected preprocess output tagged {CACHE_TAG}.\n"
                    f"Fix: python3 lesionglue/cli/preprocess.py --split {spn} --jobs 16"
                )

    def setup(self, stage: str | None = None) -> None:
        g = self.graph
        if self.fold is None:
            parts = [
                LesionDataset(root=str(self.cache_root), split=sp, dataset_root=self.dataset_root, cfg=g)
                for sp in POOL_SPLITS[self.pool]
            ]
            n = sum(len(p) for p in parts)
            self.train_ds = _CvPool(
                parts, list(range(n)), augment=True, fu_jitter_scale=self.fu_jitter_scale,
                p_drop_fu=self.p_drop_fu, p_drop_bl=self.p_drop_bl, graph=g,
            )
            self.val_ds = None
            return
        train_pids, val_pids = fold_patient_sets(
            self.dataset_root, self.fold, self.n_folds, self.cv_seed, pool_pids=pool_patient_ids(self.dataset_root, self.pool)
        )
        pool = []
        for sp in POOL_SPLITS[self.pool]:
            pool.append(LesionDataset(root=str(self.cache_root), split=sp, dataset_root=self.dataset_root, cfg=g))
        train_idx, val_idx = [], []
        off = 0
        for ds in pool:
            for i in range(len(ds)):
                pid = str(ds[i].pid)
                if pid in train_pids:
                    train_idx.append(off + i)
                elif pid in val_pids:
                    val_idx.append(off + i)
            off += len(ds)
        if not train_idx or not val_idx:
            raise ValueError(
                f"fold {self.fold}: empty train or val after patient split\n"
                "Expected cached train and val graphs covering the patients of every fold.\n"
                "Fix: lesionglue_preprocess --split all"
            )
        self.train_ds = _CvPool(
            pool, train_idx, augment=True, fu_jitter_scale=self.fu_jitter_scale,
            p_drop_fu=self.p_drop_fu, p_drop_bl=self.p_drop_bl, graph=g,
        )
        self.val_ds = _CvPool(pool, val_idx, graph=g)

    def train_dataloader(self):
        nw = self.num_workers
        return PyGDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=nw, persistent_workers=nw > 0)

    def val_dataloader(self):
        if getattr(self, "val_ds", None) is None:
            return None
        nw = self.num_workers
        return PyGDataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False, num_workers=nw, persistent_workers=nw > 0)


class _CvPool:
    def __init__(
        self,
        parts: list[LesionDataset],
        indices: list[int],
        augment: bool = False,
        fu_jitter_scale: float = 0.3,
        p_drop_fu: float = 0.1,
        p_drop_bl: float = 0.1,
        graph: GraphConfig | None = None,
    ):
        self.parts = parts
        self.indices = indices
        self.offsets = []
        o = 0
        for ds in parts:
            self.offsets.append(o)
            o += len(ds)
        self.augment = augment
        self.fu_jitter_scale = fu_jitter_scale
        self.p_drop_fu = p_drop_fu
        self.p_drop_bl = p_drop_bl
        self.graph = graph or GraphConfig()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gidx = self.indices[i]
        for ds, off in zip(self.parts, self.offsets):
            if gidx < off + len(ds):
                g = ds[gidx - off]
                if not self.augment:
                    return refresh_edges(g.clone(), self.graph)
                d = g.clone()
                if self.p_drop_fu > 0.0 or self.p_drop_bl > 0.0:
                    drop_nodes(d, self.p_drop_fu, self.p_drop_bl, self.graph, rng=np.random.default_rng())
                if not self.graph.drop_dp:
                    jitter_both(d, self.graph, sigma_fu_scale=self.fu_jitter_scale, rng=np.random.default_rng())
                return refresh_edges(d, self.graph)
        raise IndexError(gidx)
