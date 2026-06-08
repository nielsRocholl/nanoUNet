"""LightningDataModule: cached dense v5 LesionDataset + PyG DataLoader."""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning import LightningDataModule
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader

from tracking.data.augment import drop_nodes, jitter_both

from tracking.common import CACHE_ROOT, DATASET_ROOT
from tracking.data.dataset import LesionDataset
from tracking.data.features import FeatConfig, cache_tag
from tracking.data.splits import fold_patient_sets


class MatcherDataModule(LightningDataModule):
    def __init__(
        self,
        cache_root: Path | str = CACHE_ROOT,
        dataset_root: Path | str = DATASET_ROOT,
        feat: FeatConfig | None = None,
        batch_size: int = 8,
        val_batch_size: int = 1,
        num_workers: int = 2,
        fu_jitter_scale: float = 0.3,
        p_drop_fu: float = 0.1,
        p_drop_bl: float = 0.1,
        desc_jitter_frac: float = 0.0,
        fold: int | None = None,
        n_folds: int = 5,
        cv_seed: int = 0,
    ):
        super().__init__()
        self.cache_root = Path(cache_root)
        self.dataset_root = Path(dataset_root)
        self.feat = feat or FeatConfig()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.fu_jitter_scale = fu_jitter_scale
        self.p_drop_fu = p_drop_fu
        self.p_drop_bl = p_drop_bl
        self.desc_jitter_frac = desc_jitter_frac
        self.fold = fold
        self.n_folds = n_folds
        self.cv_seed = cv_seed
        if fold is not None:
            assert fold in range(n_folds)
            assert n_folds >= 2

    def prepare_data(self) -> None:
        tag = cache_tag(self.feat)
        splits = ("train", "val") if self.fold is not None else ("train", "val", "test")
        for sp in splits:
            p = self.cache_root / "processed" / f"{sp}_{tag}.pt"
            if not p.is_file():
                raise FileNotFoundError(f"run preprocess --split {sp} --feat {self.feat.mode}; missing {p}")

    def setup(self, stage: str | None = None) -> None:
        if self.fold is None:
            self.train_ds = LesionDataset(
                root=str(self.cache_root),
                split="train",
                dataset_root=self.dataset_root,
                feat=self.feat,
                augment=True,
                fu_jitter_scale=self.fu_jitter_scale,
                p_drop_fu=self.p_drop_fu,
                p_drop_bl=self.p_drop_bl,
                desc_jitter_frac=self.desc_jitter_frac,
            )
            self.val_ds = LesionDataset(
                root=str(self.cache_root),
                split="val",
                dataset_root=self.dataset_root,
                feat=self.feat,
            )
            return
        train_pids, val_pids = fold_patient_sets(self.dataset_root, self.fold, self.n_folds, self.cv_seed)
        pool = []
        for sp in ("train", "val"):
            ds = LesionDataset(root=str(self.cache_root), split=sp, dataset_root=self.dataset_root, feat=self.feat)
            pool.append(ds)
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
            raise ValueError(f"fold {self.fold}: empty train or val after patient split")
        self.train_ds = _CvPool(pool, train_idx, augment=True, fu_jitter_scale=self.fu_jitter_scale, p_drop_fu=self.p_drop_fu, p_drop_bl=self.p_drop_bl, desc_jitter_frac=self.desc_jitter_frac)
        self.val_ds = _CvPool(pool, val_idx)

    def train_dataloader(self):
        nw = self.num_workers
        return PyGDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=nw,
            persistent_workers=nw > 0,
        )

    def val_dataloader(self):
        nw = self.num_workers
        return PyGDataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=nw,
            persistent_workers=nw > 0,
        )


class _CvPool:
    def __init__(
        self,
        parts: list[LesionDataset],
        indices: list[int],
        augment: bool = False,
        fu_jitter_scale: float = 0.3,
        p_drop_fu: float = 0.1,
        p_drop_bl: float = 0.1,
        desc_jitter_frac: float = 0.0,
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
        self.desc_jitter_frac = desc_jitter_frac

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gidx = self.indices[i]
        for ds, off in zip(self.parts, self.offsets):
            if gidx < off + len(ds):
                g = ds[gidx - off]
                if not self.augment:
                    return g
                d = g.clone()
                feat, cfg = ds.feat, ds.cfg
                rng = np.random.default_rng()
                if self.p_drop_fu > 0.0 or self.p_drop_bl > 0.0:
                    drop_nodes(d, self.p_drop_fu, self.p_drop_bl, cfg.k_intra, feat, rng=rng)
                jitter_both(d, k_intra=cfg.k_intra, sigma_fu_scale=self.fu_jitter_scale, rng=rng, desc_jitter_frac=self.desc_jitter_frac, feat=feat)
                return d
        raise IndexError(gidx)
