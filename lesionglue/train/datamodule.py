"""LightningDataModule over cached dense v2 LesionDataset + PyG DataLoader."""

from __future__ import annotations

from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader

from tracking.common import CACHE_ROOT, DATASET_ROOT
from tracking.data.dataset import LesionDataset


class MatcherDataModule(LightningDataModule):
    def __init__(
        self,
        cache_root: Path | str = CACHE_ROOT,
        dataset_root: Path | str = DATASET_ROOT,
        batch_size: int = 8,
        num_workers: int = 2,
    ):
        super().__init__()
        self.cache_root = Path(cache_root)
        self.dataset_root = Path(dataset_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pos_weight = 1.0

    def prepare_data(self) -> None:
        for sp in ("train", "val"):
            p = self.cache_root / "processed" / f"{sp}_v2.pt"
            if not p.is_file():
                raise FileNotFoundError(f"run preprocess --split {sp}; missing {p}")

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = LesionDataset(root=str(self.cache_root), split="train", dataset_root=self.dataset_root)
        self.val_ds = LesionDataset(root=str(self.cache_root), split="val", dataset_root=self.dataset_root)
        meta = torch.load(self.cache_root / "processed" / "train_v2_meta.pt", map_location="cpu")
        self.pos_weight = float(meta["pos_weight"])

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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=nw,
            persistent_workers=nw > 0,
        )
