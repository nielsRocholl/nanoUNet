"""LightningDataModule: cached dense v5 LesionDataset + PyG DataLoader."""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader

from tracking.common import CACHE_ROOT, DATASET_ROOT
from tracking.data.dataset import LesionDataset
from tracking.data.features import FeatConfig, cache_tag


class MatcherDataModule(LightningDataModule):
    def __init__(
        self,
        cache_root: Path | str = CACHE_ROOT,
        dataset_root: Path | str = DATASET_ROOT,
        feat: FeatConfig | None = None,
        batch_size: int = 8,
        num_workers: int = 2,
        fu_jitter_scale: float = 0.3,
        p_drop_fu: float = 0.1,
        p_drop_bl: float = 0.1,
        desc_jitter_frac: float = 0.0,
    ):
        super().__init__()
        self.cache_root = Path(cache_root)
        self.dataset_root = Path(dataset_root)
        self.feat = feat or FeatConfig()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fu_jitter_scale = fu_jitter_scale
        self.p_drop_fu = p_drop_fu
        self.p_drop_bl = p_drop_bl
        self.desc_jitter_frac = desc_jitter_frac

    def prepare_data(self) -> None:
        tag = cache_tag(self.feat)
        for sp in ("train", "val"):
            p = self.cache_root / "processed" / f"{sp}_{tag}.pt"
            if not p.is_file():
                raise FileNotFoundError(f"run preprocess --split {sp} --feat {self.feat.mode}; missing {p}")

    def setup(self, stage: str | None = None) -> None:
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
