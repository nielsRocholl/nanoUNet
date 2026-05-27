"""Run the same validation metrics as training on val or test graphs (not predict CSVs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from tracking.common import CACHE_ROOT, DATASET_ROOT
from tracking.data.dataset import LesionDataset
from tracking.data.features import add_feat_args, desc_dim, feat_from_args
from tracking.train.module import MatcherModule


class SplitAsVal(pl.LightningDataModule):
    def __init__(self, loader):
        super().__init__()
        self._loader = loader

    def val_dataloader(self):
        return self._loader


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Same metric computation as training; printed names use test_* when --split test "
        "(Lightning still logs val_* internally because we reuse validation_step)."
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--dust-ramp-epoch",
        type=int,
        default=1000,
        help="epoch used only for dust loss ramp inside val_loss (standalone validate uses PL epoch 0); "
        "default 1000 => full dust weight, comparable to late training. Use -1 to leave unpatched.",
    )
    add_feat_args(ap)
    args = ap.parse_args()
    feat = feat_from_args(args)
    nw = max(0, args.num_workers)
    ds = LesionDataset(root=args.cache, split=args.split, dataset_root=Path(args.root), feat=feat)
    loader = PyGDataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        persistent_workers=nw > 0,
    )
    mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    ck_desc = int(mod.hparams.get("desc_dim", desc_dim(feat)))
    assert ck_desc == desc_dim(feat), f"ckpt desc_dim {ck_desc} != --feat {feat.mode} ({desc_dim(feat)})"
    if args.dust_ramp_epoch >= 0:
        mod._dust_ramp_epoch_override = args.dust_ramp_epoch
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=acc,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    out = trainer.validate(mod, datamodule=SplitAsVal(loader), verbose=False)
    if out:
        pfx = "test" if args.split == "test" else "val"
        for k, v in sorted(out[0].items()):
            disp = k.replace("val_", f"{pfx}_", 1) if k.startswith("val_") else k
            print(f"{disp}: {float(v):.6f}")
