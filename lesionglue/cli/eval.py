"""Run the same validation metrics as training on val or test graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from tracking.common import CACHE_ROOT, DATASET_ROOT, cprint, nano_header
from tracking.data.dataset import LesionDataset
from tracking.infer import graph_cfg_from_ckpt
from tracking.train.module import MatcherModule


class SplitAsVal(pl.LightningDataModule):
    def __init__(self, loader):
        super().__init__()
        self._loader = loader

    def val_dataloader(self):
        return self._loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--dust-tau", type=float, default=None, help="override checkpoint decode threshold")
    args = ap.parse_args()
    nano_header("lesion_track_eval")
    if not Path(args.ckpt).is_file():
        raise SystemExit(
            f"No checkpoint at {args.ckpt}.\n"
            f"Expected a Lightning .ckpt from lesion_track_train.\n"
            f"Fix: --ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt"
        )

    nw = max(0, args.num_workers)
    mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    gcfg = graph_cfg_from_ckpt(mod, int(getattr(mod.hparams, "k_intra", 8)))
    ds = LesionDataset(root=args.cache, split=args.split, dataset_root=Path(args.root), cfg=gcfg)
    loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, persistent_workers=nw > 0)
    mod._dust_ramp_step_override = 1_000_000_000
    if args.dust_tau is not None:
        mod.hparams.dust_tau = args.dust_tau
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=acc, devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    out = trainer.validate(mod, datamodule=SplitAsVal(loader), verbose=False)
    if out:
        pfx = "test" if args.split == "test" else "val"
        for k, v in sorted(out[0].items()):
            disp = k.replace("val_", f"{pfx}_", 1) if k.startswith("val_") else k
            cprint(f"{disp}: {float(v):.6f}")


if __name__ == "__main__":
    main()
