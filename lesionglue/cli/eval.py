"""Run the same validation metrics as training on val or test graphs.

Standalone Trainer.validate resets global_step, so EMA must be forced via
set_eval_weights. --dust-tau may be repeated; one validate pass per tau.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from lesionglue.eval.bootstrap import match_score_from_counts
from lesionglue.common import CACHE_ROOT, DATASET_ROOT, DEPLOYED_CKPT, DEPLOYED_DUST_TAU, cprint, dump_json, nano_header, require_ckpt
from lesionglue.data.cache.dataset import LesionDataset
from lesionglue.infer import graph_cfg_from_ckpt
from lesionglue.train.module import MatcherModule

DUST_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.18, 0.20, 0.22, 0.25)


class SplitAsVal(pl.LightningDataModule):
    def __init__(self, loader):
        super().__init__()
        self._loader = loader

    def val_dataloader(self):
        return self._loader


def _counts(mod: MatcherModule) -> dict[str, int]:
    return {
        "uc_ok": int(mod._uc_ok), "uc_tot": int(mod._uc_tot),
        "dis_ok": int(mod._dis_ok), "dis_tot": int(mod._dis_tot),
        "new_ok": int(mod._new_ok), "new_tot": int(mod._new_tot),
    }


def _pick_tau(rows: list[dict]) -> dict:
    best = max(rows, key=lambda r: (r["match_score"], -abs(r["dust_tau"] - 0.10), -r["dust_tau"]))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEPLOYED_CKPT), help="matcher Lightning checkpoint (default: DEPLOYED_CKPT in lesionglue/common.py)")
    ap.add_argument("--split", choices=("val", "test"), default="test", help="cached split to evaluate")
    ap.add_argument("--cache", default=str(CACHE_ROOT), help="root of the cached lesion graphs")
    ap.add_argument("--root", default=str(DATASET_ROOT), help="dataset root directory passed to LesionDataset")
    ap.add_argument("--batch-size", type=int, default=8, help="graphs per validation batch")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes; 0 = load in the main process")
    ap.add_argument("--dust-tau", type=float, action="append", default=None, help="repeatable decode threshold")
    ap.add_argument("--no-ema", action="store_true", help="score the raw training weights instead of the EMA weights")
    ap.add_argument("--out", default="", help="optional JSON path for counts + selected tau")
    args = ap.parse_args()
    nano_header("lesionglue_eval")
    ckpt = require_ckpt(args.ckpt)

    nw = max(0, args.num_workers)
    use_ema = not args.no_ema
    taus = args.dust_tau if args.dust_tau else [DEPLOYED_DUST_TAU]
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=acc, devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)

    rows = []
    gcfg = None
    for tau in taus:
        mod = MatcherModule.load_from_checkpoint(str(ckpt), map_location="cpu")
        mod.set_eval_weights(use_ema)
        gcfg = graph_cfg_from_ckpt(mod, int(getattr(mod.hparams, "k_intra", 8)))
        ds = LesionDataset(root=args.cache, split=args.split, dataset_root=Path(args.root), cfg=gcfg)
        loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, persistent_workers=nw > 0)
        mod._dust_ramp_step_override = 1_000_000_000
        used_tau = float(tau)
        mod.hparams.dust_tau = used_tau
        out = trainer.validate(mod, datamodule=SplitAsVal(loader), verbose=False)
        counts = _counts(mod)
        row = {
            "dust_tau": used_tau, "weights": "ema" if use_ema else "raw",
            "n_graphs": len(ds), **counts, "match_score": match_score_from_counts(counts),
            "metrics": {k: float(v) for k, v in (out[0].items() if out else {})},
        }
        rows.append(row)
        pfx = "test" if args.split == "test" else "val"
        cprint(f"{pfx} tau={used_tau:.3f} weights={row['weights']} match={row['match_score']:.6f} persist={counts['uc_ok']}/{counts['uc_tot']}")

    chosen = _pick_tau(rows)
    payload = {
        "ckpt": str(ckpt), "split": args.split, "cache": args.cache, "root": args.root,
        "weights": "ema" if use_ema else "raw", "intra": None if gcfg is None else gcfg.intra,
        "drop_dp": None if gcfg is None else gcfg.drop_dp, "rows": rows, "selected": chosen,
    }
    if args.out.strip():
        dump_json(args.out.strip(), payload)
        cprint(f"wrote {args.out.strip()} selected_tau={chosen['dust_tau']:.3f}")
    else:
        cprint(f"selected_tau={chosen['dust_tau']:.3f} match={chosen['match_score']:.6f}")


if __name__ == "__main__":
    main()
