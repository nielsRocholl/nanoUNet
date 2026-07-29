"""Post-hoc per-patient val metrics for one fold checkpoint.

Training-time snapshots in module.py can't tell best.ckpt / best_raw.ckpt / swa_plateau.ckpt
apart (see HANDOFF Stage B.2), and SWA has no validation pass of its own at all. This runs ONE
real validation pass of a given checkpoint over its OWN fold's held-out patients and dumps the
per-patient counts that validation_step already accumulates -- no metric logic is reimplemented
here. pool.py consumes the output to build pooled out-of-fold estimates across folds/selectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from tracking.common import CACHE_ROOT, DATASET_ROOT, dump_json
from tracking.config import load_config
from tracking.data.splits import fold_patient_sets, match_score_from_counts
from tracking.train.datamodule import MatcherDataModule
from tracking.train.module import MatcherModule

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dust-tau", type=float, default=None, help="override checkpoint decode threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert args.fold in range(cfg.n_folds)

    dm = MatcherDataModule(
        cache_root=Path(args.cache), dataset_root=Path(args.root),
        val_batch_size=cfg.val_batch_size, num_workers=cfg.num_workers,
        fold=args.fold, n_folds=cfg.n_folds, cv_seed=cfg.cv_seed,
    )
    dm.prepare_data()
    dm.setup()

    mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    mod._dust_ramp_step_override = 1_000_000_000  # full dust weight, not mid-ramp
    if args.dust_tau is not None:
        mod.hparams.dust_tau = args.dust_tau

    acc = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=acc, devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=True)
    trainer.validate(mod, dataloaders=dm.val_dataloader())

    per_patient = mod._per_patient
    _, val_pids = fold_patient_sets(args.root, args.fold, cfg.n_folds, cfg.cv_seed)
    # R15: a mismatch here means the wrong split was evaluated -- must crash, not pass silently.
    assert set(per_patient) == val_pids, (
        f"evaluated patient set != fold {args.fold} val set: "
        f"missing={val_pids - set(per_patient)} extra={set(per_patient) - val_pids}"
    )

    totals = {k: sum(p[k] for p in per_patient.values()) for k in ("uc_ok", "uc_tot", "dis_ok", "dis_tot", "new_ok", "new_tot")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "val_per_patient.json", {
        "fold": args.fold,
        "ckpt": Path(args.ckpt).name,
        "n_patients": len(per_patient),
        "match_score": match_score_from_counts(totals),
        "acc_unchanged_split": totals["uc_ok"] / totals["uc_tot"] if totals["uc_tot"] else None,
        "acc_disappeared": totals["dis_ok"] / totals["dis_tot"] if totals["dis_tot"] else None,
        "acc_newly_appearing": totals["new_ok"] / totals["new_tot"] if totals["new_tot"] else None,
        "per_patient": per_patient,
    })
