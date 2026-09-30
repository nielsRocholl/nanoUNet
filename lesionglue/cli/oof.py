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

from lesionglue.common import CACHE_ROOT, DATASET_ROOT, DEPLOYED_DUST_TAU, dump_json
from lesionglue.config import load_config
from lesionglue.eval.bootstrap import match_score_from_counts
from lesionglue.data.graph.dense import graph_config
from lesionglue.data.source.splits import fold_patient_sets
from lesionglue.train.datamodule import MatcherDataModule
from lesionglue.train.module import MatcherModule

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint to validate (e.g. best.ckpt, best_raw.ckpt, swa_plateau.ckpt)")
    ap.add_argument("--fold", type=int, required=True, help="CV fold whose held-out patients are scored; must be in [0, n_folds) of --config")
    ap.add_argument("--config", required=True, help="JSON config the checkpoint was trained with (gives n_folds, cv_seed, graph settings)")
    ap.add_argument("--root", default=str(DATASET_ROOT), help="dataset root directory passed to the datamodule and fold split")
    ap.add_argument("--cache", default=str(CACHE_ROOT), help="root of the cached lesion graphs")
    ap.add_argument("--out", required=True, help="output directory; val_per_patient.json is written here")
    ap.add_argument("--dust-tau", type=float, default=DEPLOYED_DUST_TAU, help="override checkpoint decode threshold")
    ap.add_argument("--no-ema", action="store_true", help="score the raw training weights instead of the EMA weights")
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert args.fold in range(cfg.n_folds)

    dm = MatcherDataModule(
        cache_root=Path(args.cache), dataset_root=Path(args.root),
        val_batch_size=cfg.val_batch_size, num_workers=cfg.num_workers,
        graph=graph_config(cfg), fold=args.fold, n_folds=cfg.n_folds, cv_seed=cfg.cv_seed,
    )
    dm.prepare_data()
    dm.setup()

    mod = MatcherModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    mod.set_eval_weights(not args.no_ema)
    mod._dust_ramp_step_override = 1_000_000_000  # full dust weight, not mid-ramp
    mod.hparams.dust_tau = args.dust_tau

    # CPU-only: 5 training jobs already saturate the single GPU (Round 12 constraint), and one
    # validation pass over a fold is cheap enough that GPU contention isn't worth the risk.
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=True)
    trainer.validate(mod, dataloaders=dm.val_dataloader())

    per_patient = mod._per_patient
    _, val_pids = fold_patient_sets(args.root, args.fold, cfg.n_folds, cfg.cv_seed)
    # fold_patient_sets covers the whole 270-patient train+val pool, but only 252 patients have
    # cached graphs (19 are skipped -- complete responders, registration failures, dominant-FU
    # casualties; see round12_findings.md A.1.2). So the reachable set is the pooled cache filtered
    # by fold membership, NOT val_pids itself -- comparing against val_pids can never pass.
    loaded = {str(dm.val_ds[i].pid) for i in range(len(dm.val_ds))}
    # R15: either mismatch means the wrong loader was evaluated -- crash, do not pass silently.
    assert loaded <= val_pids, f"fold {args.fold} val loader leaked non-fold patients: {loaded - val_pids}"
    assert set(per_patient) == loaded, (
        f"evaluated patient set != fold {args.fold} val loader: "
        f"missing={loaded - set(per_patient)} extra={set(per_patient) - loaded}"
    )

    totals = {k: sum(p[k] for p in per_patient.values()) for k in ("uc_ok", "uc_tot", "dis_ok", "dis_tot", "new_ok", "new_tot")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "val_per_patient.json", {
        "fold": args.fold,
        "ckpt": Path(args.ckpt).name,
        "weights": "raw" if args.no_ema else "ema",
        "n_patients": len(per_patient),
        "match_score": match_score_from_counts(totals),
        "acc_unchanged_split": totals["uc_ok"] / totals["uc_tot"] if totals["uc_tot"] else None,
        "acc_disappeared": totals["dis_ok"] / totals["dis_tot"] if totals["dis_tot"] else None,
        "acc_newly_appearing": totals["new_ok"] / totals["new_tot"] if totals["new_tot"] else None,
        "per_patient": per_patient,
    })
