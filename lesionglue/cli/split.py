"""Write lesionglue/configs/split.json: train/val from official 240, test = test_patients.csv (60)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.table import Table

from lesionglue.common import DATASET_ROOT, HOLDOUT_CSV, SPLIT_PATH, cprint, dump_json, nano_header
from lesionglue.data.source.splits import build_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT), help="Longitudinal-CT dataset root holding data_split.json (official train/val/test)")
    ap.add_argument("--holdout", default=str(HOLDOUT_CSV), help="CSV of held-out test patient ids (test_patients.csv); these become the test split")
    ap.add_argument("--out", default=str(SPLIT_PATH), help="output path for the tracking split JSON")
    ap.add_argument("--n-folds", type=int, default=5, help="number of patient-level folds carved from the official train patients")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the patient-to-fold assignment")
    ap.add_argument("--val-fold", type=int, default=0, help="fold index (0-based) used as val; the other folds form train")
    args = ap.parse_args()
    nano_header("lesionglue_split")
    official = Path(args.root) / "data_split.json"
    if not official.is_file():
        raise SystemExit(
            f"No official split at {official}.\n"
            f"Expected Longitudinal-CT data_split.json with train/val/test.\n"
            f"Fix: --root /nnunet_data/Longitudinal-CT"
        )
    sp = build_split(args.root, args.holdout, args.n_folds, args.seed, args.val_fold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out, sp)
    t = Table(title="tracking split", box=None, padding=(0, 2))
    t.add_column("split", style="cyan")
    t.add_column("n", justify="right")
    t.add_column("source")
    t.add_row("train", str(len(sp["train"])), "official train minus val fold")
    t.add_row("val", str(len(sp["val"])), f"fold {args.val_fold} of official train")
    t.add_row("test", str(len(sp["test"])), str(args.holdout))
    cprint(t)
    cprint(f"wrote {out}")


if __name__ == "__main__":
    main()
