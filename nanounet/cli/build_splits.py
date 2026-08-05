"""Write a single train/val split balanced within each source dataset.

Replaces the dataset-blind 5-fold splits_final.json on merged pools: with 17 source datasets a
plain KFold drifts to 13-25% val per cohort, so small cohorts get unplottable val sets. The old
file is backed up, never silently overwritten."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from rich.table import Table

from nanounet.common import cprint, nano_header, preprocessed_dir, raw_dir
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.plan.dataset_id import convert_id_to_dataset_name
from nanounet.plan.plans import Plans
from nanounet.plan.splits import cohort_of, make_balanced_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_id", type=int, required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds = convert_id_to_dataset_name(args.dataset_id)
    nano_header(f"nanoUNet build-splits  {ds}  val_frac {args.val_frac}", color="green")

    pp = preprocessed_dir()
    pm = Plans(join(pp, ds, args.plans + ".json"))
    cm = pm.get_configuration("3d_fullres")
    case_dir = join(pp, ds, cm.data_identifier)

    all_ids = Blosc2Folder.get_identifiers(case_dir)
    dj = load_json(join(raw_dir(), ds, "dataset.json"))
    ntr = dj.get("numTraining")
    ids = all_ids[: int(ntr)] if ntr is not None else list(all_ids)

    out = join(pp, ds, "splits_final.json")
    if os.path.isfile(out):
        if not args.force:
            raise FileExistsError(
                f"{out} already exists ({len(load_json(out))} split(s)).\n"
                f"Overwriting changes which cases are held out and invalidates every existing val curve.\n"
                f"Fix: re-run with --force (the old file is backed up automatically)"
            )
        backup = join(pp, ds, f"splits_final.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        with open(out, encoding="utf-8") as f_in, open(backup, "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
        cprint(f"backed up old splits to {backup}")

    splits = make_balanced_split(ids, args.val_frac, args.seed)
    train, val = splits[0]["train"], splits[0]["val"]

    cohort_totals: Counter = Counter(cohort_of(i) for i in ids)
    cohort_train: Counter = Counter(cohort_of(i) for i in train)
    cohort_val: Counter = Counter(cohort_of(i) for i in val)

    t = Table(title="balanced split", box=None, padding=(0, 2))
    t.add_column("dataset", style="cyan")
    t.add_column("total", justify="right")
    t.add_column("train", justify="right")
    t.add_column("val", justify="right")
    t.add_column("val %", justify="right")
    for cohort, total in sorted(cohort_totals.items(), key=lambda kv: -kv[1]):
        nv = cohort_val[cohort]
        t.add_row(cohort, str(total), str(cohort_train[cohort]), str(nv), f"{100.0 * nv / total:.1f}")
    t.add_row("TOTAL", str(len(ids)), str(len(train)), str(len(val)), f"{100.0 * len(val) / len(ids):.1f}", style="bold")
    cprint(t)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    cprint(f"wrote {out}  (1 split, {len(train)} train / {len(val)} val)")
    cprint(f"next: nanounet_build_valset -d {args.dataset_id} --plans {args.plans} --config <cfg>")
