"""Create-or-verify splits_final.json and cohorts.json for a dataset folder (D6).

Both files are shared by every plans variant of a dataset and define which cases are held out, so an
existing file is NEVER rewritten: it is verified against the dataset and kept. Only when absent are
they derived (deterministically, from the raw dataset.json) and written. The Longitudinal-CT
test-patient guard runs in both paths."""

from __future__ import annotations

import hashlib
import json
import os

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from core.ui import cprint
from nanounet.common import raw_dir
from nanounet.plan.dataset.cohorts import cohorts_doc, run_cohorts
from nanounet.plan.dataset.holdout import assert_no_test_patients
from nanounet.plan.dataset.ids import convert_id_to_dataset_name
from nanounet.plan.dataset.splits import make_balanced_split


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ensure_splits_and_cohorts(did: int, pre: str, case_ids_on_disk: set[str], val_frac: float, seed: int) -> dict:
    """Returns {"splits_sha256", "cohorts_sha256", "splits_status", "cohorts_status"} ('created'|'kept')."""
    dj = load_json(join(raw_dir(), convert_id_to_dataset_name(did), "dataset.json"))
    ids = sorted(dj["dataset"])
    sp_path, co_path = join(pre, "splits_final.json"), join(pre, "cohorts.json")
    out: dict = {}
    if os.path.isfile(sp_path):
        splits = load_json(sp_path)
        got = set(splits[0]["train"]) | set(splits[0]["val"])
        missing = sorted(got - case_ids_on_disk)
        extra = sorted(case_ids_on_disk - got)
        if got != set(ids) or missing or extra:
            raise SystemExit(
                f"{sp_path} does not match the dataset: {len(missing)} split id(s) missing from the data folder (e.g. {missing[:3]}), "
                f"{len(extra)} data-folder case(s) not in the splits (e.g. {extra[:3]}), {len(got ^ set(ids))} differ from dataset.json.\n"
                f"Expected every split id to be preprocessed and every case to be in a split; the existing file is never rewritten.\n"
                f"Fix: preprocess the missing cases (nanounet_preprocess -d {did} --resume), or remove the stale file yourself if the dataset changed on purpose"
            )
        out["splits_status"] = "kept"
    else:
        splits = make_balanced_split(ids, val_frac, seed)
        with open(sp_path, "w", encoding="utf-8") as f:
            json.dump(splits, f)
        out["splits_status"] = "created"
    n_pat = assert_no_test_patients(splits[0]["train"] + splits[0]["val"], dj)
    if os.path.isfile(co_path):
        if load_json(co_path) != json.loads(json.dumps(cohorts_doc(did))):
            raise SystemExit(
                f"{co_path} differs from the cohorts derived from the raw dataset.\n"
                f"Expected the identical content (cohort weights are deterministic from merged_sources.json); the existing file is never rewritten.\n"
                f"Fix: diff it against nanounet.plan.dataset.cohorts.cohorts_doc({did}); delete the file yourself if the sources changed on purpose"
            )
        out["cohorts_status"] = "kept"
    else:
        run_cohorts(did, pre)
        out["cohorts_status"] = "created"
    out["splits_sha256"], out["cohorts_sha256"] = sha256_of(sp_path), sha256_of(co_path)
    cprint(
        f"[bold green]✓ splits {out['splits_status']}[/bold green] sha256 {out['splits_sha256'][:16]}…  "
        f"({len(splits[0]['train'])} train / {len(splits[0]['val'])} val)  |  "
        f"[bold green]✓ cohorts {out['cohorts_status']}[/bold green] sha256 {out['cohorts_sha256'][:16]}…  |  test-patient guard: 0 of {n_pat} found"
    )
    return out
