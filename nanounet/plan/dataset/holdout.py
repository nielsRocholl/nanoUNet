"""Hard guard: the Longitudinal-CT test patients must never appear in train or val.

Matches three ways because any one alone has leaked before: the case's patient key (patient_of),
the patient id as a substring of the case id, and as a substring of the dataset.json image/label
paths (a renamed case id can still point at a test scan). Run at split creation, preprocess
splits-safety and train startup; a missing CSV is an error, never a silent skip (R12)."""

from __future__ import annotations

import csv
import os
from typing import Iterable

from nanounet.plan.dataset.splits import patient_of

TEST_PATIENTS_CSV = "/nnunet_data/Longitudinal-CT/test_patients.csv"


def load_test_patients(path: str = TEST_PATIENTS_CSV) -> set[str]:
    if not os.path.isfile(path):
        raise SystemExit(
            f"No test-patient list at {path}.\n"
            f"Expected a CSV with a 'patient' column (60 Longitudinal-CT hold-out patients) so they can be kept out of training.\n"
            f"Fix: mount /nnunet_data, or restore the file and re-run; set TEST_PATIENTS_CSV in nanounet/plan/dataset/holdout.py if it moved"
        )
    with open(path, encoding="utf-8") as f:
        pats = {r["patient"].strip() for r in csv.DictReader(f)}
    assert len(pats) == 60, f"{path}: {len(pats)} test patients, expected 60"
    return pats


def assert_no_test_patients(case_ids: Iterable[str], dataset_json: dict | None = None) -> int:
    """Raise naming every offending case; returns the number of test patients checked (0 matches)."""
    pats = load_test_patients()
    ids = list(case_ids)
    hits: dict[str, str] = {}
    for cid in ids:
        key = patient_of(cid)
        for p in pats:
            if p in cid or key in (p, f"d013_{p}"):
                hits[cid] = p
    if dataset_json is not None:
        ds = dataset_json["dataset"]
        for cid in ids:
            e = ds.get(cid)
            paths = [*e["images"], e["label"]] if e else []
            for p in pats:
                if any(p in x for x in paths):
                    hits[cid] = p
    if hits:
        rows = "\n".join(f"  {c} (test patient {p})" for c, p in sorted(hits.items())[:20])
        raise SystemExit(
            f"{len(hits)} case(s) belong to Longitudinal-CT test patients listed in {TEST_PATIENTS_CSV}:\n{rows}\n"
            f"Expected none: test patients must never be trained or validated on.\n"
            f"Fix: remove these cases from the dataset (raw dataset.json) and rebuild the splits"
        )
    return len(pats)
