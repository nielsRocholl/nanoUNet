"""exp03 fold helpers: the 300 patients, the 5-fold assignment, the graph to score per patient, and a resumable fold trainer.

One definition of the folds: `assign_folds` is lesionglue's own `fold_map`, and `lesionglue_train --pool all --fold k` builds its
held-out set with the same function on the same patient list (train+val+test of `lesionglue/configs/split.json`), so the fold a
patient is scored in is exactly the fold the model that scores it never trained on. All 300 patients get a fold, also the ones
without a graph (complete responders: nothing to match); they are scored trivially by the experiment, never dropped.

Graph caches hold one graph per (patient, FU body-region volume): on the pre-fix cache 306 graphs for 283 patients, 19 patients with
2-3 graphs, 17 patients with none. Training may use every graph; scoring uses the paper's pair, the graph whose FU image id equals
the patient's most frequent `img_id_fu` in `meta/<pid>.csv` (ties: the smallest id, as lesionglue's own `_img_ids`).
`dominant_graph_index` picks it; a patient absent from its result has no graph of the dominant region (two patients on the old cache
only had a non-dominant one) and must not be scored from another region.

`train_fold` calls the documented entry point `lesionglue.cli.train` (console script `lesionglue_train`) in a subprocess: there is no
public lesionglue function that trains one fold, and a subprocess also gives each fold a fresh CUDA context and its own log. The done
marker is `fold_metrics.json`, not `last.ckpt`: `ModelCheckpoint(save_last=True)` rewrites `last.ckpt` after every epoch, so a killed
run leaves a half-trained `last.ckpt`.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from core.ui import cprint
from experiments.common import LONGI_ROOT, MATCHER_FINAL, REPO
from lesionglue.data.source.splits import fold_map, pool_patient_ids

N_PATIENTS = 300
GRAPH_CACHE = Path("/nnunet_data/lesion_tracking/cache_v9")  # lesionglue_preprocess --split all --prop-fill unigradicon (fixed builder, tag v8_native)
FOLD_CONFIG = MATCHER_FINAL.parent / "config.json"  # the final_v8_noval recipe: fixed max_steps, no validation, last.ckpt


def all_patients() -> list[str]:
    """The 300 patients (train+val+test of the tracking split), sorted."""
    pids = sorted(pool_patient_ids(LONGI_ROOT, "all"))
    assert len(pids) == N_PATIENTS == len(set(pids)), f"expected {N_PATIENTS} distinct patients in lesionglue/configs/split.json, got {len(pids)}"
    return pids


def assign_folds(pids: list[str], k: int = 5, seed: int = 0) -> dict[str, int]:
    """patient -> fold in [0, k): lesionglue's `fold_map`, so `lesionglue_train --pool all --fold` holds out exactly these patients."""
    return fold_map(pids, k, seed)


def dominant_fu_ids(pids: list[str], root: Path = LONGI_ROOT) -> dict[str, int]:
    """pid -> most frequent `img_id_fu` over the patient's meta rows (ties: the smallest id)."""
    missing = [p for p in pids if not (root / "meta" / f"{p}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"no meta CSV for {len(missing)} patient(s), e.g. {missing[:3]}\nExpected {root}/meta/<pid>.csv\nFix: mount /nnunet_data (Longitudinal-CT)")
    out = {}
    for pid in pids:
        with (root / "meta" / f"{pid}.csv").open(newline="", encoding="utf-8") as f:
            n = Counter(int(float(r["img_id_fu"])) for r in csv.DictReader(f))
        out[pid] = max(n, key=lambda i: (n[i], -i))
    return out


def graph_keys(datasets) -> list[tuple[str, int]]:
    """(pid, img_id_fu) of every graph of the given `LesionDataset`s, concatenated in order (the index space of `dominant_graph_index`)."""
    return [(str(g.pid), int(g.img_id_fu_used)) for ds in datasets for g in ds]


def dominant_graph_index(keys: list[tuple[str, int]], root: Path = LONGI_ROOT) -> dict[str, int]:
    """pid -> position in `keys` of the graph of the patient's dominant FU region. A patient without such a graph is absent."""
    dom = dominant_fu_ids(sorted({pid for pid, _ in keys}), root)
    out: dict[str, int] = {}
    for i, (pid, fu) in enumerate(keys):
        if fu == dom[pid]:
            assert pid not in out, f"{pid}: two graphs of FU region {fu} at positions {out[pid]} and {i}"
            out[pid] = i
    return out


def train_fold(artifacts: Path, fold: int, k: int = 5, seed: int = 0, cache: Path = GRAPH_CACHE, max_steps: int | None = None) -> Path:
    """Train fold `fold` (held out) on the other k-1 folds of all 300 patients, unless it is already done; return its `last.ckpt`.

    Writes `artifacts/folds/fold<fold>/`: `config_in.json` (the recipe with n_folds=k, cv_seed=seed), `train.log`, and what
    `lesionglue_train --pool all --fold --no-val` writes (`config.json`, `fold_metrics.json`, `last.ckpt`). One fold per call; the caller
    (or a shell loop / parallel jobs) runs the others. `seed` drives both the fold assignment and the training.
    """
    out = artifacts / "folds" / f"fold{fold}"
    recipe = json.loads(FOLD_CONFIG.read_text())
    steps = int(recipe["max_steps"] if max_steps is None else max_steps)
    last, done = out / "last.ckpt", out / "fold_metrics.json"
    if last.is_file() and done.is_file():
        m = json.loads(done.read_text())
        if (m.get("fold"), m.get("seed"), m.get("max_steps"), m.get("pool")) != (fold, seed, steps, "all"):
            raise SystemExit(
                f"fold {fold} in {out} was trained with fold/seed/max_steps/pool = {m.get('fold')}/{m.get('seed')}/{m.get('max_steps')}/{m.get('pool')}, not {fold}/{seed}/{steps}/all\n"
                "Expected the finished fold to match this call, so a resumed run never mixes recipes.\n"
                f"Fix: use a fresh run dir, or delete {out} to retrain this fold"
            )
        cprint(f"status: fold {fold} training skipped | {last} exists (max_steps {steps})")
        return last
    if out.exists():  # a killed run: move it aside instead of deleting or overwriting it
        aside = out.with_name(f"fold{fold}.partial-{int(time.time())}")
        out.rename(aside)
        cprint(f"status: fold {fold} unfinished run moved to {aside.name}")
    out.mkdir(parents=True)
    (out / "config_in.json").write_text(json.dumps({**recipe, "n_folds": k, "cv_seed": seed}, indent=2) + "\n")
    cmd = [sys.executable, "-m", "lesionglue.cli.train", "--config", str(out / "config_in.json"), "--cache", str(cache), "--out", str(out),
           "--pool", "all", "--fold", str(fold), "--no-val", "--seed", str(seed), "--max-steps", str(steps)]
    cprint(f"status: fold {fold} training | {' '.join(cmd)}")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])])}  # this checkout's lesionglue, not an installed one
    t0 = time.time()
    with (out / "train.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
    if code != 0 or not (last.is_file() and done.is_file()):
        tail = "".join((out / "train.log").read_text(encoding="utf-8").splitlines(keepends=True)[-12:])
        raise SystemExit(f"fold {fold} training failed (exit {code}); last lines of {out / 'train.log'}:\n{tail}\nExpected {last} and {done}.\nFix: rerun the same command; {out} is moved aside and the fold retrains")
    cprint(f"status: fold {fold} trained | {time.time() - t0:.0f}s | {last}")
    return last
