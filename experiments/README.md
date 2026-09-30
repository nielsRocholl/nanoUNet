# experiments

The paper's experiments (LesionGlue: graph optimal-transport lesion correspondence in longitudinal CT), one folder per
experiment, every run recorded so that a table or plot can be regenerated years later from `results.json` alone.
The design and the exact protocol of each experiment are in [`docs/dev-notes/experiments_plan.md`](../docs/dev-notes/experiments_plan.md).

## Run an experiment

```bash
cd /nanoUNet
python -m experiments.<expNN_name>.run --help
python -m experiments.<expNN_name>.run --tag paper_v1
```

The `python -m` form works from the repo root without any install. After `pip install -e . --no-deps` the file form
`python experiments/<expNN_name>/run.py ...` works from anywhere. Each `run.py` opens with a plain-language
description (QUESTION / WHY / DATA / METHOD / OUTPUT / COMMAND / DEPENDS ON / RUNTIME / CAVEATS).

Smoke test (any experiment): `--limit-patients 3 --tag smoke`. A tag containing `smoke` keeps the run out of git.

Common flags (every experiment; `--device` only where GPU work happens, `--rescore` only where prediction is separate from scoring):

| flag | meaning |
|---|---|
| `--tag` | label appended to the run id; `smoke` keeps the run out of git |
| `--out-root` | root of the full run directories (default `/nnunet_data/experiments`) |
| `--resume` | `RUN_DIR` of an unfinished run: reuse it and skip artifacts that already exist |
| `--seed` | seed for every random draw (recorded in `run.json`) |
| `--limit-patients` | use only the first N patients/cases; `-1` = all |
| `--device` | torch device for the GPU work |
| `--rescore` | `RUN_DIR` whose `artifacts/` are rescored without prediction; writes a new run dir |

## The run directory

Full run: `/nnunet_data/experiments/<exp>/<run_id>/` (`run_id = YYYYMMDDTHHMMSSZ_<tag>`); everything except `artifacts/`
is mirrored to `experiments/results/<exp>/<run_id>/` (tracked in git; files above 20 MB stay on `/nnunet_data` and are
listed in `run.json` as `mirror_skipped`). Every finished run adds a line to `INDEX.jsonl` in both places.

| file | content |
|---|---|
| `command.txt` | line 1 the command as typed, line 2 the same command with every flag explicit, then cwd and git sha; a resumed run appends another block |
| `run.json` | provenance: status (`running`, `ok`, `failed`), timestamps, git sha and dirty flag, host, GPU, package versions, seeds, `NANOUNET_*`, inputs with sha256, output paths, traceback on failure |
| `log.txt` | what the terminal showed (ANSI stripped) |
| `results.json` | source of truth, schema `lesionglue-exp/1`: `paper`, `definitions`, `summary`, `tables` (every per-lesion / per-patient / per-fold row), `notes` |
| `<table>.csv` | one tidy CSV per table in `results.json` |
| `table.md` | the paper-shaped summary with confidence intervals |
| `artifacts/` | heavy files only (predicted masks, raw pair scores, fold checkpoints); never mirrored |

A run that crashes keeps its command and traceback: `run.json` says `failed` (or stays `running` if the process was killed).
Stdout carries exactly one JSON line `{"exp", "run_id", "out_dir", "status"}`; everything else is on stderr.

## Conventions

- One folder per experiment, `expNN_name/run.py` holds the science; side modules sit in the same folder; shared code lives in
  `common.py` (run record), `scoring.py` (metrics), `segment.py`, `pipeline.py` (added by the experiments that need them).
- A later experiment imports an earlier experiment's side module only where its docstring says `DEPENDS ON`. No cycles.
- Missing data is an error with a `Fix:` line, never a silent drop; a patient that cannot be processed stays in the tables with a `status` and counts as missed.
- The 200-line cap is waived inside this project (owner decision); every other nanochat-style rule stands.
- Never write under `targetsTr*` / `inputsTr*`; `/nnunet_data` is a CIFS mount (`shutil.copy2` fails, use `copyfileobj`).

## Skeleton of a `run.py`

Order is fixed (parse, validate everything, run record, work, finish); `start_run` prints the header and config table and starts the log tee.

```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)                                   # --tag --out-root --resume --seed --limit-patients --device
    ap.add_argument("--data-root", type=Path, default=LONGI_ROOT, help="...")
    args = ap.parse_args()
    abort_if(missing_paths({"data root": args.data_root}, "mount /nnunet_data"))   # E6: all problems at once, each with Fix:
    run = start_run(EXP, ap, args, inputs={"holdout": HOLDOUT_CSV}, paper={"section": "...", "table_row": 3})
    ...                                                   # predict() into run.artifacts (skip what exists), then score()
    run.finish(summary, {"per_patient": rows}, table_md=md, notes=[...], next_cmd="...")
```

## Experiments

| id | folder | paper row | status |
|---|---|---|---|
| 00a | `exp00a_data_audit` | data section | not implemented |
| 00b | `exp00b_calibration` | calibration | not implemented |
| 00c | `exp00c_seg_eval_manifest` | evaluation data for rows 1 and 2 | not implemented |
| 01 | `exp01_segmentation` | 1 | not implemented |
| 02 | `exp02_prompt_noise` | 2 | not implemented |
| 03 | `exp03_matcher_alone` | 3 | not implemented |
| 04 | `exp04_baselines` | 4 | not implemented |
| 05 | `exp05_full_pipeline` | 5 | not implemented |
| 06 | `exp06_limits` | 6 | not implemented |
| 07 | `exp07_internal_set` | 7 | not implemented |
| 08 | `exp08_external_set` | 8 | not implemented |
| 09 | `exp09_pantrack` | 9 | not implemented |

Each experiment owns one section below (arguments, literal full-run command, outputs). Edit only your own section; the
`<!-- end -->` lines keep neighbouring edits from colliding in git.

## Sections

### exp00a_data_audit

(not implemented)

<!-- end -->

### exp00b_calibration

(not implemented)

<!-- end -->

### exp00c_seg_eval_manifest

(not implemented)

<!-- end -->

### exp01_segmentation

(not implemented)

<!-- end -->

### exp02_prompt_noise

(not implemented)

<!-- end -->

### exp03_matcher_alone

(not implemented)

<!-- end -->

### exp04_baselines

(not implemented)

<!-- end -->

### exp05_full_pipeline

(not implemented)

<!-- end -->

### exp06_limits

(not implemented)

<!-- end -->

### exp07_internal_set

(not implemented)

<!-- end -->

### exp08_external_set

(not implemented)

<!-- end -->

### exp09_pantrack

(not implemented)

<!-- end -->

## Runbook

(the lead fills this from the agents' reports: the run order of the plan's Sec. 10 with the literal commands and estimated runtimes)

<!-- end -->

## Container contract (private-set experiments 07 and 08)

(written by the exp07/exp08 agent: flags and paths, the single Python environment, GPU need, licence note)

<!-- end -->
