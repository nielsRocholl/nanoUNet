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
| 01 | `exp01_segmentation` | 1 | implemented (smoke ok) |
| 02 | `exp02_prompt_noise` | 2 | implemented (smoke ok) |
| 03 | `exp03_matcher_alone` | 3 | not implemented |
| 04 | `exp04_baselines` | 4 | implemented, smoke-tested (8 patients); full run needs exp03 `folds.py` on main |
| 05 | `exp05_full_pipeline` | 5 | implemented (smoke pending; numbers wait for the matcher retrain) |
| 06 | `exp06_limits` | 6 | implemented, smoke-tested on a synthetic merge; full run needs an exp03 run |
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

Does the segmenter answer the point rather than the image? Four scenarios per scan (S1 all lesions clicked, S2 a strict subset, S3 no click,
S4 a decoy click on empty tissue) plus the prompt drop (S1 with the prompt channels zeroed), scored per lesion (Dice, NSD at 1 mm, detection at
IoU > 0.1) with patient-level bootstrap CIs. Cases come from the exp00c manifest; nanoUNet runs here, ULS+ and nnInteractive are run by the owner
on the click files this run writes and are scored from their prediction folders. Shared code: `experiments/segment.py`.

```bash
# 1. nanoUNet, all scenarios (also writes the click files the other systems need)
python -m experiments.exp01_segmentation.run --tag paper_v1
# 2. after ULS+ / nnInteractive were run on <run dir>/artifacts/prompts: score them too (placeholder paths)
python -m experiments.exp01_segmentation.run --tag paper_v1_ext --rescore /nnunet_data/experiments/exp01_segmentation/<RUN_ID> --external uls_plus=/path/to/uls_plus_predictions nninteractive=/path/to/nninteractive_predictions
# prompts only, no GPU, no scoring (the same files as in step 1)
python -m experiments.exp01_segmentation.run --tag paper_v1_prompts --emit-prompts-only
```

| flag | default | meaning |
|---|---|---|
| `--manifest` | `experiments/exp00c_seg_eval_manifest/seg_eval_v1.json` | eval manifest (schema `seg-eval-manifest/1`) |
| `--tier` | all three | `seen-cohort`, `outside`, `healthy` |
| `--sources` | all | keep only these manifest source names |
| `--max-cases-per-source` | -1 | first N cases of every source in manifest order; -1 = all |
| `--max-lesions-per-case` | -1 | click at most N lesions per case (seeded subset); -1 = all |
| `--methods` | `nanounet` | methods run here; empty (`--methods`) scores only `--external` folders |
| `--external` | none | `NAME=DIR` prediction folders, NAME = `nninteractive` or `uls_plus`; repeatable |
| `--overlap-policy` | `common` | `common`: every method on the cases clean for all three systems (headline); `own`: each method on its own clean set |
| `--backends` | both | registration-error table backends the click offsets are drawn from (`original`, `unigradicon`) |
| `--emit-prompts-only` | off | write lesion caches and click files to `artifacts/`, then stop |
| `--rescore` | none | `RUN_DIR` whose `artifacts/` (caches, clicks, nanoUNet masks) are rescored; needed to add `--external` folders later |
| `--tag`, `--out-root`, `--resume`, `--seed`, `--limit-patients`, `--device` | | common flags (table above) |

**Click files for the external systems.** One file per case and scenario: `<run dir>/artifacts/prompts/<scenario>/<case_id>.json` with
`<scenario>` in `S1`, `S2`, `S3`, `S4`. Format: `{"points": [{"name": "<lesion id or decoy>", "point": [x, y, z]}]}`, `[x, y, z]` = 0-based voxel
index of that case's image (NIfTI/SimpleITK index order, native grid). S1 = one click per lesion, S2 = the clicked subset (only cases with >= 2
lesions), S4 = one decoy click, S3 = `{"points": []}` (no click). The image of a case is the `image` column of `cases.csv` of the run
(`case_id` = file stem of the prediction). Clicks are identical for every system (seeded per case).

**Prediction folders.** `DIR/<scenario>/<case_id>.nii.gz` for every case the system is scored on and every scenario that case has a click
file for (`S1`, `S2`, `S3`, `S4`; no `S1_noprompt`). Binary mask, non-zero = lesion foreground, same grid (size) as the case image; a system
that emits nothing without a click writes an empty mask for `S3`. Missing or misplaced files stop the run at startup with the list.

Outputs (`results.json` tables): `per_lesion` (`method, tier, cohort, organ, patient, case, lesion_id, size_mm, size_bin, scenario, clicked,
offset_mm, click_hit, iou, hit, dice, nsd`), `per_case` (case means, S3/S4 foreground voxels and any-FG flag, S2 leak and selectivity margin),
`summary_table` (every CI by `all`, cohort, organ and size bin), `cases`; `artifacts/`: `lesions/`, `prompts/`, `preds/nanounet/<scenario>/`.
Runtime: about 20-40 s per case on one A100 (4 passes), 1-2 h at about 300 cases; resumable per pass (`--resume`), scoring alone with `--rescore`.
Notes that matter: S3 for nanoUNet is a tile with no click in it (S1 tiles with zeroed prompt), because a call with an empty click list does
nothing; ULS+ saw the held-out Longitudinal-CT cases, so under `--overlap-policy common` those cases are excluded for every method.

<!-- end -->

### exp02_prompt_noise

What does registration error cost the segmenter, and at which click offset does a lesion stop being found? For every lesion and replicate one
offset is drawn from the empirical registration-error table and scaled by s; all lesions of a scan are clicked at once and segmented in one pass
per (s, replicate). Dice and detection are reported against the scale and against the effective offset in mm, per lesion-size bin, with
patient-level bootstrap CIs. Replicate 0 at s = 1 uses exactly exp01's S1 clicks, which `--crosscheck-run` verifies.

```bash
python -m experiments.exp02_prompt_noise.run --tag paper_v1 --crosscheck-run /nnunet_data/experiments/exp01_segmentation/<RUN_ID>
```

| flag | default | meaning |
|---|---|---|
| `--manifest` | `experiments/exp00c_seg_eval_manifest/seg_eval_v1.json` | eval manifest (schema `seg-eval-manifest/1`) |
| `--tier` | `seen-cohort outside` | tiers to run (scans without lesions are skipped) |
| `--sources` | all | keep only these manifest source names |
| `--max-cases-per-source` | 10 | first N cases of every source in manifest order; -1 = all |
| `--max-lesions-per-case` | 8 | click at most N lesions per case (seeded subset); -1 = all |
| `--scales` | `0 0.25 0.5 0.75 1.0 1.5` | offset scales s (0 = true seed, 1 = the full empirical draw, 1.5 = stress point) |
| `--replicates` | 3 | independent offset draws per lesion (scale 0 runs once) |
| `--overlap-policy` | `common` | `common`: cases clean for all three systems; `own`: cases clean for nanoUNet |
| `--backends` | both | registration-error table backends the offsets are drawn from |
| `--crosscheck-run` | none | exp01 `RUN_DIR` (same manifest, `--seed`, `--backends`, `--max-lesions-per-case`) to cross-check against its S1 |
| `--rescore` | none | `RUN_DIR` of an exp02 run whose masks are rescored (no GPU) |
| `--tag`, `--out-root`, `--resume`, `--seed`, `--limit-patients`, `--device` | | common flags (table above) |

Outputs: `per_lesion` (one row per lesion, replicate and scale: `scale, replicate, offset_vox_zyx, offset_mm_zyx, offset_mm, offset_bin, size_mm,
size_bin, cohort, click_hit, dice, nsd, hit`), `by_scale`, `by_offset_mm`, `cases`; `artifacts/`: `lesions/`, `prompts/s<s>_r<r>/`, `preds/s<s>_r<r>/`.
Runtime: 16 passes per case (1 + 5 x 3), 40-90 s per case, 2-4 h at the default caps; resumable per pass. For the same clicks as exp01 use the
same `--seed`, `--backends` and `--max-lesions-per-case` (exp01 default -1, exp02 default 8: pass `--max-lesions-per-case -1` for an exact cross-check).

<!-- end -->

### exp03_matcher_alone

(not implemented)

<!-- end -->

### exp04_baselines

Reimplemented Di Veroli (iterative greedy overlap) and Qahqaie (unbalanced optimal transport, registration-trust term omitted) matchers, nested-tuned inside the exp03 folds and scored like our matcher (node supply Lstar, all 300 patients). Needs `experiments/exp03_matcher_alone/folds.py` (fold assignment); CPU only.

| flag | meaning |
|---|---|
| `--data-root` | Longitudinal-CT layout root (default `/nnunet_data/Longitudinal-CT`) |
| `--prop-fill` | propagated point for BL lesions that lack one: `none`, or `unigradicon` (default; uniGradICON `bl_click` only where its sanity check passed, the same rule as the graph builder's fill) |
| `--workers` | processes for the per-patient input preparation (default 8) |
| `--ours-run` | exp03 `RUN_DIR` whose `per_patient` table (columns `pid`, `decoder`, count keys of `scoring.COUNT_KEYS`) gives the paired deltas |
| `--ours-decoder` | decoder value of those rows to compare against (default `hungarian`) |
| `--rescore` | `RUN_DIR` of an earlier run: reuse its `artifacts/inputs/`, redo tuning and scoring (12 s for 8 patients) |

Smoke: `python -m experiments.exp04_baselines.run --limit-patients 8 --tag smoke` (needs patients from at least two folds, hence 8; 1.7 min). Full run (about 15-30 min preparation plus a few minutes of scoring):

```bash
python -m experiments.exp04_baselines.run --tag paper_v1 --ours-run /nnunet_data/experiments/exp03_matcher_alone/<run_id>
```

Outputs: `per_patient`, `chosen_params` (per fold), `tuning_scores`, `missing` tables; `edges_<method>.json`; `table.md` with all four class recalls and edge F1 per method (`diveroli_tuned`, `diveroli_published_r5|r7|r10`, `qahqaie_tuned`). Not implemented: the secondary Di Veroli input (uniGradICON-warped BL masks) and the with-unclear sensitivity row.

<!-- end -->

### exp05_full_pipeline

What does it cost when the segmenter, not the annotation, supplies the lesions? The held-out 60 patients (one scan pair each, the dominant
follow-up region) run through the deployment pipeline under three node supplies: **A** annotated lesions both sides (control), **B** annotated
BL + FU segmented from the propagated points (the protocol setting), **C** both scans segmented (BL from the true BL points, FU from the
propagated points; predicted nodes tied to annotation by IoU > 0.1). Matcher = `common.MATCHER_FINAL` (EMA weights), segmenter =
`common.SEG_CKPT` (EMA), decoders `hungarian` and `sinkhorn` at the matcher checkpoint's own `dust_tau`. Every setting reports recall per class
beside the identity ceiling, edge P/R/F1 and patient-bootstrap CIs, plus paired deltas A-B and B-C; the headline excludes `linking_unclear`
lesions and the with-unclear row is stored beside it. Raw pair logits are kept so decoding and scoring can be redone (`--rescore`). The pipeline
itself is `experiments/pipeline.py`, reused by exp07, exp08 and exp09.

**Status: implemented; smoke ok. The numbers are not meaningful until the owner retrains the matcher on the fixed graph cache and repoints
`MATCHER_FINAL`** (the current checkpoint never saw a merge-target node, merge recall is 0 by construction). Setting A builds its nodes with
`lesionglue.data.graph.dense.build_hetero_data`, so it follows the graph-builder fix, but it must be re-verified once that fix is on main.
Settings B and C build nodes from masks at inference time and do not depend on the fix.

```bash
# full run (held-out 60, settings A B C)
python -m experiments.exp05_full_pipeline.run --tag paper_v1
# resume a crashed run / redo only decoding and scoring (new run dir, optionally another tau)
python -m experiments.exp05_full_pipeline.run --tag paper_v1 --resume /nnunet_data/experiments/exp05_full_pipeline/<RUN_ID>
python -m experiments.exp05_full_pipeline.run --tag paper_v1_rescored --rescore /nnunet_data/experiments/exp05_full_pipeline/<RUN_ID>
# smoke (3 patients, kept out of the repository), GPU jobs under the shared lock
flock /tmp/gpu.lock python -m experiments.exp05_full_pipeline.run --limit-patients 3 --tag smoke
```

| flag | default | meaning |
|---|---|---|
| `--data-root` | `/nnunet_data/Longitudinal-CT` | dataset root (`inputsTrBL`, `inputsTrFU`, `targetsTrBL`, `targetsTrFU`, `meta`) |
| `--patients-csv` | `<data root>/test_patients.csv` | CSV with a `patient` column (the held-out 60) |
| `--patients` | none | explicit patient ids instead of the CSV (debugging, e.g. the known-bad `3988c7f88e`) |
| `--settings` | `A B C` | node supplies to run |
| `--matcher-ckpt` | `common.MATCHER_FINAL` | matcher checkpoint; the owner repoints the constant after the retrain |
| `--prop-fill` | `none` | setting A only: `unigradicon` fills BL lesions without `cog_propagated` (the graph builder's opt-in fill); use it if the matcher was trained on a cache built with `--prop-fill unigradicon` |
| `--tau` | the checkpoint's `dust_tau` | decoder cut-off; change only to rescore a sensitivity row |
| common flags | | `--tag --out-root --resume --seed --limit-patients --device --rescore` (table above) |

Outputs (run dir): `results.json` tables `per_patient` (per patient x setting x decoder x `headline|with_unclear`: `status`, `error`, node counts,
`t_seg`, `t_track`, counts per class `ok/tot/ceil`, `tp/fp/fn`, and for the headline rows the node -> annotated-id lists and the decoded links in
annotated-id space), `metrics` (every CI), `deltas`; `table.md` per decoder and variant. `artifacts/`: `pipeline.json`, `records/<pid>_<setting>.json`,
`scores/<pid>_<setting>.npz` (raw scores; setting A also `_A_unclear`), `masks/<pid>_<setting>/{matches.csv, pred_fu.mha, pred_bl.mha}`.
A patient the pipeline cannot process stays in the tables with its `status` and counts as fully missed (printed and listed in `notes`).
Runtime (measured on the smoke, A100 shared): about 30 s per patient for B, 35 s for C (FU about 3-8 s of segmentation, the rest is reading the CTs and building the graph), A 10-60 s (graph build), i.e. about 1.5-2 h for the 60 patients; resumable.

<!-- end -->

### exp06_limits

What the formulation can express but the data cannot score: split audit (no annotated splits), merge structure (38 events, group sizes), merge recall per contributing lesion under `hungarian` and `sinkhorn` (overall and by group size, patient bootstrap), the analytical limit `k <= 1/tau`, and an expressibility table. Reads the stored raw scores of an exp03 run (`artifacts/scores/<pid>.npz`); no inference, CPU, seconds.

| flag | meaning |
|---|---|
| `--data-root` | Longitudinal-CT layout root (default `/nnunet_data/Longitudinal-CT`) |
| `--from-run` | exp03 `RUN_DIR` whose stored scores are decoded again (required) |

```bash
python -m experiments.exp06_limits.run --tag paper_v1 --from-run /nnunet_data/experiments/exp03_matcher_alone/<run_id>
```

Smoke: any exp03 run dir works (`--tag smoke`); a synthetic k = 3 merge gave hungarian 1/3 and sinkhorn 3/3 as expected. Merge recall is only meaningful for scores produced on a cache built with the fixed graph builder (plan Sec. 2).

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
