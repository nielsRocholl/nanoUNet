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
| 00a | `exp00a_data_audit` | data section | implemented, full run paper_v1 on the old cache (rerun after the v9 cache) |
| 00b | `exp00b_calibration` | calibration | implemented, full run paper_v1 |
| 00c | `exp00c_seg_eval_manifest` | evaluation data for rows 1 and 2 | implemented, full run paper_v1 (`seg_eval_v1.json`) |
| 01 | `exp01_segmentation` | 1 | implemented (smoke ok); full run needs the owner's ULS+ and nnInteractive prediction folders for the external rows |
| 02 | `exp02_prompt_noise` | 2 | implemented (smoke ok) |
| 03 | `exp03_matcher_alone` | 3 | implemented (20-step smoke ok on `cache_v9_merge`, with exp04 and exp06 smoked on its output); full run on `cache_v9_merge` |
| 04 | `exp04_baselines` | 4 | implemented, smoke-tested (8 patients); full run needs an exp03 run |
| 05 | `exp05_full_pipeline` | 5 | implemented (smoke ok with the retrained matcher, 3 patients); full run on the held-out 60 |
| 06 | `exp06_limits` | 6 | implemented, smoke-tested on a synthetic merge; full run needs an exp03 run |
| 07 | `exp07_internal_set` | 7 | implemented (smoke ok with the retrained matcher and `difference_weighting` 0.2.0; ours equals exp05 setting B on 3 patients); full run on the held-out 60 |
| 08 | `exp08_external_set` | 8 | implemented (thin wrapper of exp07, smoke ok on 2 patients) |
| 09 | `exp09_pantrack` | 9 | implemented (smoke ok, 3 patients; a `pid` lookup bug found and fixed); needs an exp04 run for the tuned baselines |

Each experiment owns one section below (arguments, literal full-run command, outputs). Edit only your own section; the
`<!-- end -->` lines keep neighbouring edits from colliding in git.

## Sections

### exp00a_data_audit

Checks every number the paper states about its data (300 patients, 4530 lesions, merge events, held-out 60, the 21 cohorts and the volume counts, prompt-encoding statistics, PanTrack) against the files, plus the `linking_unclear` reconciliation, the patients missing from the graph cache and the known-bad patient. Mismatches are reported, never fixed. CPU only.

| flag | default | meaning |
|---|---|---|
| `--data-root` | `/nnunet_data/Longitudinal-CT` | Longitudinal-CT root (`meta/`, `inputsTr*`, `data_split.json`, `derivatives/`) |
| `--holdout-csv` | `<data-root>/test_patients.csv` | the held-out 60 |
| `--corpus-dir` | `.../NanoUNet_preprocessed/Dataset900_Merged` | cohorts, splits, `*_centroids.json` sidecars of the merged corpus |
| `--raw-dir` | `/nnunet_data/NanoUNet_raw` | only listed, to count the cohort folders |
| `--pantrack-dir` | `/nnunet_data/raw/PanTrack` | PanTrack JSON files and labels |
| `--graph-split` | `lesionglue/configs/split.json` | split the graph caches were built from |
| `--graph-cache-dir` | `.../cache_v9_merge/processed` | only the tiny `*_meta.pt` files are opened |
| `--graph-cache-tag` | `v8_native` | cache tag in the file names |
| `--workers` | 8 | threads for sidecar reads and PanTrack label scans |

Full run: `python -m experiments.exp00a_data_audit.run --tag paper_v1` (about 2 min). Outputs: `claims` (`{claim, paper_value, measured_value, match, source}`), `unclear_reconciliation`, `graph_cache`, `graph_missing`, `empty_sets`, `no_cog_propagated`, `special_patients`, `per_patient`, `merge_events`, `cohorts`, `prompt_variants`, `pantrack_scans`, `pantrack_pairs`; `table.md` lists the claims as OK/MISMATCH.

<!-- end -->

### exp00b_calibration

Verifies and documents the shipped registration-error table (never rewrites it): recomputes the residuals of both backends, compares `n_per_bin` and every offset triple with the table, and counts the rows that come from the held-out 60. CPU only.

| flag | default | meaning |
|---|---|---|
| `--data-root` | `/nnunet_data/Longitudinal-CT` | Longitudinal-CT root (`meta/`, `inputsTrFU/`, `derivatives/`) |
| `--table` | `<data-root>/derivatives/registration_error_table.json` | table to verify |
| `--holdout-csv` | `<data-root>/test_patients.csv` | the held-out 60 |
| `--segmenter-plan` | `.../Dataset900_Merged/nnUNetResEncUNetLPlans_h200_smallpv.json` | only for the frame check (spacing) |
| `--workers` | 16 | threads for the FU header reads |

Full run: `python -m experiments.exp00b_calibration.run --tag paper_v1` (about 15 s). Outputs: `reproduction` (per backend and size bin), `per_bin` (median/p90/p95/max mm, per-axis SD), `overall`, `leakage`, `frame_check`, `per_lesion`.

<!-- end -->

### exp00c_seg_eval_manifest

Pins the single-timepoint evaluation set for exp01/exp02 in `experiments/exp00c_seg_eval_manifest/seg_eval_v1.json` (schema `seg-eval-manifest/1`): seen-cohort (held-out 60 + validation cases), outside (capped, seeded, patient-disjoint subsets of the clean PancancerCTSeg sources, header-checked against our training volumes) and healthy scans. The source matrix is the data `SOURCES` in `run.py`. CPU only.

| flag | default | meaning |
|---|---|---|
| `--cap` | 30 | patients per outside source (seeded, one scan per patient) |
| `--val-per-cohort` | 5 | validation cases per training cohort in the seen-cohort tier (0 = none) |
| `--include-luna25` | off | also draw from LUNA25 (pseudo-label masks, detection only) |
| `--manifest-out` | `experiments/exp00c_seg_eval_manifest/seg_eval_v1.json` | where a full run writes the manifest (a smoke run writes into its run dir) |
| `--header-cache` | `$NANOUNET_TMPDIR/exp00c_headers.json` | local scratch cache of NIfTI header reads |
| `--corpus-dir` | `.../NanoUNet_preprocessed/Dataset900_Merged` | our training volumes (`dataset.json`, `splits_final.json`, `gt_segmentations/`) |
| `--workers` | 16 | threads for header reads |

Full run: `python -m experiments.exp00c_seg_eval_manifest.run --tag paper_v1` (a few minutes; the first run fills the header cache). Outputs: the manifest, plus `sources`, `cases`, `dropped`, `coverage`, `candidates_per_source` in `results.json`.

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

Given the annotated lesions of both scans (nothing to miss), which pairs are the same lesion? This is the ceiling of the identity task: every later experiment's drop (segmenter-supplied lesions, other datasets) is measured from here.

| flag | default | meaning |
|---|---|---|
| `--data-root` | /nnunet_data/Longitudinal-CT | Longitudinal-CT layout root (meta/, targetsTrBL/FU/) |
| `--cache` | required | graph cache root built with the fixed builder (must hold train, val and test caches of the current CACHE_TAG) |
| `--config` | lesionglue/configs/complete.json | lesionglue training config (fixed max_steps recipe) |
| `--max-steps` | -1 | override the config's max_steps; -1 = use the config (smoke runs use a few steps) |
| `--parallel-folds` | 1 | trainings run at the same time (one fits the GPU well; five saturate it) |
| `--skip-training` | off | only score folds whose last.ckpt already exists in this run (with --resume) |

```bash
python -m experiments.exp03_matcher_alone.run --tag paper_v1 --cache /nnunet_data/lesion_tracking/cache_v9_merge
```

Runtime: 5 trainings of 7400 steps (about 30 min each alone on one A100; `--parallel-folds` runs several at once) plus seconds of scoring.

Depends on: experiments.common, experiments.scoring, folds.py (this folder), the lesionglue console entry `lesionglue.cli.train`.

Caveats: Numbers are only valid on a cache built with the fixed graph builder (merge-target nodes; plan Sec. 2) — the tag in the cache name must be the current lesionglue CACHE_TAG. The with-unclear sensitivity row is not produced by this version. Training uses the owner's recipe seed unless --seed is given; the fold split seed is fixed at 0.

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

Does the system generalise to unseen patients of the same centre? Masks (Dice, NSD, detection) and lesion identity (recall per class, edge F1) for our pipeline against the prompted longitudinal segmenter of Kirchhoff et al. (LongiSeg), both driven by the same propagated points on the same scan pairs. exp08 runs the very same protocol on a new centre (site tag only).

| flag | default | meaning |
|---|---|---|
| `--data-root` | /nnunet_data/Longitudinal-CT | dataset root in the Longitudinal-CT layout (inputsTrBL/FU, targetsTrBL/FU, meta) |
| `--patients-csv` | none | CSV with a `patient` column (default: <data root>/test_patients.csv) |
| `--patients` | none | explicit patient ids instead of --patients-csv (debugging; ids then appear in command.txt) |
| `--methods` | ['ours', 'kirchhoff'] | systems to run: ours (nanoUNet + LesionGlue) and/or kirchhoff (LongiSeg) |
| `--seg-ckpt` | /nnunet_data/NanoUNet_results/nanounet/Dataset900_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_final_ft250_fromlast/finetune/bestsel-epoch=153-val_prompt_score=0.7261.ckpt | our segmenter checkpoint <model dir>/finetune/<file>.ckpt (EMA weights; plans.json etc. are read from <model dir>) |
| `--matcher-ckpt` | /nnunet_data/lesion_tracking/runs/final_v9_noval/seed0/last.ckpt | our matcher checkpoint (EMA weights) |
| `--longiseg-model` | /nnunet_data/LongiSeg/_model | LongiSeg model folder (plans.json, dataset.json, fold_0..fold_4); research-only weights |
| `--anonymize`, `--no-anonymize` | True | replace patient ids by a salted hash in every output (files, tables, log); coordinates are never stored |
| `--salt` | none | salt of the patient hash (default: random, kept in artifacts/salt.txt so --resume and --rescore reproduce the ids) |
| `--validate-only` | off | check layout, models and environment, report every problem with its Fix, run nothing |
| `--tau` | none | decoder cut-off; default: the matcher checkpoint's own dust_tau (only change it to rescore a sensitivity row) |

```bash
python -m experiments.exp07_internal_set.run --tag paper_v1
```

Runtime: About 1 min per patient for ours (segmentation dominates) plus about 2.2 s per lesion and ~15 s model loading for Kirchhoff: one to two hours for 60 patients on one A100. Resumable (--resume skips finished units and retries failed ones); --rescore takes a minute.

Depends on: experiments/common.py, scoring.py, segment.py (via pipeline.py), pipeline.py; kirchhoff.py (this folder); the LongiSeg source and model; the matcher and segmenter checkpoints. exp08 and exp09 import this folder.

Caveats: NUMBERS ARE NOT MEANINGFUL until the matcher is retrained on the fixed graph cache (experiments plan Sec. 2) and `MATCHER_FINAL` is repointed: the current checkpoint never saw a merge-target node, merge recall is 0 by construction. A patient either method cannot process stays as status failed and counts as fully missed in identity (its mask lesions are unknown and absent from the mask table). The LongiSeg model is research-only (Longitudinal-CT, KiTS23, LiTS, PanTS licences). `predict_case` needs a CUDA device. Kirchhoff's `new` class is null by construction, its macro recall averages the three defined classes.

<!-- end -->

### exp08_external_set

Does the system generalise to a new centre it has never seen? The same masks (Dice, NSD, detection) and identity (recall per class, edge F1) protocol as exp07, our pipeline against Kirchhoff et al. (LongiSeg), on scan pairs from a partner lab.

| flag | default | meaning |
|---|---|---|
| `--data-root` | /nnunet_data/Longitudinal-CT | dataset root in the Longitudinal-CT layout (inputsTrBL/FU, targetsTrBL/FU, meta) |
| `--patients-csv` | none | CSV with a `patient` column (default: <data root>/test_patients.csv) |
| `--patients` | none | explicit patient ids instead of --patients-csv (debugging; ids then appear in command.txt) |
| `--methods` | ['ours', 'kirchhoff'] | systems to run: ours (nanoUNet + LesionGlue) and/or kirchhoff (LongiSeg) |
| `--seg-ckpt` | /nnunet_data/NanoUNet_results/nanounet/Dataset900_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_final_ft250_fromlast/finetune/bestsel-epoch=153-val_prompt_score=0.7261.ckpt | our segmenter checkpoint <model dir>/finetune/<file>.ckpt (EMA weights; plans.json etc. are read from <model dir>) |
| `--matcher-ckpt` | /nnunet_data/lesion_tracking/runs/final_v9_noval/seed0/last.ckpt | our matcher checkpoint (EMA weights) |
| `--longiseg-model` | /nnunet_data/LongiSeg/_model | LongiSeg model folder (plans.json, dataset.json, fold_0..fold_4); research-only weights |
| `--anonymize`, `--no-anonymize` | True | replace patient ids by a salted hash in every output (files, tables, log); coordinates are never stored |
| `--salt` | none | salt of the patient hash (default: random, kept in artifacts/salt.txt so --resume and --rescore reproduce the ids) |
| `--validate-only` | off | check layout, models and environment, report every problem with its Fix, run nothing |
| `--tau` | none | decoder cut-off; default: the matcher checkpoint's own dust_tau (only change it to rescore a sensitivity row) |

```bash
python -m experiments.exp08_external_set.run --data-root /data/site_b --patients-csv /data/site_b/test_patients.csv --tag site_b
```

Runtime: Same as exp07: about 1-2 h for 60 patients on one A100; resumable; --rescore takes a minute.

Depends on: experiments/exp07_internal_set/run.py (the protocol), kirchhoff.py (same folder), and everything exp07 depends on.

Caveats: Same as exp07: numbers are not meaningful until the matcher is retrained on the fixed graph cache and `MATCHER_FINAL` is repointed; the LongiSeg weights are research-only. Nothing here is tuned on the external data.

<!-- end -->

### exp09_pantrack

Does the identity matcher transfer to a disease and a scan protocol it never saw? PanTrack is pancreatic cancer with hepatic metastases (portal-venous CT, one centre); our pipeline is scored at Lstar and Lhat beside Di Veroli, Qahqaie and (when available) Kirchhoff.

| flag | default | meaning |
|---|---|---|
| `--data-root` | /nnunet_data/raw/PanTrack | PanTrack root (images/, labels/, totalseg/, patients.json, tracking.json, organ_annotations.json) |
| `--patients` | none | explicit PanTrack patient ids (e.g. PanTrack_001) instead of all 45 (smoke runs) |
| `--settings` | ['A', 'C'] | ours: A = Lstar, C = Lhat (both scans segmented), B = annotated BL + segmented FU; none = baselines only |
| `--matcher-ckpt` | /nnunet_data/lesion_tracking/runs/final_v9_noval/seed0/last.ckpt | matcher checkpoint (EMA weights) |
| `--tau` | none | decoder cut-off; default: the matcher checkpoint's own dust_tau |
| `--baselines-run` | none | exp04 RUN_DIR whose chosen_params give the Di Veroli / Qahqaie hyperparameters (default: Di Veroli published d=1 p=0.10 r=7, no Qahqaie) |
| `--workers` | 3 | processes for the scan statistics and the baseline inputs (each holds two CTs in memory) |

```bash
python -m experiments.exp09_pantrack.run --tag paper_v1 --baselines-run /nnunet_data/experiments/exp04_baselines/<run_id>
```

Runtime: Validation and baselines are CPU (reading 161 labels + TotalSeg masks and 322 CT reads: about 10-15 min with 4 workers). Ours: a few minutes per pair and setting on one A100 (scans are up to 972 slices); 116 pairs with B and C take hours: resumable with --resume, rescore with --rescore.

Depends on: experiments/common.py, scoring.py, pipeline.py, segment.py, exp04_baselines/{diveroli,qahqaie}.py, pantrack.py; an exp04 run for the tuned baseline parameters (optional); exp07 kirchhoff.py (not yet).

Caveats: NUMBERS ARE NOT MEANINGFUL until MATCHER_FINAL is repointed to the model retrained on the fixed graph cache (graph-builder defects, plan Sec. 2). Liver annotations are partial by design: an unannotated detected lesion is a false positive for every method, depressing precision and the ceiling. The vocabulary has no pancreas, so pancreatic lesions enter as `Others`, a type the matcher saw rarely. Lesions new in FU cannot be prompted (no propagated point exists for them), so under B and C they are never found. Di Veroli's dilation is in voxels on an anisotropic grid (0.4 mm slices). Patient 3988c7f88e is a Longitudinal-CT case and does not occur here.

<!-- end -->

## Runbook

Order of the full runs (plan Sec. 10). Every run is tagged `paper_v1`; after each one, `git add experiments/results && git commit && git push` (the container is ephemeral). A failed
sanity check means investigate, not tune. GPU work runs one slot at a time on a 40 GB card (on a larger card two slots may overlap; keep at most 2 `nanounet_predict`-sized jobs
inside a slot and stagger their starts by about 20 s, the host RAM cap is what bites first).

```bash
cd /nanoUNet
export NANOUNET_RAW=/nnunet_data/NanoUNet_raw NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed NANOUNET_RESULTS=/nnunet_data/NanoUNet_results NANOUNET_TMPDIR=/tmp/nanounet_tmp
```

| slot | what | command | est. | sanity check |
|---|---|---|---|---|
| 0 | exp00a, b, c (CPU, done once; rerun exp00a after the v9 cache) | `python -m experiments.exp00a_data_audit.run --tag paper_v1` (also `exp00b_calibration`, `exp00c_seg_eval_manifest`) | minutes | every `claims` row `match: true` or explained in `notes` |
| 0b | rebuild the graph cache, retrain the final matcher, repoint `MATCHER_FINAL` in `experiments/common.py` | `lesionglue_preprocess --split all --root /nnunet_data/Longitudinal-CT --cache /nnunet_data/lesion_tracking/cache_v9_merge --prop-fill unigradicon --jobs 16`, then `lesionglue_train --config lesionglue/configs/complete.json --root /nnunet_data/Longitudinal-CT --cache /nnunet_data/lesion_tracking/cache_v9_merge --out /nnunet_data/lesion_tracking/runs/final_v9_noval/seed0 --seed 0` | ~10 min + ~30 min | training prints `fit=250 val=0` |
| 1 | exp03 (5 fold trainings + scoring) | `python -m experiments.exp03_matcher_alone.run --tag paper_v1 --cache /nnunet_data/lesion_tracking/cache_v9_merge` | 5 x ~30 min (`--parallel-folds` to overlap) | every patient in exactly one fold; four class recalls and edge F1 present; missed-patient count printed |
| 2 | exp04 and exp06 (CPU, may overlap slot 3) | `python -m experiments.exp04_baselines.run --tag paper_v1 --ours-run /nnunet_data/experiments/exp03_matcher_alone/<run_id>`; `python -m experiments.exp06_limits.run --tag paper_v1 --from-run /nnunet_data/experiments/exp03_matcher_alone/<run_id>` | ~1 h; seconds | `expressible_classes` per method; chosen hyperparameters stored per fold, none tuned on its own fold |
| 3 | exp05 (settings A, B, C), then exp07 on the held-out 60 as validation | `python -m experiments.exp05_full_pipeline.run --tag paper_v1`; `python -m experiments.exp07_internal_set.run --tag paper_v1` | ~2 h; ~1-2 h | setting A identity ceiling 1.0 in every class; patient `3988c7f88e` present as `status` missed; exp07 "ours" equals exp05 setting B (same numbers) |
| 4 | exp01 (nanoUNet, then the external systems from the owner's prediction folders), exp02 | commands in the exp01 and exp02 sections | 1-2 h; 2-4 h | exp02 at s=0 equals exp01 S1 on the same clicks |
| 5 | exp09 | `python -m experiments.exp09_pantrack.run --tag paper_v1 --baselines-run /nnunet_data/experiments/exp04_baselines/<run_id>` | hours (resumable) | validation table has no errors; Kirchhoff column empty by design |
| 6 | exp07 / exp08 at the partner labs | see the container contract below | minutes per case | outside this machine |

<!-- end -->

## Container contract (private-set experiments 07 and 08)

What the owner's container must provide so that `python -m experiments.exp07_internal_set.run` (new centre: `exp08_external_set`) runs with no network and nothing else installed:

- **Python environment:** this repo (`pip install -e . --no-deps`) with its normal dependencies, plus `difference_weighting` 0.1.0 (`pip3 install --no-deps git+https://github.com/MIC-DKFZ/Longitudinal-Difference-Weighting.git`) and the LongiSeg source on `PYTHONPATH` (default `/nnunet_data/LongiSeg`). LongiSeg runs in the same process and the same environment (checked: it reproduces its own environment's prediction to 1 voxel of 22,764 on torch 2.7.1). One environment, no second venv.
- **Models (read-only):** matcher `--matcher-ckpt` (default `common.MATCHER_FINAL`), segmenter `--seg-ckpt` (default `common.SEG_CKPT`, EMA), LongiSeg `--longiseg-model`. All paths are flags; nothing is written outside `--out-root`.
- **Data:** a folder in the Longitudinal-CT layout (`inputsTrBL|FU/<pid>_<idx>.nii.gz` + `.json`, `targetsTrBL|FU/`, `meta/<pid>.csv`) and a patient list CSV with a `patient` column (`--data-root`, `--patients-csv`). The FU points in `inputsTrFU/*.json` must be PROPAGATED points made with the propagation the public dataset shipped with; the code never registers.
- **GPU:** one CUDA device (LongiSeg `predict_case` needs CUDA); about 1-2 h for 60 patients on an A100-class card.
- **Privacy:** `--anonymize` (default on) replaces patient ids by a salted hash in every file, table and log line and keeps no coordinates; the id map and the salt stay in `artifacts/` at the lab. Only `results.json` / `table.md` come back.
- **First command in the container:** `python -m experiments.exp07_internal_set.run --validate-only --data-root ... --patients-csv ...` checks layout, models and environment and lists every problem with a `Fix:` line.
- **Licence:** the LongiSeg weights are research-only (Longitudinal-CT, KiTS23, LiTS, PanTS licences).
- **Validation before it travels:** run exp07 on the held-out 60 (the default) and check that "ours" equals exp05 setting B.

<!-- end -->
