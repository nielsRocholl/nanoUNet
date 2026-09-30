# Monorepo: nanoUNet as the home of every CT-lesion project

Date: 2026-09-29
Status: done on branch `monorepo`; checker 0 errors, 55 warns; open items below

## Decisions

| question | decision |
|---|---|
| shape | one repo, one env, one top-level package per project; no uv workspace, no per-project pyproject |
| tracking name | `lesionglue` (the method is LesionGlue); console scripts `lesionglue_*` |
| where docs/configs/scripts live | inside the project folder, so a paper links one folder |
| shared code | `core/` (terminal UI only for now); imports no project |
| lesion-tracking history | kept: `git filter-repo --filename-callback` (tracking/ -> lesionglue/, rest -> lesionglue/), merged with `--allow-unrelated-histories` |
| seg + track composition | its own project `segtrack/` (deps nanounet + lesionglue), console `segtrack_run` |
| coupling rule | new R21: declared one-way deps in `PROJECTS` (check.py), enforced on every import incl. lazy ones |

## Commits

| commit | kind | what |
|---|---|---|
| merge | S | lesion-tracking `fa78009` history under `lesionglue/` |
| adopt | S+C | `tracking.*` -> `lesionglue.*`, `lesion_track*` -> `lesionglue_*`, path prefixes, pyproject extra, drop `.cursor/ lib/ runs/ tests/` (R16) |
| R20 | S | `model/ eval/ data/{source,features,graph,cache,instances}/`, `match_utils` -> `objective` |
| nanounet folder | S+C | `docs/ configs/ scripts/ README.md` -> `nanounet/` |
| segtrack | S+C | out of `nanounet/`, plus `e2e_{eval,track}.py` from lesionglue |
| core | S | UI helpers byte-identical to `core/ui.py`; lesionglue keeps its rank-0 gate as wrappers |
| skill | C | R21, multi-project checker, layout docs, root README |
| `08b1ee3` | C | R11: 9 bare prints -> `cprint` (stderr); text unchanged, `markup=False`; no caller reads their stdout |
| `620d72f` | S | R1: validation scoring -> `lesionglue/train/val_score.py`, bound as class attrs (no mixin); module.py 244 -> 174 |
| `00869fb` | L | U1: `lesionglue_track` progress bar on the stderr console |
| `4b02001` | S | R21: `_agg`/`_node_rows`/`_positive_matrix` -> public `aggregate`/`node_rows`/`positive_matrix` |
| `16fd3f1` `2bf48de` | L | R13: 7 lesionglue CLIs get `main()`; no work at import |
| `f1e44bd` | C | U8: 114 `help=` strings (lesionglue CLIs + segtrack scripts) |
| `923da96` | S | R20: `cli/qc_view.py` -> `eval/qc_view.py` |
| `2594227` | L | 8 more console scripts; all 13 `lesionglue/cli/*` are `lesionglue_*` |
| `c8fcf00` `ef7026c` | C+L | K7: header + `config_table(core.ui.arg_rows(...))` + `next:` on 14 CLIs; K7 accepts `nano_banner` |
| `929b024` | C | D1-D3: `lesionglue/docs/index.md` + `steps/{data,train,eval,track,qc}.md`, 91/91 flags |
| `df6a055` | C | stale `--decode` help, `--meta` in a Fix line, README match-CSV columns |

## Gates

| gate | result |
|---|---|
| lesionglue golden vs lesion-tracking `fa78009` (63 graph tensors, forward + 4 loss terms, track CSVs hungarian + sinkhorn, seeded random ckpt) | 81/81 equal after each commit |
| lesionglue AST guard | 186/186 defs present; diffs = import paths + path text |
| nanounet AST guard vs `de9614f` (into core+nanounet+segtrack) | 472/472 present; diffs = import paths + path text |
| nanounet equivalence harness vs `de9614f` (stages A-H, `--allow-text`) | golden 486/486 equal; AST guard 513/513 defs ok; surface 21 keys equal except `help/build_valset`, `help/segtrack` (path text, prog name `segtrack_run`) and `rows/segtrack` (ERROR vs ERROR on empty probe dir; equal with a paired-case probe); harness deleted (R16) |
| R1 split (`620d72f`) | AST 16/16; lesionglue golden 81/81; validation-hook state (2 epochs) identical HEAD vs tree |
| checker | 0 errors; warns 189 -> 55 after the backlog pass |
| backlog pass (`00869fb`..`df6a055`) | lesionglue golden 81/81 after each code commit; `--help` byte-identical for S/L commits; AST equal once inserted calls/help kwargs are stripped; audit/split/pool outputs identical on real or synthetic inputs |
| imports | every module imports without side effects (`audit.py` fixed in `16fd3f1`) |

## Not done (lesionglue style debt, pre-existing)

- ~~R1 `train/module.py` 244 LOC~~ (`620d72f`); ~~R11 bare prints in `cli/pool.py`, `cli/baseline_distance.py`, `baselines/nearest_mask/run.py`, `segtrack/scripts/e2e_eval.py`~~ (`08b1ee3`).
- ~~U8 flags without `help=`~~ (`f1e44bd`); ~~K7 lesionglue~~ (`ef7026c`); ~~D3 lesionglue step docs~~ (`929b024`).
- ~~`cli/audit.py`, `cli/report.py` run at import~~ (`16fd3f1`); ~~`cli/track.py` progress on stdout~~ (`00869fb`).
- ~~Only 5 of 14 lesionglue CLIs are console scripts~~ (`2594227`).
- ~~Cross-project private imports in `segtrack/scripts/e2e_eval.py`~~ (`4b02001`).

## Still open (55 warns + findings)

- E1: 17 lesionglue + 7 nanounet raises without a `Fix:` line (text-only C pass).
- G2: `lesionglue/model/matcher.py:103-106` `.tolist()` in `_dust_graph`; `nanounet/model/loss/cc_dice_ce.py` syncs. Needs a before/after throughput number (G4) on GPU.
- nanounet CLIs: K6 x3, K7 x7, snake_case `--dataset_id`/`--num_processes` (kept for cluster scripts).
- `lesionglue/scripts/round9.sh` tau sweep greps `val_match_score:` from `eval.py` stdout; eval writes to stderr and prints no such key (broken already in lesion-tracking `fa78009`).
- `lesionglue/README.md` (291 lines) now overlaps `docs/steps/`; slim it to a paper-facing overview when convenient. Broken link `blueprint.md` (pre-existing).
