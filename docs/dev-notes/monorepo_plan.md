# Monorepo: nanoUNet as the home of every CT-lesion project

Date: 2026-09-29
Status: done on branch `monorepo`; checker 0 errors; lesionglue warn debt tracked below

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

## Gates

| gate | result |
|---|---|
| lesionglue golden vs lesion-tracking `fa78009` (63 graph tensors, forward + 4 loss terms, track CSVs hungarian + sinkhorn, seeded random ckpt) | 81/81 equal after each commit |
| lesionglue AST guard | 186/186 defs present; diffs = import paths + path text |
| nanounet AST guard vs `de9614f` (into core+nanounet+segtrack) | 472/472 present; diffs = import paths + path text |
| nanounet equivalence harness vs `de9614f` (stages A-H, `--allow-text`) | golden 486/486 equal; AST guard 513/513 defs ok; surface 21 keys equal except `help/build_valset`, `help/segtrack` (path text, prog name `segtrack_run`) and `rows/segtrack` (ERROR vs ERROR on empty probe dir; equal with a paired-case probe); harness deleted (R16) |
| R1 split (`620d72f`) | AST 16/16; lesionglue golden 81/81; validation-hook state (2 epochs) identical HEAD vs tree |
| checker | 0 errors, 189 warns (all pre-existing lesionglue debt) |
| imports | 168 modules import except `lesionglue/cli/audit.py` (parses argv at import; pre-existing) |

## Not done (lesionglue style debt, pre-existing)

- ~~R1 `train/module.py` 244 LOC~~ (`620d72f`); ~~R11 bare prints in `cli/pool.py`, `cli/baseline_distance.py`, `baselines/nearest_mask/run.py`, `segtrack/scripts/e2e_eval.py`~~ (`08b1ee3`).
- U8: ~100 flags without `help=`; K6/K7 missing guards/headers; D3 no `docs/steps/` for lesionglue CLIs.
- `cli/audit.py`, `cli/report.py` run at import (R13). `cli/track.py` progress bar writes to stdout (U1).
- Only 5 of 14 lesionglue CLIs are console scripts; the rest run as `python lesionglue/cli/<cmd>.py`.
- Cross-project private imports in `segtrack/scripts/e2e_eval.py` (`_node_rows`, `_positive_matrix`, `_agg`).
