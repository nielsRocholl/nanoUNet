# Monorepo: nanoUNet as the home of every CT-lesion project

Date: 2026-09-29
Status: done on branch `monorepo`; lesionglue style debt tracked below

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

## Gates

| gate | result |
|---|---|
| lesionglue golden vs lesion-tracking `fa78009` (63 graph tensors, forward + 4 loss terms, track CSVs hungarian + sinkhorn, seeded random ckpt) | 81/81 equal after each commit |
| lesionglue AST guard | 186/186 defs present; diffs = import paths + path text |
| nanounet AST guard vs `de9614f` (into core+nanounet+segtrack) | 472/472 present; diffs = import paths + path text |
| nanounet equivalence harness | see final report |
| imports | 168 modules import except `lesionglue/cli/audit.py` (parses argv at import; pre-existing) |

## Not done (lesionglue style debt, pre-existing)

- R1 `train/module.py` 244 LOC; R11 bare prints in `cli/pool.py`, `cli/baseline_distance.py`, `baselines/nearest_mask/run.py`, `segtrack/scripts/e2e_eval.py`.
- U8: ~100 flags without `help=`; K6/K7 missing guards/headers; D3 no `docs/steps/` for lesionglue CLIs.
- `cli/audit.py`, `cli/report.py` run at import (R13). `cli/track.py` progress bar writes to stdout (U1).
- Only 5 of 14 lesionglue CLIs are console scripts; the rest run as `python lesionglue/cli/<cmd>.py`.
- Cross-project private imports in `segtrack/scripts/e2e_eval.py` (`_node_rows`, `_positive_matrix`, `_agg`).
