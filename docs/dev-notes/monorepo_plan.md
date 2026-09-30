# Monorepo: nanoUNet as the home of every CT-lesion project

Date: 2026-09-29
Status: done on branch `monorepo`; checker 0 errors, 0 warns (G2 closed on A100, 2026-09-30)

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

## Closed in the second pass

| commit | kind | what |
|---|---|---|
| `01b55f1` + this commit | L | `round9.sh`: tau sweep reads `selected.dust_tau` from eval's `--out` JSON (was always skipped since `fa78009`); default `CFG` -> `lesionglue/configs/base.json` |
| `835e9f8` | C | nanounet E1 Fix lines / reasoned waivers, E4 waivers (producer thread re-raises in `__iter__`; reader probe raises with Fix) |
| lesionglue E1 commit | C | Expected + Fix on 19 lesionglue errors, 2 invariant waivers, R3 waiver on a frozen record; golden 81/81 |
| README commit | C | lesionglue README 293 -> 101 lines; `docs/reference/{config,layout,experiments}.md`; 42-fact inventory, no loss |
| nanounet K7 commit | L | K6 guards, config table + `next:` on all 7 nanounet commands; legacy snake flags waived (U8) |

## GPU smoke 2026-09-30 (`dlc-arceus`, A100-SXM4-40GB)

Host checkout `/nanoUNet`, not the Slurm container (no `srun`/`apptainer` on this node). Python 3.11.3. cgroup `memory.max` 64 GiB. `lesionglue_eval` default cache `/nnunet_data/lesion_tracking/cache` has no `v7_native`; deployed graphs are `/nnunet_data/lesion_tracking/cache_v7`. Default eval started writing a new cache there; that staging dir was removed. Smoke eval/train used `--cache` `cache_v7`.

| # | status | key number | wall |
|---|---|---|---|
| A1 | pass | test match **0.970133**, 57 graphs, EMA, dust_tau 0.125 | 10.9 s |
| A2 | pass | `rows` + `selected.dust_tau` **0.20** (val, 46 graphs) | 14.6 s |
| A3 | fail | 200 steps, then SWA saw 1 plateau update (needs 5). No ckpt, no `val_match_score` on stderr. `val_check_steps` is 250. Before `b8814fd`, import died on 3.12 f-string quotes | 66.6 s |
| A4 | blocked | no A3 checkpoint, so no `next:` oof/pool | — |
| A5 | pass | 57 CSVs, columns `bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id`, 3 skips. First process SIGKILL (−9) at 2833 s under the 64 GiB cgroup (46 CSVs); remainder 982 s, exit 0 | 3815 s |
| A6 | pass | header, config table, `next:`, `epoch_wall_time_sec` e0 15.02 / e1 10.97. Batch 2 and `--dl-bucket s` (script is H200 batch 12 / xl). No val manifest | 113 s |
| A7 | fail | `last.ckpt` stem is 3 input channels; `N_PROMPT_CHANNELS=1` builds 2 (negative prompt channel removed) | 30.3 s |

## G2 (closed)

`_dust_graph` `.tolist()`: host counts matched GPU `bincount` on one 8-graph batch. Step time `(median t600 − median t100) / 500`, 3 repeats, fold 0.

| arm | 100 s walls | 600 s walls | step_s | GPU sm mean | Δ |
|---|---|---|---|---|---|
| before | 40.46 / 41.20 / 42.65 | 165.75 / 172.16 / 172.24 | 0.2619 | 20% | |
| CPU sizes | 36.52 / 36.85 / 39.04 | 164.12 / 164.39 / 165.11 | 0.2551 | 23% | −2.6% |

The before 600 s spread (6.5 s) exceeds the median shift. Waived in `e3a7ff3`.

`cc_dc_ce` vs `dc_ce`, Dataset900, batch 2, 4 iters/epoch, median `epoch_wall_time_sec` epochs 1–3 (epoch 0 dropped):

| arm | e1 | e2 | e3 | median | Δ |
|---|---|---|---|---|---|
| dc_ce | 5.676 | 4.317 | 4.168 | 4.317 | |
| cc_dc_ce | 9.800 | 7.138 | 8.650 | 8.650 | +100% |

CPU CC labelling stays. Waived in `d8d10d3`. `--loss` help states this cost.

## Still open

- A3/A4: a 200-step fold train cannot finish; `on_train_end` requires 5 SWA plateau updates and validation is every 250 steps.
- A7: Dataset999 `h200_instance_1200ep` checkpoint does not load (3 vs 2 input channels).
