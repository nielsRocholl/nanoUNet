# nanochat-style pure-refactor plan

Date: 2026-09-23
Status: implemented (2026-09-24, branch nanochat-refactor)
Scope: whole repo. The audit below is the original plan; outcomes are in the last column of §4 and the notes under §6 and §9.
Inputs: nanochat `92d63d4` (2026-07-03, same commit as `references/nanochat.md`), `check.py --json`, 15 Sonnet audit agents
(4 nanochat readers + 11 nanoUNet readers: cli, data, prompt, plan, model, train, pretrain, infer, diag, root modules, docs).

Classes: **S** = move/rename/inline/split with byte-identical function bodies · **C** = docstring/comment/help=/error text/docs ·
**L** = anything that changes what executes (listed in §7, never mixed into S/C PRs) · **X** = delete confirmed-dead code
(zero execution impact, but deletion is not on the allowed list, so each one needs your OK).

## 0. Preconditions (before PR-0)

| # | issue | evidence | action |
|---|---|---|---|
| P1 | Baseline isn't a commit: 87 uncommitted paths (longi/register removal, `longi_row.py → inference_row.py`) | `git status` | commit the WIP first. The golden harness pins a SHA |
| P2 | Stale editable install: `.venv/bin` has `nanounet_longi_*`, `nanounet_register_longi` and lacks `nanounet_build_splits`, `nanounet_build_valset`, `nanounet_segtrack` | docs agent | `uv pip install -e .` before capture |
| P3 | One audit agent ran `graphify update .` (regenerated the already-dirty `graphify-out/*` from the same tree; no source touched) | prompt agent | none, or `git checkout graphify-out` if unwanted |

## 1. Phase 1: nanochat vs `references/nanochat.md` (new or contradicting only)

| # | digest claim | verdict | evidence |
|---|---|---|---|
| 1 | eval/bench scripts use `main()` + `parser.error` (`base_eval.py:128,143`) | line off by one; partly false | `parser.error` is at `base_eval.py:144`. `chat_eval.py:178` is guard-only with no `main()`. 6 of 9 scripts are fully flat with no guard |
| 2 | flags grouped by `#` headers, each has `help=` | partial | headers only in `base_train.py:42-79`, `chat_sft.py:36-66`, `chat_rl.py:33-59`. `infer_bench.py:105` `-t/--temperature` has no `help=`. Kebab-case is 56/56 |
| 3 | "pretty table first, then one JSON line" (`infer_bench.py:19-23,164`) | partial | table is hand-aligned f-strings (`infer_bench.py:202-226`). JSON at `:241` is printed **unconditionally** (there is no `--json` flag) |
| 4 | shell sweeps grep status lines (`miniseries.sh:85-92`) | **contract already broken** | `miniseries.sh:85` greps `"Number of parameters:"`, which `base_train.py` never prints (`:253` prints `"Parameter counts:"`). `scaling_laws.sh:101-106` is OK |
| 5 | one stable status line (`base_train.py:567`) | two shapes | `chat_sft.py:469` logs `mfu` and `base_train.py:567` logs `bf16_mfu` |
| 6 | `Task` has 6 subclasses | 7 | incl. `TaskMixture` `tasks/common.py:129` and `TaskSequence` `:164` |
| 7 | asserts carry actual-vs-expected f-strings | overgeneralized | in engine/core_eval/tasks only 2 of ~34 do (`mmlu.py:52`, `arc.py:45`). Most messages are static (`engine.py:128,143`) |
| 8 | 41 asserts vs 11 raises | 40 / 11 | `nanochat/*.py` count |
| 9 | ckpt save `checkpoint_manager.py:22-56` | `:22-58` | `save_checkpoint` ends at 58 |
| 10 | `dev/LOG.md` has 29 entries with hypothesis/numbers/verdict | 34 entries; free prose | only `:1077` uses `**Hypothesis:**`. Every entry ends in a verdict. Suffixes `(negative)`/`(Reverted)` |
| 11 | `tests/` has 5 files | 6 | + `test_attention_fallback.py` |
| 12 | NEW: section-banner comments inside big files | contradicts **R6** | `optim.py:17,65,182`, `tokenizer.py:28,261`. nanochat uses banners where we split files |
| 13 | NEW: type hints only on compiled kernels and small utils | R7 is stricter | 0/25 defs hinted in `gpt.py`, 0/19 in `tokenizer.py`; `optim.py:24-34` hinted |
| 14 | NEW: import-time side effects are load-bearing | same class of coupling as our K1 | `common.py:32,68-69` (COMPUTE_DTYPE, logging); `flash_attention.py:23-50` downloads a kernel at import under a broad `except` (`:45`) |
| 15 | NEW: 3 output mechanisms | R11/U1 deviation | `print0`, bare `print` (`dataset.py`, `tokenizer.py:138`), `logger.info` (`checkpoint_manager.py`) |
| 16 | NEW: determinism split | harness template | global `torch.manual_seed(42)` for init (`common.py:184-190`) plus a local `torch.Generator` per sampling call (`engine.py:182-183`, `gpt.py:536-539`). `use_deterministic_algorithms` is commented out with a reason (`common.py:189-190`) |
| 17 | NEW: equivalence tiers in tests | harness template | exact `torch.equal` for determinism (`test_optim.py:64-77`, `test_engine.py:201-223`). `allclose` only across implementations (`test_attention_fallback.py:39-45` atol 1e-2; grads 0.05 at `:241-248`). Seeds via an explicit Generator |
| 18 | NEW: deletions logged with numbers | precedent for §6/§7 | `LOG.md:72-105` (autocast removal lists the touched files), `:1075-1089` (grad-clip deleted, ~2% overhead) |
| 19 | NEW: reuse via cross-script imports, not a shared module | noted | `chat_sft.py:26` imports `scripts.chat_eval`; `base_train.py:36` imports `scripts.base_eval` |
| 20 | NEW: quirks kept for ckpt compat, with a comment | precedent for K4/K5/K6 | `gpt.py:58-59` "kept for checkpoint compatibility" |
| 21 | NEW: known nanochat debt (don't copy) | add to the inconsistencies list | duplicated dispatch `core_eval.py:184-194` and `:232-239`; `all_reduce` without barrier `loss_eval.py:54-58` vs `core_eval.py:256-259`; broad swallow `engine.py:41-44`; `input()` at `chat_cli.py:53`; user-facing asserts `base_train.py:339,409`, `chat_sft.py:123` |
| 22 | NEW: `-1` and `None` sentinels coexist | U8 note | `base_train.py:56..77` (`-1` = auto) vs `chat_sft.py:47-53` (`None` = inherit) |

## 2. Phase 2: audit summary

`check.py`: **3 errors, 224 warns, 93 files**. U8 130 (123 missing help, 7 snake_case), E1 69, E4 14, G2 4, D3 4, R2 2, D6 1.

| error | verdict |
|---|---|
| `diag/mem_diag.py` R1 216 LOC | real → S01 |
| `model/dice_helpers.py` R4 | real → S02 |
| `nanounet/docs/steps/track.md:68` D4 `--no-ema` | **false positive**: `cli/segtrack.py:53` uses `BooleanOptionalAction`, which generates `--no-ema` (→ K-1) |

No R3/R6/R11/R14 violations anywhere. Near the R1 cap: `valset.py` 198, `sampling.py` 197, `resampling.py` 197, `score.py` 197, `case_pp.py` 197,
`build_valset.py` 195, `data_module.py` 193, `lightning_module.py` 191, `segtrack.py` 189. The next feature in any of these needs a split plan, not +LOC.

## 3. Coupling register (blockers for "just a move")

| K | file:line | kind | constraint |
|---|---|---|---|
| K1 | `cli/train.py:5-9,29`, `cli/pretrain.py:5-9,26` | import order | `set_safe_tmpdir()` + `init_dataloader_ipc()` run before torch. `quiet_lightning_runtime()` runs before any `pytorch_lightning` import. `train/fit.py` has no guard of its own, so never import `train.fit`/`lightning_module`/`data_module`/`pretrain.module` from anything imported earlier. `cli/segtrack.py:62` calls it inside `main()` (asymmetric) |
| K2 | `runtime.py:59` | process-global mutation | `set_safe_tmpdir` rewrites `TMPDIR/TMP/TEMP` + `tempfile.tempdir`. Called 2× per process (module-level, then `main()` with `results_tmp=`) and in worker inits `train/patch_iterable.py:62-63`, `pretrain/dataset.py:34-36` |
| K3 | `dataloader_prefs.py:69` | process-global | `file_system` sharing strategy. Idempotent calls at `cli/train.py:9,44`, `train/data_module.py:134,167`, `pretrain/dataset.py:165`. Must run before the first DataLoader iter |
| K4 | `train/ema.py:33` | **ckpt key** | class name `EMACallback` = `callbacks["EMACallback"]` (PL `state_key` = `__qualname__`), hardcoded at `infer/predictor.py:14`. Moving the module is S; renaming the class is L |
| K5 | `train/lightning_module.py:58`, `pretrain/module.py:53` | **ckpt key** | attr `self.net` → `net.*` prefix read by `infer/predictor.py:18` and `model/mae_transfer.py:22,51`. A rename silently loads 0 tensors (`[MAE] loaded 0 encoder tensors`) |
| K6 | `lightning_module.py:49` (`save_hyperparameters()`, no ignore), `pretrain/module.py:43` | **ckpt hparams** | every ctor kwarg name is a `hyper_parameters` key. `num_epochs` is read by `lightning_ckpt.py:16` and `nanounet/scripts/slurm_final_900_h200.sh:290`. Only primitives are pickled; `load_from_checkpoint` is unused; infer uses the `torch.load` dict (`infer/predictor.py:49`) |
| K7 | `lightning_module.py:90` `save_config(asdict)` → `cli/predict.py:65`, `cli/segtrack.py:127` | **on-disk JSON** | `nano_config.json` = `config.py` dataclass **field names/nesting** (no schema version). Class names are free to change |
| K8 | `planner_resenc.py:54,106,112,172-177`, `planner.py:151`, `plans.py:16-22,165,171`, `model/network.py:21-24`, `data/io.py:92-98` | **on-disk plans.json** | `resample_data_or_seg_to_shape.__name__`, reader class `__name__` (`SimpleITKIO`), `conv_op`/`norm_op` `__module__.__name__` resolved via `pydoc.locate`/`importlib`. Renaming any of these is L |
| K9 | `prompt/centroids.py:95-100` | on-disk sidecar | keys `centroids_zyx`, `bboxes_zyx`, `seed_zyx`, `volume_vox` |
| K10 | `plan/prep/preprocess.py:128` `_worker`, `prep/fingerprint.py:73` `_analyze_case`, `prompt/centroids.py:123` `_write_centroids_for_case` | spawn pickling | resolved by module path in the child. Moves must be atomic, and the harness must run `-np ≥ 2` (the `-np 1` path bypasses the Pool) |
| K11 | `train/patch_iterable.py:61` `worker_init` (imported by `data_module.py:21`, `data/valset.py:34`); `pretrain/dataset.py:33` `_worker_init` (partial); `CohortSampler` attr `patch_iterable.py:96`; `Blosc2Folder`/`ValManifest` attrs `data/valset.py:125,131` | DataLoader pickling | Linux uses fork today (no `set_start_method`), macOS uses spawn. Keep the old import path as a re-export when moving |
| K12 | `data/sampling.py:112-197` (+ child rng `:47`), `patch_iterable.py:133,136`, `pretrain/dataset.py:72`, `data_module.py:147,178`, `fit.py:81-82`, `cli/build_valset.py:130-137`, `plan/splits.py:88` | **RNG order** | per-patch draw order: bbox → false-pos → instance target → per-variant child `rg2`. Worker seeds `base + wid*10007` (+777_777). Six named valset streams consumed in cohort×scenario order. `pretrain/masking.py:26` draws from the **unseeded global** torch RNG (no `seed_everything` anywhere). Prep uses local `RandomState(1234)` (`case_pp.py`, `fingerprint.py`) |
| K13 | `data/blosc2_dataset.py:83,112` `blosc2.set_nthreads(1)`; `model/network.py:65-75` `torch.set_num_threads` swap; `infer/tta.py:15-23` module cache globals; `prompt/encoding.py:16` `lru_cache`; `diag/mem_diag.py` `_MEM_DIAG` (inherited under fork only) | hidden globals | keep `max_cat`/`cat_status`/`predict_batch_with_tta` in one file. Never duplicate `_build_ball_strel`. Construction order of `Blosc2Folder` is part of the contract |
| K14 | env vars | contract | `NANOUNET_RAW/PREPROCESSED/RESULTS` (`common.py:92`), `NANOUNET_DEF_N_PROC` (`common.py:29`, dead), `NANOUNET_TMPDIR`, `NANOUNET_ALLOW_ROOT_CGROUP`, `TMPDIR`, `HOME`, `SLURM_JOB_ID` (`runtime.py`), `NANOUNET_DL_KEEP_WORKERS`, `NANOUNET_MAE_KEEP_WORKERS`, `NANOUNET_DL_FORCE_NO_WORKERS`, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` (`dataloader_prefs.py:27,33,60`), `NANOUNET_MEM_DIAG`, `NANOUNET_MEM_LOG_EVERY` (`mem_diag.py:23,27`), `NANOUNET_N_PROC_DA` (`network.py:66`), `NANOUNET_SINGLE_PATCH_ACCUM_DTYPE` (`predict_case.py:27,33`), `NANOUNET_SEGTRACK_MODEL/TRACK` (`cli/segtrack.py:69-70`), `WANDB_RUN_ID` (`lightning_module.py:91`) |
| K15 | `pyproject.toml [project.scripts]` | entry points | 8 scripts, all `nanounet.cli.<x>:main`. `build_splits`, `build_valset`, `lesion_weights` have no `__main__` guard (`python -m` is a silent no-op). `cli/preprocess.py:29-46` rebuilds `sys.argv` as string flags to call `build_valset.main()` in-process |
| K16 | `nanounet/scripts/slurm_final_900_h200.sh:290` | non-Python import | `from nanounet.lightning_ckpt import pl_ckpt_epoch_and_target`. Grep `*.sh` on every move |
| K17 | `common.py:30` `_REPO_ROOT = Path(__file__).parent.parent`, `runtime.py:97` | `__file__` depth | don't move `common.py`/`runtime.py` to another depth |
| K18 | `cli/segtrack.py:159-162` | output contract | the **first line** of any `SystemExit` raised under `run_case` is printed in the skip line. C error-text edits keep line 1 stable |
| K19 | `model/losses.py:112` (cc3d/scipy), `cli/pretrain.py:127`, `cli/preprocess.py` (build_valset) | lazy imports | keep them deferred (import cost on the default path) |
| K20 | `data/valset.py:34` → `train.patch_iterable` | inverted layering | `data/` imports `train/`. Fixing it touches K11 (S11, optional) |

## 4. Refactor table (S/C only)

| ID | area | finding | fix | LOC Δ | risk | rule | outcome |
|---|---|---|---|---|---|---|---|
| S01 | diag | `mem_diag.py` is 216 LOC | split out `diag/mem_probe.py` (`proc_rss_kb, proc_fds, cgroup_path, _read_int, cgroup_mem_bytes, cgroup_epoch_deltas, gpu_mem_bytes, snapshot, _wandb_scalars, log_wandb_scalars`, ~139). `mem_diag.py` keeps the flag/log half (`_MEM_DIAG`, `set_mem_diag`, `mem_diag_enabled`, `mem_log_every`, `*worker_log_dir`, `append_jsonl`, `log_snapshot`, `worker_diag_*`, ~84). `diag/__init__.py` re-exports the same 9 names, so there are **0 caller edits** (no one imports the submodule directly) | +8 | low | R1 | done (9ecb399) |
| S02 | model | `dice_helpers.py` name | rename → `model/dice_metrics.py`. Update importers `train/ema.py:23`, `train/val_metrics.py:24`, `train/lightning_module.py:17` (import line only), docstring `model/dice_loss.py:3`, `code.md` layout. No shim needed (nothing pickles it, K6) | 0 | low | R4 | done (bf3bf52) |
| S03 | pretrain (hot) | `masking.py` is 29 LOC with 1 fn | move `bottleneck_mask` verbatim into `pretrain/module.py` (143→~166). RNG draw site unchanged (K12) | −5 | low | R2 | done (8b5f2bd) |
| S04 | train (hot) | `patch_size.py` is 26 LOC with 1 fn | move `get_patch_size` verbatim into `data/augment.py` (154→~176). Only importer: `train/data_module.py:22` (`data_module.py` itself would hit 211 LOC) | −4 | low | R2 | done (8b5f2bd) |
| S05 | root | `print0 = cprint` second name (`common.py:118`) | delete the alias; rename-map `print0→cprint` at `pretrain/dataset.py:14,156,168` | −1 | low | R10/U1 | done (d9c9265) |
| S06 | data | private `_sidecar_path` imported cross-module (`cli/build_valset.py:21`) | rename → `sidecar_path` (`data/valset.py:53` + 2 call sites) | 0 | low | naming | done (d9c9265) |
| S07 | cli | duplicate `init_dataloader_ipc` in the second import list (`cli/train.py:33`) | drop the name from line 33 only. Both calls (`:9`, `:44`) stay | 0 | low | R10 | done (d9c9265) |
| S08 | prompt | `bbox_fits_in_patch` is module-private but public-named (`cluster.py:19,47`) | rename → `_bbox_fits_in_patch` (optional) | 0 | low | naming | done (d9c9265) |
| S09 | plan | cgroup readers live in `plan/prep/preprocess.py:26-77` while cgroup code lives in `diag/cgroup.py` | move `_cgroup_mem_limit_gb`, `_cgroup_oom_kills`, `_dead_worker_error` verbatim → `diag/cgroup.py` (52→~105); preprocess.py 164→~112 (optional) | 0 | low | concept | done (b571c1b) |
| S10 | train/data (hot) | `data/` imports `train/` (K20) | move `worker_init` + `collate_patches` → `data/` sibling of `valset.py`, with a re-export at `train/patch_iterable.py` for K11 (optional) | +2 | med | layering | done (06b2246) |
| S11 | infer | `export.py` shrinks to ~35 LOC after X02/X03 | merge the rest into `patch_export.py` (~187) (only if X02/X03 approved) | −10 | low | R2 | not approved (X02/X03) |
| C01 | cli | 123 flags have no `help=` | add `help=`: `train_parser.py` 39, `segtrack.py` 25 (`_mode` 38-57), `pretrain.py` 18, `predict.py` 14, `preprocess.py` 10, `build_valset.py` 9, `build_splits.py` 5, `lesion_weights.py` 3. **Snake_case flags (7) are not renamed** (L28) | +123 | low | U8 | done (6ca4287, ddb350e) |
| C02 | cold modules | boundary raises without `Fix:` | reword to what / expected / `Fix:`: `config.py:71,84,90,92,95,110,113,134,142`; `lightning_ckpt.py:17,25,35`; `dataloader_prefs.py:45`; `runtime.py:142`; `cli/train.py:58,78`; `train_parser.py:67,69,72,74,76,78`; `cli/pretrain.py:121`; `cli/predict.py:91`; `cli/build_valset.py:42,44`; `train/fit.py:46`; `plan/dataset_id.py:66,68`; `plan/labels.py:32,34`; `plan/plans.py:102,111,152`; `plan/prep/merge.py:33,36,39,41,45,47`; `plan/resenc/planner.py:128`; `model/lr_schedule.py:66,68`; `model/mae_transfer.py:21,50` (TypeError); `model/network.py:26` (ImportError); `infer/predictor.py:53`; `infer/export.py:78` | ~+60 | low | E1 | done (2c4e5e6) |
| C03 | hot modules | same, but the raise sits inside a hot-path body (text only; evaluated only on raise) | `data/io.py:45,50,53,101,114`; `data/normalization.py:113`; `data/sampling.py:128`; `data/valset.py:52`; `data/error_table.py:35` (+ waiver at `:66-93`, which is a checker FP); `infer/patch_export.py:121`; `infer/points_pad.py:23`; `prompt/coords.py:33,35,39,67,119,121`; `prompt/cluster.py:66`; `pretrain/dataset.py:160-164`. K18: keep line 1 | ~+30 | low | E1 | done (e43d160) |
| C04 | all | 14 unwaived narrow swallows + 2 unflagged + 1 broad | add `# nanochat-style: allow E4 (<why>)`: `blosc2_dataset.py:34,49`; `mem_diag.py:46,87`; `cgroup.py:22`; `tmp_purge.py:38,49,68,78`; `plan/prep/preprocess.py:31,37,50`; `dataloader_prefs.py:65`; `runtime.py:101`; `patch_export.py:149`; broad `except Exception` at `mem_diag.py:102-114` (waiver only; narrowing it = L32) | +17 | low | E4 | done (545f96f) |
| C05 | user docs | stale or missing | `reference/config.md:43-46`: delete `large_lesion.*` (0 hits in code/configs); add `sampling.instance_targets/cohorts/require_weights` rows (`config.py:29-36`). `steps/train.md` and `steps/pretrain.md`: add `--mem-diag`. New `steps/lesion_weights.md` (or a section in `valset.md`) with 7 flags (D6). `README.md:51`: add the 3 missing entry points. README + `index.md` doc maps: add `valset.md`, `instance_targets.md`. `reference/losses.md:30`: literal command. `valset.md`: add Inputs/outputs. `track.md:88`: `## Common errors` | +60 | low | D2-D6 | done (0f4d1fa) |
| C06 | dev-notes | stale scratch | `final_run_plan.md:260` imports `nanounet.infer.longi_row` and `predict_patch` (active plan; breaks if run); `radiom_embed_api.md:32,37` `predict_patch_logits` row (linked from `steps/predict.md:113`); `cgroup_memory.md:57,158,160-165` claims `--mem-diag` was removed (false: `cli/train.py:42`, `cli/pretrain.py:95-100`); `infer_engine_plan.md:138-163`; `segtrack_wiring_validation_plan.md:3-4` dead link; `HANDOFF_prompt_sensitivity_sweep.md:97`; `longi_*.md` ×3 describe deleted features (archive/delete); add a date+status header ×8 (`docs.md`) | ±0 | low | D4 | done (ddf62a7) |
| C07 | docstrings | after S01/S02/S03/S04 | update `code.md` package-layout block + module docstrings of each touched host file | +6 | low | R6 | done (9ecb399, bf3bf52, 8b5f2bd) |

**Rejected (no change):** `lightning_ckpt.py → runtime.py` (conceptual misfit + K16). `prompt/centroids.py` writer/reader split (142 LOC, not forced). `plan/resenc/*` 3-way split (already a clean concept boundary). `plan/cohorts.py` vs `data/cohorts.py` (writer vs reader, not a duplicate). `plan/splits.py` vs `cli/build_splits.py` (R13 exemplar). `points_pad.py`/`inference_row.py` (legitimate satellites; merging breaks R1). `score.py` (live: `cli/predict.py --gt-dir`).

## 5. Dead code (X, each needs your OK; zero execution impact; X01-X05 confirmed by repo-wide grep; the agents also checked graphify)

| ID | def | evidence | LOC |
|---|---|---|---|
| X01 | `prompt/centroids.py:22-43` `centroids_from_seg` | 0 callers in `nanounet/ scripts/ configs/`. Only prose at `dev-notes/epcm_plan_v0.md:155` | −22 |
| X02 | `infer/export.py:16-21` `save_preprocessed_seg` | 0 callers; mentioned only in the `export.py:33` docstring | −6 |
| X03 | `infer/export.py:24-49` `export_preprocessed_seg_to_native` | 0 callers, 0 doc mentions | −26 |
| X04 | `data/io.py:76-85` `read_images`/`read_seg`/`write_seg` module wrappers | all call sites use `rw.*` or `SimpleITKIO().*` | −10 |
| X05 | `plan/plans.py:128-130` `Plans.image_reader_writer` property | never read. **The JSON key stays** (K8) | −3 |
| X06 | `common.py:29` `DEFAULT_NUM_PROCESSES` (reads `NANOUNET_DEF_N_PROC` at import) | 0 readers | −1 |
| keep | `infer/patch_export.py:93-153` `patch_logits_to_native_seg`, `native_seg_to_nifti_bytes` | 0 in-repo callers but **external Radiom API** (`steps/predict.md:109-111`) | 0 |
| keep | `nanounet/__init__.py:3` `__version__` | unread in repo; external tools may read it | 0 |

## 6. PR series (S/C only, each independently shippable, in order)

| PR | contents | hot-path module? | gates |
|---|---|---|---|
| PR-0 | equivalence harness `equiv/` (§8) + baseline goldens at the P1 SHA + self-check (2 baseline runs identical) | – | self-check |
| PR-1 | S01 (mem_diag split) + C07 part | no | 1-3 |
| PR-2 | S02 (dice_helpers rename) + C07 part | import line in `lightning_module.py` | 1-3 (+4: import-only, one cheap arm) |
| PR-3 | S03 + S04 (R2 inlines) + C07 part | **yes** | 1-4 |
| PR-4 | S05 + S06 + S07 (+ S08) small renames | no (`pretrain/dataset.py` call-site rename only) | 1-3 |
| PR-5 | C01a: `help=` for `train_parser.py`, `pretrain.py` + train/pretrain doc tables | no | 1-3 (help diff only) |
| PR-6 | C01b: `help=` for predict, segtrack, preprocess, build_valset, build_splits, lesion_weights + doc tables | no | 1-3 (help diff only) |
| PR-7 | C02 (Fix: lines, cold modules) | no | 1-3 (raise text only) |
| PR-8 | C03 (Fix: lines, hot modules) | text-only | 1-3 (+4 optional) |
| PR-9 | C04 (E4 waivers) | comments only | 1 |
| PR-10 | C05 (user docs) | – | `check.py` |
| PR-11 | C06 (dev-notes) | – | – |
| opt-A | S09 (cgroup helpers → `diag/cgroup.py`) | no | 1-3 |
| opt-B | S10 (`worker_init`/`collate_patches` → `data/`) | **yes** | 1-4 |
| opt-C | approved X items (+ S11 if X02/X03 approved) | no | 1-3 |

Outcomes (2026-09-24, `nanochat-refactor`): PR-0 `187362c`, PR-1 `9ecb399`, PR-2 `bf3bf52`, PR-3 `8b5f2bd`, PR-4 `d9c9265`, PR-5 `6ca4287`, PR-6 `ddb350e`, PR-7 `2c4e5e6`, PR-8 `e43d160`, PR-9 `545f96f`, PR-10 `0f4d1fa`, PR-11 `ddf62a7`, opt-A `b571c1b`, opt-B `06b2246`. opt-C / S11 / X01–X06: not approved. G4 for PR-3 and opt-B: PASS (2026-09-29, §10; sup repeat −0.5%, mae +1.2%). PR-2 and PR-8 covered by the same A/B arms (HEAD contains them). Harness removed in `5b3b265`.
| opt-D | skill/checker edits (§9) | – | `check.py` self-run |

After the series: delete `equiv/`, run `graphify update .`, then `check.py`. Expected result: 0 errors; U8/E1/E4/D3/D6 warns → ~0 (7 snake_case U8 warns remain unless L22 is approved).

## 7. L table (logic candidates, incl. perf). **Not scheduled.**

| ID | file:line | idea | expected gain | risk | G4 / verification plan |
|---|---|---|---|---|---|
| L01 | `model/cc_dice_ce.py:68,86,109` | remove `.cpu()/.numpy()/.item()` syncs in `_cc_term` (known debt, `gpu.md`) | step time with `--loss cc_dc_ce` only; est. 3-10% | med (cc3d on CPU is intrinsic) | `--loss cc_dc_ce`, `--epochs 4`, median e1-3, loss curve A/B |
| L02 | train loop | `gc.collect(); gc.freeze(); gc.disable()` after step 1 (nanochat `base_train.py:586-594`) | removes GC stalls; est. 0-3% | low | G4 standard arms; check host RSS with `--mem-diag` |
| L03 | fit/Trainer | `torch.backends.cudnn.benchmark=True` (fixed patch) | 5-15% (`gpu.md`) | **breaks bit-identity** (kernel choice) | G4 + val Dice parity over 1 run |
| L04 | train step | `torch.compile(net, dynamic=False)`; eager net kept for sliding window | 10-30% | med-high (recompiles, DS outputs) | G4 + recompile count log |
| L05 | CLI default | `--precision bf16-mixed` on H200 vs `16-mixed` | 0-10%, no GradScaler | numerics | G4 + Dice parity |
| L06 | net | `channels_last_3d` | arch-dependent | med | G4 |
| L07 | `pretrain/dataset.py:33-37` | MAE worker init lacks `pin_worker_threads()` (`patch_iterable.py:61-64` has it): possible BLAS oversubscription | unknown, maybe MAE throughput | low | MAE `epoch_wall_time_sec` A/B + `nvidia-smi dmon` |
| L08 | `runtime.py:63-70` | `explicit` candidate listed twice; `_results_scratch()` called twice | none (hygiene) | low | unit read |
| L09 | `pretrain/masking.py:26` | unseeded global torch RNG: MAE masks aren't fold-reproducible. Pass a seeded Generator | reproducibility | changes the MAE mask stream | golden diff expected; document |
| L10 | dedup, cold (body-changing extraction) | `mae_transfer.py:15-23≡44-52`; `_strip_pl_state` `predictor.py:18` vs `mae_transfer.py:22,51`; spacing block `export.py:36-39` vs `patch_export.py:71-73`; `resolve_tta` `cli/predict.py:73≡cli/segtrack.py:130`; Plans→case_dir ×3 `build_splits.py:38-40`, `build_valset.py:105-107`, `lesion_weights.py:49-52`; ModelCheckpoint list `cli/pretrain.py:157-166≡fit.py:102-111`; folder scan `dataset_id.py:58-69` vs `merge.py:11-17` | −40 LOC | low each | harness 1-3 (golden must still match) |
| L11 | dedup, **hot** | 2-class collapse `losses.py:66-70` vs `dice_helpers.py:63-67`; soft-dice ratio ×3 `dice_loss.py:90`, `cc_dice_ce.py:95,111`; `_META_KEYS` `patch_render.py:80` vs `lightning_module.py:144-147`; autocast ctx ×5; epoch timer `lightning_module.py:95-97` vs `pretrain/module.py:88-92`; `filter_centroids_in_patch` `centroids.py:46-55` vs `cluster.py:83-93`; `allocate` vs `_largest_remainder` `valset_alloc.py:11-46` (equivalence unproven) | −30 LOC | med | harness 1-3 + G4 |
| L12 | `cli/pretrain.py:45-181` vs `train/fit.py:28-128` | two MAE orchestrations that **diverge**: CLI resume lacks `pl_ckpt_assert_epochs_match` (`fit.py:47`) and mem-diag snapshots | −100 LOC | med (behavior converges) | decide which is canonical |
| L13 | `network.py:63`, `runtime.py:24-25,30` | local `import os` → top; inline the `_fs_type` pass-through | none | low | AST guard will flag; approve |
| L14 | `build_splits.py`, `build_valset.py`, `lesion_weights.py` EOF | add a `if __name__ == "__main__": main()` guard | `python -m` works | low | CLI surface |
| L15 | `cli/segtrack.py:62-67` | `--help` unreachable without the `tracking` pkg; parse args before `_require_tracking()` | UX | low | CLI surface |
| L16 | `cli/predict.py:69-71` | silent cuda→cpu downgrade (R17 needs a logged reason); segtrack `:85-89` hard-fails instead | UX | low | CLI surface |
| L17 | `cli/train_parser.py:65-97` | collect all problems, raise once | UX | low | error-path tests |
| L18 | invariants | raise→assert: `dataloader_prefs.py:45`, `normalization.py:113`, `coords.py:33,35,39,67`, `cluster.py:66`, `points_pad.py:23`, `patch_export.py:121`. assert→raise for user JSON: `config.py:86,145` | R5/E2 | low (`-O` semantics) | – |
| L19 | CLI output | U2/U3: `config_table` missing in build_splits, build_valset, lesion_weights, preprocess, pretrain; `next:` missing in lesion_weights, predict, pretrain, segtrack, train | UX | low (U12 lines) | CLI surface |
| L20 | CLI | U9 `--json` is implemented nowhere | agent UX | low | – |
| L21 | `pretrain/module.py:43` | `output_dir` absolute path baked into hparams | hygiene | ckpt contract K6 | – |
| L22 | flags | kebab aliases for 7 snake_case flags (`--dataset_id` ×6, `--num_processes`) | U8 | med (slurm/docs) | CLI surface |
| L23 | `--ema` | default off in predict (`cli/predict.py:33`), on in segtrack (`cli/segtrack.py:53`) | consistency | changes outputs | – |
| L24 | `plan/prep/fingerprint.py:80` | reuse `_dead_worker_error` diagnostics | E1 | low | – |
| L25 | `prompt/coords.py:124` | context for a malformed `point` entry (adds control flow) | E1 | low | – |
| L26 | `diag/mem_diag.py:102-114` | narrow `except Exception` | E4 | low | – |
| L27 | `cli/segtrack.py:62` | move `quiet_lightning_runtime()` to module level like train/pretrain | U5 | K1 | import probe |
| L28 | `model/lr_schedule.py:68` | `epoch_offset` param has no caller or flag | −3 LOC | low | – |

## 8. Equivalence harness (PR-0; temporary, deleted after the series)

Layout: `equiv/{ast_guard.py, synth.py, capture.py, cli_surface.py, renames.json}` at repo root. It sits outside `nanounet/`, so the checker ignores it (R16). Goldens go to
`$EQUIV_OUT/<sha>/` (not committed). One command: `python equiv/run.py --base <sha>` runs 1→3 and exits non-zero on any diff.

### 8.1 AST guard (`ast_guard.py --base <sha> [--allow-text]`)
- Parse every `nanounet/**/*.py` at `<sha>` (`git show`) and at HEAD. Index every `FunctionDef`/`AsyncFunctionDef`/`ClassDef` by **qualname** (nested included).
- Normalize: drop the leading docstring; `ast.dump(include_attributes=False)` (line numbers ignored; decorators, defaults and annotations included).
  Apply `renames.json` (`{"symbols": {"print0": "cprint", ...}, "modules": {"nanounet.model.dice_helpers": "nanounet.model.dice_metrics"}, "deleted": [...]}`) to `Name.id`/`Attribute.attr`/`ImportFrom`.
  With `--allow-text` (C PRs), blank the string constants in `Raise` args, the `assert` msg and `help=` kwargs.
- Verdicts: MOVED (other module, identical: OK) · CHANGED (**fail**) · MISSING (fail unless in `deleted`) · NEW (fail unless an allow-listed re-export).
- Module level: non-import statements are compared per module as a sequence (fail on change). Imports are diffed and printed. For the K1 files (`cli/train.py`, `cli/pretrain.py`, `cli/segtrack.py`), the statement prefix up to the last side-effect call must be identical.
- Contract grep (fail on change): `class EMACallback`, `self.net =`, the ctor kwarg lists of `NanoUNetLM`/`NanoMAELM`, `config.py` dataclass field lists, `resample_data_or_seg_to_shape`, `SimpleITKIO`, `_EMA_CB`, K9 keys, K14 env var strings, and `nanounet\.[a-z_.]+` strings in `scripts/*.sh` must import.

### 8.2 Golden capture (`capture.py`, CPU, deterministic)
Env: `CUDA_VISIBLE_DEVICES=""`, `PYTHONHASHSEED=0`, `OMP/MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`,
`random/np/torch` seeded to 0 **before each stage** (K12: the MAE mask and aug workers use global RNG). Fresh `NANOUNET_RAW/PREPROCESSED/RESULTS/TMPDIR` tree at a
**fixed path** (ckpt/json contents embed absolute paths, L21). Hash = sha256 over `dtype|shape|bytes` per tensor/array; files hashed as raw bytes **and** decoded content.

| stage | what runs | captured |
|---|---|---|
| A synth | `synth.py`: 2 raw datasets (901, 902), 4 cases each, 40×48×56, anisotropic spacing (2.5, 0.8, 0.8) to hit the resampling + aniso DA paths, 2-4 ellipsoid lesions, fixed-seed noise, `dataset.json`, a lesion-type CSV for `lesion_weights` | inputs hash |
| B preprocess | `nanounet_preprocess -d 901 902 --merged-id 903 -np 2 --gpu-memory-gb 8 --valset-config nanounet/configs/default.json --valset-n 16` (merge + fingerprint + plan + spawn pools, K10) → `nanounet_build_splits`, `nanounet_build_valset`, `nanounet_lesion_weights` | every file under `preprocessed/` + `raw/Dataset903*`: blosc2 bytes + decoded arrays, JSON bytes + canonical, `*_centroids.json`, plans.json, splits, cohorts, valset manifest + `.targets.npz` |
| C dataloaders | the `NanoDataModule` wiring used by `run_supervised`, `num_workers=0` **and** `2` (fork); 6 train batches + all val batches; MAE `build_pretrain_dataloaders` 4 batches; `--config nanounet/configs/instance_conditional.json` variant | every tensor/str per batch key (images, segs, heatmaps/prompts, bboxes, modes, keys/meta) |
| D model/loss | `NanoUNetLM` from plans, init under seed 0; fixed batch from C; losses `dc_ce`, `cc_dc_ce`, consistency>0; `backward`; 1 SGD step; LR values for poly + stretched over 20 steps; `EMACallback` 3 updates; `NanoMAELM` masked loss + grads | forward outputs (all DS levels), loss scalars (exact bits), grads per param (sorted names), post-step params, LR list, EMA shadow |
| E keys | `state_dict()` of NanoUNetLM, NanoMAELM, EMA shadow; `hparams` dicts | key, shape, dtype lists |
| F micro-train | `nanounet_train -d 903 -f 0 --accelerator cpu --precision 32 --epochs 1 --iters-per-epoch 2 --val-iters 1 --no-wandb --ema-decay 0.999` (+ `--mae-pretrain --mae-epochs 1 --mae-iters-per-epoch 2` path) and `nanounet_pretrain` 1 epoch/2 iters. The **base** run's ckpts are kept as fixtures `old_sup.ckpt`, `old_mae.ckpt` | ckpt `state_dict`, optimizer state, `callbacks["EMACallback"]`, `hyper_parameters`, `epoch`, `nano_config.json`, metrics CSV minus time columns |
| G predict | `nanounet_predict --device cpu` on 1 case with a points JSON built from its centroids: default, `--tta`, `--ema`, `--disable-tta`; `--gt-dir` for `score.py` | output NIfTI raw bytes + decoded array + score JSON |
| H old-ckpt load | at HEAD: `load_net_from_ckpt(old_sup.ckpt, ema=False/True)`, `load_mae_encoder`/`load_full_net(old_mae.ckpt)`, `pl_ckpt_epoch_and_target`, `nanounet_train --resume old_sup.ckpt --epochs 2` | loaded tensor hashes, resumed-run hashes. Cluster, once per PR series: load one real production `finetune/last.ckpt` with `--ema`; record key count + sha |

Self-check in PR-0: capture twice at the base SHA; everything must match. Any field that is still nondeterministic (e.g. blosc2 header metadata, PL
timestamps) is excluded **by name, with a reason** in `capture.py`. It is never excluded silently.

### 8.3 CLI surface (`cli_surface.py`)
- All 8 console scripts: `--help` captured by calling `main()` with a patched `sys.argv`. Needed because 3 lack a guard (K15); segtrack needs `tracking` installed or a stub module on `sys.path` (L15).
- `config_table` rows: monkeypatch `nanounet.common.config_table` to record its rows, then raise `SystemExit` → run each CLI with a fixed valid argv against the synth tree.
- Import-side-effect probe, one fresh subprocess per CLI module: `TMPDIR`, `tempfile.tempdir`, `torch.multiprocessing.get_sharing_strategy()`, `warnings.filters`,
  logger levels, whether PL was already in `sys.modules` when `quiet_lightning_runtime` ran (hook via `sys.meta_path` recorder).
- Pickle probe: `pickle.dumps` under spawn for `worker_init`, `_worker_init` partial, `CohortSampler`, `Blosc2Folder`, `ValManifest`, K10 pool functions. Their `__module__` must import.
- S PRs: identical output. C PRs: the diff may only touch `help=` text / error text lines; the diff is printed for review.

### 8.4 Throughput (G4, cluster)
Only for PRs that touch a hot-path module (PR-3, opt-B; optional for PR-2/PR-8). Protocol per `gpu.md`: same node type and allocation, both arms back-to-back,
fixed `--dl-bucket/--batch-size/--iters-per-epoch/--val-iters/--precision`, `--epochs 4`, discard epoch 0, median `epoch_wall_time_sec` e1-3 + GPU util.
Pass: |Δ| < 2% (noise band). Otherwise repeat once, then reject. The table goes into the PR description.

**Done =** 1-3 bit-identical (C PRs: text-only diffs) **before** the PR is proposed; plus 4 where required.

## 9. Skill rules to fix or add (edits to `SKILL.md` / `check.py`; rule IDs stay stable, new IDs appended)

| # | target | change | evidence |
|---|---|---|---|
| K-1 | `check.py` D3/D4 | expand `action=argparse.BooleanOptionalAction` into `--no-X` | FP `track.md:68` |
| K-2 | `check.py` E1 | add `TypeError`, `ImportError`, `NotImplementedError`, `OSError` to `BOUNDARY_EXC` | `mae_transfer.py:21,50`, `network.py:26`, `export.py:78`, `coords.py:121` |
| K-3 | `check.py` E1 | resolve a `Name` arg to a same-function string assignment containing `Fix` | FP `error_table.py:66-93` |
| K-4 | `check.py` E4 | flag a broad `except Exception` whose body doesn't re-raise (not only `pass`/`continue` bodies) | `mem_diag.py:102-114` missed |
| K-5 | `check.py` D3 | key flags by (flag, file) and require them in *that command's* step doc; currently `setdefault` keeps the first definition only | `--mem-diag` in `train_parser.py:57` missed (only `pretrain.py:94` reported) |
| K-6 | `check.py` new warn | a `nanounet/cli/*.py` module with `main` but no `__main__` guard | K15 |
| K-7 | `check.py` new warn | U2/U3 mechanical: `main()` calls `nano_header`, `config_table`, and emits `next:` | §7 L19 list |
| K-8 | `check.py` D4 | `nanounet\.[a-z_.]+` module paths in user docs **and** `scripts/*.sh` must import | K16, `final_run_plan.md:260` class of bug |
| K-9 | `check.py` D-rule | dev-notes/handoffs open with `Date:` + `Status:` (`docs.md` says so; 8 files violate) | C06 |
| M-1 | `SKILL.md` new **R18** | "On-disk and ckpt names are frozen": `EMACallback`, `self.net`, LightningModule ctor kwarg names, `config.py` field names, plans.json `__name__` strings, sidecar keys, `NANOUNET_*` env names. Renaming them = logic change | K4-K9, K14 |
| M-2 | `SKILL.md` new **R19** | "Pure-refactor protocol": S/C/L classes, hot paths move-only, AST guard + golden + CLI surface, L never mixed into S/C | this plan |
| M-3 | `code.md` deviations | add R6 banners (nanochat uses them; we split instead) and R7 (nanochat hints only kernels/utils; we hint public signatures) as deliberate deviations | §1 #12-13 |
| M-4 | `nanochat.md` | fix §1 #1-11 (lines, counts, "asserts carry f-strings" overgeneralization); add #14-22 | §1 |
| M-5 | `SKILL.md` U9 | state honestly that no command implements `--json` yet ("applies to new commands; L20 tracks the backfill"). Cite nanochat's unconditional last-line JSON | `infer_bench.py:241` |
| M-6 | `SKILL.md` U12 | cite nanochat's own broken grep as the reason the rule exists | `miniseries.sh:85` |
| M-7 | `gpu.md` | mark `cudnn.benchmark`/`compile`/`bf16` as **bit-identity breaking**: never inside a refactor series | L03-L05 |
| M-8 | `SKILL.md` E4 | explicit carve-out: broad except allowed only inside an R17 import-time capability probe, with a waiver | nanochat `flash_attention.py:45` |

Outcomes: K-1..K-9 and M-1..M-8 done (`5b4e953`). K-6 and K-7 surface new warns (expected). Seven snake_case U8 warns remain (L22, not renamed).

## 10. Cluster verification (2026-09-29)

Setup: 1× NVIDIA H200 (143 GB, idle, driver 580.178.04), 24 CPUs, python 3.11.3, torch 2.7.1+cu118, nanounet editable from `/nanoUNet`
(both arms import `/nanoUNet/nanounet/__init__.py`; both `nanounet_train --help` ok). Arms: A = `e043e4d` (baseline), B = `d1a5aad` (HEAD).
Dataset 900 (`Dataset900_Merged`), fold 0, plans `nnUNetResEncUNetLPlans_h200_smallpv`, `--config nanounet/configs/longrun900.json` (as `nanounet/scripts/slurm_final_900_h200.sh`).
Common flags: `--val-iters 10 --val-every-n-epochs 1 --dl-bucket l --precision 16-mixed --no-wandb`, batch size from plans, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`,
`NANOUNET_TMPDIR=/root/.cache/nanounet_tmp`. sup: `--epochs 4 --iters-per-epoch 80`. mae: `--mae-pretrain --mae-epochs 4 --mae-iters-per-epoch 80 --epochs 1 --iters-per-epoch 20 --dl-persistent-workers`.

Deviations from the protocol (20 min budget): 80 iters/epoch and 10 val iters instead of 250/50. No `--val-manifest`, because it forces all 200 val batches
(which took more than 12 min from the CIFS mount). `NANOUNET_PREPROCESSED` = a local 126-case subset (4 train + 2 val per cohort, all 21 cohorts, `random.Random(0)`, files byte-copied
from `/nnunet_data/NanoUNet_preprocessed/Dataset900_Merged`, `splits_final.json` restricted to the subset). From the CIFS mount, training was I/O-bound (0.25 it/s).
MAE stage rows = the first 4 `epoch_wall_time_sec` rows (the MAE and sup stages both log `epoch` 0 in one CSV). GPU util = mean `sm` over the non-zero `nvidia-smi dmon` samples.

sup (run order A, B):

| arm    | sha     | epoch_wall_time_sec e1/e2/e3 | median | GPU util | Δ      |
|--------|---------|------------------------------|--------|----------|--------|
| before | e043e4d | 86.80 / 86.85 / 83.73        | 86.80  | 87%      |        |
| after  | d1a5aad | 79.18 / 74.87 / 77.32        | 77.32  | 96%      | -10.9% |

sup repeat (run order B, A, as required because |Δ| ≥ 2%):

| arm    | sha     | epoch_wall_time_sec e1/e2/e3 | median | GPU util | Δ      |
|--------|---------|------------------------------|--------|----------|--------|
| before | e043e4d | 90.18 / 87.94 / 84.36        | 87.94  | 88%      |        |
| after  | d1a5aad | 87.02 / 87.50 / 92.86        | 87.50  | 81%      | -0.5%  |

mae (run order A, B):

| arm    | sha     | epoch_wall_time_sec e1/e2/e3 | median | GPU util | Δ      |
|--------|---------|------------------------------|--------|----------|--------|
| before | e043e4d | 43.54 / 43.14 / 42.77        | 43.14  | 78%      |        |
| after  | d1a5aad | 43.56 / 43.66 / 44.07        | 43.66  | 80%      | +1.2%  |

Verdict: sup REPEAT → **PASS** (the first pair's −10.9% was a speed-up in the second-run arm; the repeat is −0.5%. B ran first there and was not faster, so this is run-order noise, not code).
mae **PASS** (+1.2%). No bisect needed.

Production checkpoint load: `Dataset900_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_final_ft250_fromlast/finetune/last.ckpt` (epoch/target 235/250,
EMA shadow present), CPU load via `load_net_from_ckpt`. **IDENTICAL** JSON at both SHAs: n_keys 956 (956 `net.*` keys in ckpt);
ema=False sha256 `38ecfc5376b8fe62afe44725152c8591c8028f3fcf5b841f6fe8074ec0f86be3`; ema=True sha256 `4999a44a1623ebdd7c46a1fdb391cee7276e1a4a6cb15b133f2308be6910e637`.

GPU inference parity (optional §4 step): not run (time budget).
