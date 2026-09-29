# Code: structure, naming, idioms

Load this for any `.py` under `nanounet/`. Rule IDs refer to SKILL.md.

## What nanochat actually does (verified at commit 92d63d4)

- **A file is one concept, not one size.** `gpt.py` (555 LOC) holds the whole model: config, layers, forward,
  generate, and FLOP accounting. `loss_eval.py` (65 LOC) is one function. `common.py` (328 LOC) collects the
  cross-cutting infrastructure: dtype/device detection, DDP init, `print0`, `DummyWandb`, and peak-FLOP tables.
- **All 13 library files open with a docstring.** Its length matches how tricky the file is: 3 lines for
  `checkpoint_manager.py`, 70 lines of design notes for `fp8.py`.
- **Classes appear only where state is owned.** `nn.Module` layers, `KVCache`, `DummyWandb` (null object), and one
  `Optimizer` subclass because the torch API requires it. Two dataclasses in the whole library. Dispatch is
  `if group['kind'] == 'adamw': ... elif ...: ... else: raise`. No ABCs, no registries.
- **Asserts dominate** (41 asserts vs 11 raises), each with an actual-vs-expected f-string:
  `assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"`.
- **Comments are dense and mostly explain *why*.** Shape annotations like `# (B, T, H, D)` are everywhere. Paper and PR URLs sit right
  above the code that implements them. TODOs and "Hack:" notes are left honestly in place.
- **Hardware detection happens once at import**, with a reason: `COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()`.

## Deliberate nanoUNet deviations

| Deviation | Why |
|---|---|
| **R1: <200 LOC hard cap** (nanochat goes to 555) | Smaller context per file for humans and agents. Forces a concept split before a file sprawls. |
| **R20: concept subfolders**, `nanounet/<area>/<concept>/` (nanochat is one flat `nanochat/`) | 90+ modules. The path should say what a file is before you open it: `data/valset/build.py`. |
| **PyTorch Lightning** instead of a hand-rolled loop | Multi-GPU, checkpointing, and logging for free. Non-trivial custom logic goes in the LightningModule, not in callbacks (R14). |
| **Heavy upstream reuse** (`dynamic_network_architectures`, `batchgeneratorsv2`, `acvl_utils`, `cc3d`, `blosc2`, `SimpleITK`) | Reimplementing them is bloat. |
| **R16: temporary tests** (nanochat keeps a small permanent `tests/`) | Research velocity. Re-evaluate if a regression bites twice. |
| **R6: no section banners** (nanochat uses them inside big files) | We split on a concept boundary instead. nanochat banners: `optim.py:17,65,182`, `tokenizer.py:28,261`. |
| **R7: public signatures are hinted** (nanochat hints kernels and small utils only) | `gpt.py` 0/25 defs hinted, `tokenizer.py` 0/19. We hint public signatures; tensor code still prefers a shape comment. |

## Package layout (keep this accurate; update it when you add or move a folder)

```
nanounet/
├── cli/            one file per console script (+ train_parser.py, segtrack_cases.py helpers); flat by rule
├── data/
│   ├── store/      blosc2_dataset (preprocessed cases), io (SimpleITK reader/writer)
│   ├── volume/     crop, resampling, normalization
│   ├── augment/    transforms (train/val chains), spatial_points (click-carrying transforms)
│   ├── patch/      sampling (click jitter, build_patch), bbox, instance_target, error_table, cohorts
│   ├── valset/     manifest (schema + dataset), alloc, build
│   └── loader/     prefs (worker/prefetch presets), workers (worker_init, collate)
├── prompt/         centroids, click coords, encoding, clustering
├── plan/           plans, labels
│   ├── dataset/    ids, splits, cohorts, lesion_types
│   ├── prep/       fingerprint, case_pp, preprocess, merge
│   └── resenc/     ResEnc planner, VRAM loop, topology
├── model/          network, lr_schedule, mae_transfer
│   └── loss/       losses (DC+CE, build_loss), dice, cc_dice_ce, dice_metrics
├── train/          fit (MAE + supervised orchestration)
│   ├── patches/    data_module, iterable, render
│   └── module/     lightning_module, ema, val_metrics
├── pretrain/       MAE pretraining (augment, dataset, module)
├── infer/
│   ├── predict/    predictor (ckpt load), io, points_pad, roi_slices, inference_row, tta, case
│   ├── export/     volume (logits → native seg), tiles (tile paste, NIfTI bytes)
│   └── segtrack/   track, case
├── diag/           cgroup, mem_diag (flag + JSONL), mem_probe (RSS/cgroup/GPU readers), tmp_purge
├── common.py       console + rich helpers, env paths, logging
├── config.py       dataclass config + load/save
└── runtime.py  lightning_ckpt.py  score.py   (flat single-concept modules)
```

## Folder layout (R20)

- **Two levels, never three.** `nanounet/<area>/<concept>/<module>.py`. An area is a pipeline stage (`data`, `train`,
  `infer`); a concept is a noun inside it (`valset`, `patch`, `export`).
- **Group at 7.** An area with more than 6 flat modules groups them. Its entry points (`train/fit.py`,
  `model/network.py`) and true singletons may stay flat next to the subfolders.
- **2–8 modules per concept subfolder.** One module is not a concept: keep it flat in the area. Nine means two concepts.
- **The folder is part of the name.** `valset/build.py`, `predict/case.py`, `loss/dice.py`. Never `valset/valset_build.py`.
- **Every subfolder `__init__.py` is one docstring line** naming the concept. No re-exports unless a caller count
  justifies it (`diag/`), and no imports that run code: `cli/train.py` imports `data.loader.prefs` before torch (K1).
- **`cli/` stays flat.** One file per console script, 1:1 with `[project.scripts]`. Helpers sit next to their command.
- **Placing a new file:** pick the area by pipeline stage, then the concept by what it *is*. If no concept fits and
  the area is at 6 flat modules, make the concept folder now and move its sibling in the same change.
- **Moving a file is S (R19)**, but paths leak: `[project.scripts]`, `scripts/*.sh`, spawn/DataLoader pickles
  (K10/K11), docs, and this layout block. Rewrite every dotted and slash path in one commit. Re-export only for
  on-disk pickles (checkpoints store none; K4/K6 are class and kwarg names, not module paths).
- A homeless function goes into `common.py` or flat at package root, never into a new folder for one file.

## Hard rules in detail

- **R1/R2 splitting.** Split on a noun: `patch/sampling.py` → `patch/sampling.py` + `patch/bbox.py`. Never split into `_part2.py` or
  `_impl.py`. After a split, both files still need a docstring. Re-export only if callers are many.
- **R3 dispatch.** Use `if/elif/else`. If `kind` is user-supplied, the `else` raises an E1 message that lists the valid values.
  If `kind` is internal, `assert kind in VALID, kind` first. A class is justified only
  when it owns state that lives across calls (a model, a cache, a dataset handle).
- **R5 asserts.** Use them on shapes, dtypes, and internal invariants: `assert seg.ndim == 4, f"seg {seg.shape}: expected (C,Z,Y,X)"`.
  An assert with no message is fine only when the condition reads as its own message.
- **R12 vs R17.** Never fall back on **data**: no recomputing centroids from seg, no guessed spacing, no default plan. Falling back on
  **capability** is fine, but only once, at import, and logged: bf16→fp16 on pre-Ampere, FA3→SDPA. The difference is that capability
  fallbacks produce the same result more slowly, while data fallbacks produce a *different* result silently.
- **R13 CLI shape.** `main()` does argparse, then validate (collect all problems, E6), then `nano_header`, then
  `config_table`, then work, then summary with `next:`. Business logic lives in the package, and the CLI only orchestrates.
  Reference: `nanounet/cli/build_splits.py` (83 LOC).
- **R15 startup validation.** Check paths, plan keys, checkpoint architecture match, and GPU/cgroup memory before building the
  dataloader. `validate_train_args` in `cli/train_parser.py` is the pattern.

## Naming

- Use `snake_case` for functions, files, and folders, `PascalCase` for classes, and `UPPER_CASE` for constants. Private helpers get a `_`-prefix.
- Keep names short and precise: `seg`, `bbox`, `hm`, `lr`, `cts`, `zyx`, `pslc`, the way nanochat uses `B, T, C, q, k, v, idx`.
- Put the axis order in the name when it's ambiguous: `centroid_zyx`, `spacing_xyz`. Silent axis flips are the #1 3D bug.
- Files and folders are nouns: `centroids.py`, `infer/`. Never `centroid_utils.py` or `inference_helpers/`.

## Exemplars: read one of these before writing a new file

| File | Why |
|---|---|
| `nanounet/prompt/centroids.py` | Its docstring explains *why* `seed_zyx` exists (a centroid falls outside a concave lesion in ~12% of cases). |
| `nanounet/cli/build_splits.py` | A complete CLI: header, validate, work, rich table, backup instead of silent overwrite, `next:` line. |
| `nanounet/data/patch/sampling.py` | Its docstring states the order of operations and the one shared-state subtlety. |

## Things we will not write

- A 400 LOC `BaseSampler` with three subclasses and a registry.
- `sampling_utils/spurious_helpers.py` holding one 5-line function.
- A `try/except` around every cc3d call that logs and falls back to scipy.
- A `Settings` singleton imported into every module.
- A wrapper class around `pl.Trainer`.
