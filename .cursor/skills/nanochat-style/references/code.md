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
| **Grouped subfolders**, at most 2 levels (`plan/prep/`) | 90+ modules. A flat package would be unreadable. |
| **PyTorch Lightning** instead of a hand-rolled loop | Multi-GPU, checkpointing, and logging for free. Non-trivial custom logic goes in the LightningModule, not in callbacks (R14). |
| **Heavy upstream reuse** (`dynamic_network_architectures`, `batchgeneratorsv2`, `acvl_utils`, `cc3d`, `blosc2`, `SimpleITK`) | Reimplementing them is bloat. |
| **R16: temporary tests** (nanochat keeps a small permanent `tests/`) | Research velocity. Re-evaluate if a regression bites twice. |

## Package layout (keep this accurate; update it when you add a folder)

```
nanounet/
├── cli/        one file per console script (+ train_parser.py, segtrack_cases.py helpers)
├── data/       blosc2 dataset, crop/resample/normalize, augment, sampling, valset build
├── prompt/     centroids, click coords, encoding, clustering
├── plan/       dataset ids, plans, splits, labels; prep/ = preprocessing, resenc/ = ResEnc planner
├── model/      network, losses (dice, cc_dice_ce), dice_metrics, lr schedule, MAE transfer
├── train/      LightningModule, data module, fit, EMA, patch iterable/render, val metrics
├── pretrain/   MAE pretraining (dataset, module)
├── infer/      predictor, predict_case/io, TTA, ROI slices, export, segtrack
├── diag/       cgroup, mem_diag, tmp_purge (runtime diagnostics)
├── common.py   console + rich helpers, env paths, logging
├── config.py   dataclass config + load/save
└── runtime.py  dataloader_prefs.py  lightning_ckpt.py  score.py   (flat single-concept modules)
```

Keep folders to roughly 6–16 files. A homeless function goes into `common.py` or flat at package root,
never into a new folder for one file.

## Hard rules in detail

- **R1/R2 splitting.** Split on a noun: `sampling.py` → `sampling.py` + `patch_bbox.py`. Never split into `_part2.py` or
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
| `nanounet/data/sampling.py` | Its docstring states the order of operations and the one shared-state subtlety. |

## Things we will not write

- A 400 LOC `BaseSampler` with three subclasses and a registry.
- `sampling_utils/spurious_helpers.py` holding one 5-line function.
- A `try/except` around every cc3d call that logs and falls back to scipy.
- A `Settings` singleton imported into every module.
- A wrapper class around `pl.Trainer`.
