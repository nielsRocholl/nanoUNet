---
name: nanochat-style
description: Non-negotiable philosophy for nanoUNet — nanochat-inspired code (<200 LOC files, no utils/ABC/factories), rich-CLI UX, actionable errors, zero GPU starvation, structured docs. Use when writing, reviewing, or refactoring anything under nanounet/ or scripts/, when adding/changing CLI commands or their output, when writing error messages, when touching dataloaders/training loops (throughput matters), when writing or updating docs/, or when the user references nanochat-style or nanoUNet conventions.
---

# nanochat-style: the nanoUNet way

Four pillars, all **non-negotiable**:

1. **Code** reads like nanochat: small, flat, procedural, zero framework ceremony.
2. **UX** is polished: every CLI is rich-formatted, informative, never noisy.
3. **Failures teach**: every error tells the user what is wrong and what to run next.
4. **GPU never starves**: the data path is engineered and *measured* so compute is the bottleneck.

Plus one supporting habit: **docs are structured and current** (small files, tables, diagrams, copy-paste commands).

## 1. Code style (distilled from Karpathy's nanochat)

Direct observations from [nanochat](https://github.com/karpathy/nanochat) (`nanochat/gpt.py`, `nanochat/dataloader.py`, `nanochat/common.py`, `scripts/base_train.py`):

- One file = one concept. `gpt.py` (~550 LOC) is the whole transformer; `dataloader.py` is a generator function; `optim.py` is just the optimizer. Files split only on concept boundaries, never for size cosmetics.
- Every file opens with a short `"""docstring"""` listing what it contains and any non-obvious feature ("rotary embeddings", "QK norm", "100% utilization, no padding"). Reader oriented in 30 seconds.
- Scripts are top-to-bottom procedural: argparse at the top, code falls through to the loop, cleanup at the bottom. They read like a notebook, not a framework entry point.
- Tiny utilities live together in one `common.py` (~330 LOC, a dozen helpers). No `utils/` package, no registries, no abstract-base anything.
- Dataclasses for config, argparse for CLI. Globals fine when constant (`COMPUTE_DTYPE` detected once at import). Classes only when state is owned.
- No defensive programming: `assert split in ("train","val")`, errors die at the call site.
- Comments say *why*, never *what*.
- No ABCs, factories, plugins, mixins, Strategy patterns. If a class can be a function, it is.

### How nanoUNet adopts this (deviations, all deliberate)

1. **Grouped subfolders** (nanochat is flat). One level deep, noun names, never `utils/` or `helpers/`:
   ```
   nanounet/
   ├── data/       prompt/      plan/       model/
   ├── train/      pretrain/    infer/      register/
   ├── cli/        # one file per console command
   ├── common.py   # console, paths, env, seed — nanochat-style grab bag
   ├── config.py   # dataclass config + load/save
   └── runtime.py, mem_diag.py, dataloader_prefs.py, lightning_ckpt.py
       # flat single-concept modules are fine, exactly like nanochat
   ```
   Keep folders small (roughly ≤6 files; `cli/` exempt). A homeless file goes in `common.py` or flat at package root — never a new folder for one file.
2. **PyTorch Lightning** replaces the hand-rolled loop. Use `LightningModule`, `LightningDataModule`, `Trainer`, `WandbLogger`, `ModelCheckpoint` **directly** — no `BaseTrainer`, no wrapper layer. Non-trivial custom logic goes in the LightningModule, not in callbacks.
3. **Heavy reuse of upstream packages** (`dynamic_network_architectures`, `batchgeneratorsv2`, `acvl_utils`, `cc3d`, `blosc2`, `SimpleITK`). Reimplementing them is bloat.

### Hard rules

| # | Rule |
|---|------|
| R1 | **<200 LOC per file.** Hard limit. Split on a concept boundary, not arbitrarily. |
| R2 | No file <~30 LOC hosting one function. Inline into nearest sibling or `common.py`. |
| R3 | No ABCs, factories, registries, plugins. Two cases = `if cfg.x == "a": ... else: ...`. |
| R4 | No `utils/` package. Real noun names: `geometry.py`, `centroids.py`. |
| R5 | No defensive programming. `assert` on invariants; raise at boundaries only (see §3). |
| R6 | One module docstring per file: what's inside + non-obvious features. No section banners. |
| R7 | Type hints on public signatures and dataclasses. Skip inside small helpers. |
| R8 | Dataclasses for config, argparse for CLI, JSON on disk. No Hydra/OmegaConf/Pydantic. |
| R9 | Constants and detected facts at module top as `UPPER_CASE`. No `Settings()` singleton. |
| R10 | Comments explain *why*. Delete any comment paraphrasing the next line. |
| R11 | No bare `print` outside `common.py`. All output via `cprint`/`nano_header`/rich (see §2). |
| R12 | **No fallbacks for missing data, ever.** Missing centroids/plan/file → raise with fix (§3). |
| R13 | CLI scripts top-to-bottom procedural. `main()` exists only because console_scripts needs it; it holds argparse + orchestration, no business logic. |
| R14 | No abstraction layers over Lightning (see deviation 2). |
| R15 | Validate config and inputs at startup; crash before the first expensive step. The training loop assumes valid state. |
| R16 | Tests are temporary: write, validate, delete. No permanent `tests/` folder. |

### Naming

`snake_case` functions/files/folders, `PascalCase` classes, `UPPER_CASE` constants. Short precise names: `bbox`, `seg`, `hm`, `lr` (nanochat uses `q`, `k`, `v`, `B`, `T`). Files and folders are nouns: `centroids.py`, `infer/` — never `centroid_utils.py`, `inference_helpers/`.

## 2. UX: rich CLI, always

nanoUNet must feel like a polished tool, not a research script. All conventions already have infrastructure in `nanounet/common.py` — use it, don't reinvent.

| # | Rule |
|---|------|
| U1 | One `Console` for the whole package (`common.py`, stderr). Every user-facing line goes through `cprint`, `nano_header`, or a rich renderable. Never mix in raw `print`/`tqdm`. |
| U2 | Every CLI command opens with a header (`nano_header`): command name, resolved dataset/config, device. And closes with a summary: what was produced, where it was written, and the suggested next command. |
| U3 | Resolved configuration is shown as a rich `Table` (argument, value, source: cli / config file / default) before work starts. No wall-of-text config dumps. |
| U4 | Anything taking >2s gets a rich `Progress` bar (spinner + elapsed/remaining). Rank-0 only under DDP; never nested bars. |
| U5 | Third-party noise (Lightning, SimpleITK, wandb banners) is suppressed by default (`quiet_lightning_runtime` pattern). `--verbose` re-enables it. |
| U6 | Tabular information (per-case results, fold summaries, timing breakdowns) is a rich `Table`, never aligned-by-hand text. |
| U7 | Output is calm: no duplicate lines, no leftover debug prints, no progress spam in logs. If wandb captures it, the terminal doesn't need to repeat it. |

## 3. Errors that teach

An error message is UX. Every failure at a user boundary (CLI args, config load, file I/O, environment) must answer three questions:

1. **What is wrong** — exact path/field/value, quoted.
2. **What was expected** — where the code looked, what format it wanted.
3. **What to run next** — a literal copy-paste command or config edit.

```python
raise FileNotFoundError(
    f"No preprocessing plan at {plan_path}.\n"
    f"Expected output of the plan step for dataset {dataset_id}.\n"
    f"Fix: nanounet_preprocess -d {dataset_id}   (see docs/steps/preprocess.md)"
)
```

| # | Rule |
|---|------|
| E1 | Boundary errors use the 3-question template above; point at the relevant `docs/steps/*.md` when one exists. |
| E2 | Internal invariants use bare `assert` — if it fires, it's our bug, not the user's. |
| E3 | All validation happens at startup (R15): paths, config fields, GPU presence, checkpoint compatibility. Never fail 20 minutes into training on something checkable at t=0. |
| E4 | No try/except that logs and continues, no silent fallback or recompute (R12). Crash loudly, once, with the fix. |
| E5 | Tracebacks are for our bugs. User mistakes (missing file, bad flag) exit with a clean rich-formatted error, not a 40-frame stack. |

## 4. GPU efficiency: compute is the bottleneck, always

This is a core design goal of nanoUNet: **hardware runs at the absolute limit of its ability. GPU starvation is a bug**, same severity as a wrong loss. Pretraining and main training already achieve near-full utilization — every change must preserve that.

| # | Rule |
|---|------|
| G1 | The data path never blocks the GPU: pinned CPU staging buffers, `non_blocking=True` H2D copies, prefetch the next batch while the GPU runs fwd/bwd (nanochat `base_train.py` pattern). `persistent_workers=True`; worker counts respect `dataloader_prefs`. |
| G2 | No CPU–GPU sync points inside the hot loop. `.item()`, `.cpu()`, tensor prints only at logging intervals — and know that each one is a sync. |
| G3 | Heavy CPU work (augmentation, blosc2 decode, resampling) lives in dataloader workers, never in the training process main thread. |
| G4 | **Measure, don't guess.** Any change touching the data path or training step requires a before/after throughput check (samples/s or step time; `mem_diag`/wandb already log this). Report the numbers in the PR/commit. |
| G5 | A feature that costs step time must justify it explicitly. "Utilization dropped from ~100% to 85%" is a rejected change until fixed. |
| G6 | Watch the known killers: undersized `num_workers`, `pin_memory=False`, per-step host-side logging, synchronous metric computation, CPU-side resampling in the loop, accidental `.cuda()` inside `__getitem__`. |

## 5. Documentation: small, structured, current

Same philosophy as code: no single large file, every doc scoped to one concept.

```
docs/
├── index.md          # pipeline overview: mermaid flow diagram, quickstart, links
├── steps/            # one file per pipeline stage
│   ├── preprocess.md
│   ├── plan.md
│   ├── pretrain.md
│   ├── train.md
│   └── predict.md
├── reference/        # config fields, losses, sampling internals
└── dev-notes/        # scratch / experiments, not user-facing
```

| # | Rule |
|---|------|
| D1 | `index.md` holds the big picture: a mermaid flowchart of the pipeline (raw → preprocess → plan → pretrain → train → predict), a quickstart command sequence, and links into `steps/`. |
| D2 | Every step doc has, in order: 3-line summary, copy-paste command block, **argument table**, inputs/outputs (paths + formats), common errors with fixes. |
| D3 | Argument tables are mandatory and use this exact format: |
| D4 | Docs are code: <200 lines per file, updated **in the same change** that adds or alters a CLI flag or output path. Stale docs are a bug. |
| D5 | Commands in docs are literal and runnable — real dataset placeholders (`-d 501`), no pseudo-syntax. |

Argument table format (D3):

```markdown
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d, --dataset` | int | required | Dataset ID (e.g. 501) |
| `--fold` | int | 0 | Cross-validation fold |
```

## Reference: a "good" file

```python
"""4-mode patch sampling: pos, pos+spurious, pos+no_prompt, negative.

Mode draws from cfg.sampling.mode_probs. For non-negative modes the patch is
forced to overlap a foreground voxel from properties['class_locations'].
Centroids may carry a propagation offset to simulate baseline->follow-up COG.
"""

from dataclasses import dataclass
import numpy as np

from nanounet.config import SamplingConfig
from nanounet.prompt.propagation import apply_propagation_offset

MODE_POS, MODE_SPUR, MODE_NO_PROMPT, MODE_NEG = 0, 1, 2, 3


def mode_bbox(shape, class_locations, cfg: SamplingConfig, rng):
    mode = int(rng.choice(4, p=cfg.mode_probs))
    force_fg = mode != MODE_NEG
    lbs, ubs = _bbox(shape, force_fg, class_locations, rng)
    return lbs, ubs, mode

def _bbox(...): ...
```

~100 LOC, no class, no factory, top docstring tells you everything. **This is the target.**

## Reference: things we will not write

- A 400 LOC `BaseSampler` with three subclasses and a registry.
- `sampling_utils/spurious_helpers.py` with one 5-line function.
- `raise ValueError("invalid input")` — which input? expected what? fix how?
- A try/except around every cc3d call that logs and falls back to scipy.
- A dataloader change merged without a before/after samples/s number.
- A new `--flag` whose argument table entry is "TODO".
- A `Settings` singleton imported into every module.

If you catch yourself writing any of the above, stop and fix it.

## Review checklist (run on every change)

- [ ] Every touched file <200 LOC, has a *why*-docstring, no banner comments.
- [ ] No new abstraction that a function or an `if` could replace.
- [ ] All output goes through `common.py` rich helpers; command has header + summary.
- [ ] Every new failure path names the problem, the expectation, and the fix command.
- [ ] Validation moved to startup; nothing fails late that could fail at t=0.
- [ ] Data-path change → before/after throughput numbers reported.
- [ ] CLI flag or output path changed → step doc + argument table updated in same change.
