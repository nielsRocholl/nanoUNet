---
name: nanochat-style
description: nanoUNet's non-negotiable engineering standard, with a checker script. nanochat-style code (<200 LOC files, flat procedural, no utils/ABC/factories), a rich CLI that is also machine-drivable (stderr for humans, JSON on stdout, clean exit codes), errors that name the fix, zero GPU starvation (measured, not assumed), docs kept in sync with code. Use whenever writing, reviewing, refactoring or planning code under nanounet/ or scripts/; adding or changing a CLI command, flag, output line or error message; touching dataloaders, sampling, augmentation, losses, the training step or inference (throughput matters); editing docs/; or when the user says nanochat, nanoUNet style, "clean up", "make it production quality" or asks for a code-quality review.
---

# nanochat-style: the nanoUNet standard

Write like an engineer who hand-wrote conv kernels before frameworks existed and ran a latency-bound
trading desk. That means three things. Use the smallest code that is correct. Use the fastest path, and prove it is fastest with a
number. Build a tool that a stranger, or an agent, can drive without reading the source.

Five pillars. All are **non-negotiable**:

1. **Code** reads like [nanochat](https://github.com/karpathy/nanochat): small files, flat, procedural, no framework ceremony.
2. **UX** is polished for humans (rich, calm, informative) *and* scriptable for agents (JSON, exit codes, no prompts).
3. **Failures teach**: every user-facing error says what is wrong, what was expected, and what to run next.
4. **The GPU never starves**: the data path is engineered and *measured* so compute is the bottleneck.
5. **Docs are code**: small, structured, updated in the same change, and checked mechanically for staleness.

## Workflow: run this on every change

1. **Orient.** Run `graphify query "<concept>"` (per CLAUDE.md), then read the module docstring of each file you will touch.
   Grep `nanounet/common.py` before writing any helper. It probably exists already.
2. **Load only the references for the area you touch** (see the map below). Don't load them all.
3. **Write.** Follow the rule index. If you break a rule deliberately, waive it inline with a reason:
   `# nanochat-style: allow R1 (why)`. A waiver with no reason is itself a violation.
4. **Check.** Run `python .claude/skills/nanochat-style/scripts/check.py --changed`. Fix every `error`. Fix every
   `warn` on lines you wrote. Debt you didn't touch isn't yours, but never add to it. The checker covers
   R1 R2 R3 R4 R6 R11 U1 U8 E1 E4 G2 D3 D4 D6. Every other rule is judgment, so apply it yourself.
5. **Run the gates for your area.**
   - Data path, sampling, augmentation, loss, or the train/infer step: report throughput before/after (G4, `references/gpu.md`).
   - CLI flag, output path, or log line: update the step doc and its argument table in the same change (D4). `<cmd> --help` must read cleanly.
   - New or changed error: trigger it once and paste the rendered message into your report.
6. **Refresh the graph.** Run `graphify update .`.
7. **Report.** List the rule IDs you touched or waived, the checker result, and any throughput numbers. Don't write "should be faster".

## Reference map (load on demand)

| You are touching | Read |
|---|---|
| Any `.py` under `nanounet/` (structure, naming, classes, asserts) | `references/code.md` |
| `nanounet/cli/`, terminal output, error messages, exit codes, `--json` | `references/cli.md` |
| Dataloaders, sampling, augmentation, losses, `lightning_module`, `infer/`, anything per-step | `references/gpu.md` |
| `docs/`, README, dev-notes, experiment logs | `references/docs.md` |
| "Does nanochat really do X?", or justifying and challenging a rule | `references/nanochat.md` |

## Rule index

IDs are stable. Code comments cite them, e.g. `(R12)`. **auto** = `scripts/check.py` enforces it.

| ID | Rule | |
|---|---|---|
| R1 | **<200 LOC per file.** Split on a concept boundary. This is stricter than nanochat, by choice. | auto |
| R2 | No file under ~30 LOC hosting one function. Inline it into a sibling or `common.py`. | auto |
| R3 | No ABCs, factories, registries, plugins, mixins. Two cases means `if cfg.x == "a": ... else: ...` | auto |
| R4 | No `utils`/`helpers` names. Use real nouns: `geometry.py`, `centroids.py`. | auto |
| R5 | Guard invariants with `assert cond, f"actual {x} vs expected {y}"`. Raise only at user boundaries (E1). | |
| R6 | Every module opens with a docstring: what's inside, plus the non-obvious *why*. No section-banner comments. | auto |
| R7 | Type-hint public signatures and dataclasses. In tensor code, shape comments beat hints: `# (B, C, Z, Y, X)`. | |
| R8 | Dataclasses for config, argparse for CLI, JSON on disk. No Hydra, OmegaConf, or Pydantic. | |
| R9 | Put constants and once-detected facts at module top as `UPPER_CASE`. No `Settings()` singleton. | |
| R10 | Comments explain *why*. Shape and units annotations are welcome. Delete comments that paraphrase code. | |
| R11 | No bare `print`. All output goes through `cprint`, `nano_header`, `config_table`, `nano_progress`, or rich. | auto |
| R12 | **No fallbacks for missing data, ever.** Missing centroids, plan, or file means raise with the fix. | |
| R13 | CLI files run top-to-bottom: parse, validate, header, work, summary. `main()` holds orchestration only. | |
| R14 | No abstraction layer over Lightning. Use `LightningModule`, `Trainer`, and callbacks directly. | |
| R15 | Validate everything at startup. The loop assumes valid state. | |
| R16 | Tests are temporary: write, validate, delete. No permanent `tests/`. (nanochat differs, see nanochat.md.) | |
| R17 | Hardware capability differences are detected **once at import**, with the reason logged, e.g. `COMPUTE_DTYPE, REASON = ...`. This is the only allowed "fallback". | |
| R18 | **On-disk and ckpt names are frozen.** `EMACallback`, `self.net`, LightningModule ctor kwargs, `config.py` field names, plans.json `__name__` strings, sidecar keys, `NANOUNET_*` env names. Renaming any of them is a logic change. | |
| R19 | **Pure-refactor protocol.** Changes are S (move, byte-identical body), C (comments, help, error text, docs), or L (anything else). L is never mixed into an S/C commit. Hot paths are move-only. A series ships with an AST guard, a golden capture, and a CLI surface diff. | | |
| U1 | One stderr `Console` (`common.py`). No raw `print`, no tqdm. | auto |
| U2 | Every command opens with `nano_header` and closes with a summary (outputs, paths, time) plus `next: <literal command>`. | |
| U3 | Show the resolved config via `config_table` (argument, value, source) before work starts. | |
| U4 | Anything taking more than ~2 s gets `nano_progress`. Never nest bars. | |
| U5 | Third-party noise is suppressed by default. A `--verbose` flag restores it. | |
| U6 | Tabular info is a rich `Table`, never text aligned by hand. | |
| U7 | Output is calm: no duplicates, no debug leftovers, no progress spam in logs. | |
| U8 | Flags are kebab-case and every one has `help=`. `-1`/`None` means "auto" and the help text says so. | auto |
| U9 | **Machine contract.** Humans read stderr. `--json` prints one JSON object as the **last stdout line**. No command implements it yet (L20). nanochat prints that JSON unconditionally (`infer_bench.py:241`). | |
| U10 | Exit codes: 0 ok, 1 user error (`SystemExit(msg)`), 2 argparse usage. A traceback means *our* bug. | |
| U11 | Never block on input. No `input()`, no confirmation prompts. Destructive actions need an explicit flag (`--force`). | |
| U12 | Status lines are a stable, greppable contract: `key: value \| key: value`. Changing one is a breaking change. nanochat's own grep is already stale: `miniseries.sh:85` looks for `Number of parameters:`, which `base_train.py` never prints. | |
| E1 | Boundary errors answer 3 lines: **what is wrong**, **expected**, `Fix: <literal command>` (plus a doc link). | auto |
| E2 | An internal invariant is a bare `assert`. If it fires, the bug is ours. | |
| E3 | Everything checkable at t=0 fails at t=0: paths, fields, GPU, checkpoint compatibility. | |
| E4 | No swallowing. A narrow `except OSError: pass` is allowed only for best-effort side effects, with a waiver. A broad `except` is allowed only inside an R17 import-time capability probe, and it still needs the waiver. | auto |
| E5 | User mistakes exit through `raise SystemExit(msg)`, which prints cleanly with exit code 1. No 40-frame stack. | |
| E6 | Report **all** startup problems in one error, not one per run. Invalid choice means list the valid ones. | |
| G1 | Data path: pinned staging, `non_blocking=True`, prefetch while the GPU runs, workers per `dataloader_prefs`. | |
| G2 | No CPU-GPU sync (`.item()`, `.cpu()`, `.tolist()`, prints) in the hot path, except at log steps. | auto |
| G3 | Heavy CPU work (augmentation, blosc2 decode, resampling) runs in workers, never on the main thread. | |
| G4 | **Measure, don't guess.** Data-path or step changes ship with a before/after `epoch_wall_time_sec`. | |
| G5 | Any step-time regression must be justified in writing. Otherwise it is rejected. | |
| G6 | Know the killers: too few workers, no pinning, per-step host logging, sync metrics, `.cuda()` in `__getitem__`. | |
| G7 | Inference runs under `@torch.inference_mode()`. Benchmarks warm up before timing. Use `synchronize()` only around timers. | |
| D1 | `docs/index.md` holds the mermaid pipeline, the quickstart, and links into `steps/`. | |
| D2 | A step doc contains, in order: summary, command block, argument table, inputs/outputs, common errors with fixes. | |
| D3 | Every CLI flag appears in a docs argument table (exact format in `references/docs.md`). | auto |
| D4 | User docs are under 200 lines, runnable, and **never stale**: no dead commands or flags. Update in the same change. | auto |
| D5 | Doc commands are literal and runnable (`-d 501`). No pseudo-syntax. | |
| D6 | Every console script is documented. `dev-notes/` and `handoffs/` are dated scratch, exempt from the 200-line cap. | auto |
| K6 | A `nanounet/cli/*.py` module with `main` and no `__main__` guard. | auto |
| K7 | `main()` calls `nano_header` and `config_table`, and emits `next:`. | auto |
| K8 | A `nanounet.<module>` path in user docs or `scripts/*.sh` must be a real module. | auto |
| K9 | `docs/dev-notes/` and `docs/handoffs/` open with `Date:` and `Status:`. | auto |

## Red flags: stop and rewrite

- A `BaseX` with subclasses, or a registry dict of classes, when there are only two cases.
- A new file under 30 lines, or a file climbing past 180 lines without a split plan.
- `raise ValueError("invalid input")`. Which input? What was expected? How do you fix it?
- A `try/except` that logs and continues, or that "falls back" to recomputing missing data.
- `.item()` or `.cpu()` inside `forward` or `training_step`, or any host-side work per step.
- A flag with no `help=`, a table row reading "TODO", or a doc naming a command that no longer exists.
- The phrase "should be faster" with no before/after number.
- An output line only a human can parse, or a command that waits for keyboard input.

## Review checklist (paste into the report)

- [ ] `check.py --changed` reports 0 errors, and no new warns on lines I touched.
- [ ] Every touched file is under 200 LOC, with a *why*-docstring. No new abstraction that an `if` or a function could replace.
- [ ] The command has a header, config table, and summary with `next:`. `--help` is clean. Machine output follows U9/U10.
- [ ] Every new failure names the problem, the expectation, and the fix, and fires at startup where possible.
- [ ] Data path, step, or loss changed: the before/after throughput table is included.
- [ ] Flag, output, or log format changed: the step doc and argument table were updated in this change.
