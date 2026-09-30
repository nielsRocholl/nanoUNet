# CLI, output, errors: for humans and agents

Load this when touching any `<project>/cli/`, any terminal output, or any error message. Rule IDs refer to SKILL.md.

Two audiences, one command. **Humans** read rich output on **stderr**: the one `Console(stderr=True)` in `core/ui.py`, shared by every project.
**Agents and scripts** read **stdout** and the exit code. Rich is on stderr, so stdout stays clean for machines.
Keep it that way.

## Helpers that already exist (`core/ui.py`, plus env helpers in `nanounet/common.py`): use them, don't reinvent

| Helper | Use |
|---|---|
| `nano_header(title, color="cyan")` | Opening panel: command, dataset, key mode (U2) |
| `nano_banner(title, subtitle)` | Big centred panel. Only for long-running entry points (train/pretrain). |
| `config_table(rows, title="config")` | `rows = [(arg, value, "cli"/"config"/"default"), ...]` (U3) |
| `arg_rows(ap, args)` | Rows for every parsed flag, source `cli`/`default`, for commands with no config file (U3) |
| `nano_progress(total, desc)` | Context manager yielding `advance(n)`. Transient, and degrades cleanly without a TTY (U4). |
| `cprint(msg)` | Every other line. Rich markup is allowed: `[green]`, `[bold]`, `[dim]`. |
| `nano_rule()` | Dim separator between phases |
| `console()` | The singleton, for `rich.Table` / `Panel` renderables: `cprint(table)` also works |
| `quiet_lightning_runtime()` | Silences Lightning/torch banners and warnings (U5) |
| `raw_dir()` / `preprocessed_dir()` / `results_dir()` | Env-derived roots. They raise with a `Fix:` when the env var is unset. |

If you need a new shared output helper (a summary block, a JSON emitter), add it to `core/ui.py`, keep it
under 15 lines, and use it from ≥2 commands in the same change.

## Command anatomy (R13)

```python
def main() -> None:
    args = build_parser().parse_args()
    validate(args)                                   # E3/E6: every problem, one SystemExit, before any heavy import/IO
    nano_header(f"nanoUNet predict  {ds}  fold {args.fold}")
    config_table(config_rows(args))                  # U3
    t0 = time.perf_counter()
    with nano_progress(len(cases), "predict") as advance:
        for c in cases:
            ...; advance()
    cprint(f"[green]done[/green] {len(cases)} cases → {out}  ({time.perf_counter() - t0:.1f}s)")
    cprint(f"next: segtrack_run -m $NANOUNET_RESULTS/nanounet/<run> ...")           # U2: literal, copy-pasteable
    if args.json:                                                          # U9: last stdout line
        print(json.dumps({"status": "ok", "outputs": [out], "n_cases": len(cases), "seconds": round(time.perf_counter() - t0, 1)}))
```

The `print(json.dumps(...))` is the **one** sanctioned stdout write. Waive it inline: `# nanochat-style: allow R11 (U9 json line)`.

## Machine contract (U8–U12)

| Aspect | Contract |
|---|---|
| Flags | kebab-case (`--val-frac`). Every flag has `help=` with units, and says whether `-1`/`None` means auto. Short aliases only for the very common ones (`-d`, `-f`, `-i`, `-o`). Legacy snake flags (`--dataset_id`) stay for compatibility. Don't add new ones. |
| `--json` | A single-line JSON object as the **last stdout line**: `status`, `outputs` (paths), key metrics, and `next` (command string). Numbers must be JSON-safe: `None`, not `inf`/`nan`. nanochat's `infer_bench.py` sets the precedent. |
| Exit code | `0` success. `1` user error via `raise SystemExit(msg)`, which Python prints to stderr and exits with code 1, no traceback. `2` argparse usage. Any traceback is a bug in *our* code. |
| Non-interactive | No `input()`, no "are you sure". Destructive or overwriting actions refuse by default and name the flag: `Fix: re-run with --force (old file is backed up)`. See `cli/build_splits.py`. |
| Dry run | Long commands (train, pretrain, predict over a folder) should support `--dry-run`: validate, print the config table, exit 0. It's the cheapest way for an agent to check a command line. |
| Status lines | `key: value \| key: value`, fixed keys, fixed number formats (`loss: 0.1234`, `dt: 512ms`, `epoch: 012/1000`). Slurm scripts and agents grep them, so treat a format change as an API change. |
| Verbosity | Quiet by default. `--verbose` restores third-party logs. Never gate *errors* behind `--verbose`. |

**Status today:** no command implements `--json`, `--dry-run`, or `--verbose` yet. `quiet_lightning_runtime()` runs
unconditionally at import in `cli/train.py` and `cli/pretrain.py`. Add these features when you touch a command. Don't
retrofit every command in one sweeping diff.

## Errors (E1–E6)

User boundaries are CLI args, config files, files on disk, env vars, and hardware. Errors there use the 3-line template:

```python
raise SystemExit(
    f"No preprocessing plan at {plan_path}.\n"                                   # what is wrong (exact value, quoted)
    f"Expected output of the plan step for dataset {dataset_id}.\n"               # what was expected / where we looked
    f"Fix: nanounet_preprocess -d {dataset_id}   (see nanounet/docs/steps/preprocess.md)"  # literal command + doc
)
```

- **CLI code** uses `SystemExit(msg)`: a clean message with exit code 1. **Library code** called from a CLI raises
  `FileNotFoundError`/`ValueError` with the same 3 lines. The CLI then shows a short traceback, which is acceptable for
  library-level failures, but prefer validating in the CLI first (E3).
- **E6, collect then fail.** Validation appends to `problems: list[str]` and raises once:
  `raise SystemExit("\n\n".join(problems))`. An agent fixes 5 flags in one round-trip, not 5.
- **Invalid choice:** list the valid values and the closest match:
  `--loss 'dice'` → `Expected one of: dc_ce, cc_dc_ce. Did you mean dc_ce?` (`difflib.get_close_matches`).
- **Internal invariant:** use `assert cond, f"actual vs expected"` (E2), never a `Fix:` line. The user can't fix our bug.
- **Swallowing (E4):** a broad `except` with `pass` or `continue` is banned. A narrow `except OSError: pass` is allowed only for
  best-effort side effects (fadvise hints, temp cleanup, a stat race during a purge) and needs
  `# nanochat-style: allow E4 (why)`. Never on the path that produces a result.
- **Warnings** that change results don't exist. They are errors. A warning is only for "slower than it
  could be" (R17), printed once at startup.

## Output hygiene (U6, U7)

- Tables for anything with more than 2 columns or more than 3 rows (per-case results, per-cohort splits, timing breakdowns).
- One line per case in batch loops, at most: `[3/40] case_id  12.1s  dice: 0.83`.
- No output inside dataloader workers. No per-step prints in training (Lightning/wandb own that).
- Numbers: fixed precision per quantity. Seconds `.1f`, loss `.4f`, dice `.3f`, percentages `.1f`, counts with `,`.
