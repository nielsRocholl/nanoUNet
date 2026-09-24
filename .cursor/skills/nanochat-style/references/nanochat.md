# nanochat evidence digest

Source: github.com/karpathy/nanochat @ `92d63d4` (2026-07-03), ~15k lines read in full. Load this to justify or
challenge a rule, or when the user asks "how does nanochat do X?". To re-verify, run
`git clone --depth 1 https://github.com/karpathy/nanochat` and grep the cited lines.

## Philosophy (README)

- "there are no giant configuration objects, model factories, or if-then-else monsters" (README:201). This is the source of R3 and R8.
- **Cognitive complexity counts as cost**, alongside dollars. A PR must work for *every* `--depth`, not only one size (README:108).
  A gain is weighed against bloat, per the "gnarly or … significantly bloats the code" clause (LEADERBOARD.md:51).
- **Anti-magic:** autocast is rejected as "magic we don't control" (LOG.md:78). It's replaced by one global `COMPUTE_DTYPE`, detected
  once. This is the source of R17.
- **AI policy:** disclose substantial LLM-written parts that you don't fully understand (README:203).
- **Single dial:** `--depth` is the only knob, and width, heads, LR, horizon, and batch size are derived from it (batch size via a
  power law anchored at depth 12, LOG.md:223-269). The nanoUNet analogue is the planner + `--dl-bucket` presets.
  Prefer a derived default (`-1`/`None` = auto, with the derivation printed) over a new required flag.

## Code facts

| Fact | Evidence |
|---|---|
| Every library file has a docstring, sized to how tricky the file is (3 to 70 lines) | `checkpoint_manager.py:1-3`, `fp8.py:1-70` |
| Files follow concepts, not a size cap: gpt 555, optim 459, engine 352, common 328 LOC | `wc -l nanochat/*.py` |
| No ABCs. The one base class is `tasks/common.py:Task` (`raise NotImplementedError`, **7** subclasses including `TaskMixture` `:129` and `TaskSequence` `:164`) | `tasks/common.py:85-164` |
| Dispatch by `if/elif` on a string, with `else: raise` | `optim.py:441-446` |
| **40** asserts vs 11 raises. Actual-vs-expected f-strings are the exception (2 of ~34 in engine/core_eval/tasks: `mmlu.py:52`, `arc.py:45`); most messages are static | `engine.py:128,143` |
| Capability fallbacks resolved once at import: FA3→SDPA, bf16→fp32 on pre-SM80, each with a reason string | `flash_attention.py:49-71`, `common.py:17-32` |
| Null-object `DummyWandb` instead of `if use_wandb:` everywhere | `common.py:216-223` |
| Shape comments everywhere: `# (B, T, H, D)` | `gpt.py:88`, `optim.py:25-27` |
| Checkpoint = `model_{step:06d}.pt` + `meta_{step:06d}.json` + per-rank optim shards. Old checkpoints are patched by `_patch_missing_keys`, not by a migration system | `checkpoint_manager.py:22-58` |

## Script / UX facts

| Fact | Evidence |
|---|---|
| Training scripts are module-level procedural with no `main()`. `parser.error` is at `base_eval.py:144`. `chat_eval.py:178` is guard-only. 6 of 9 scripts are fully flat | `base_eval.py:144` |
| Flags are kebab-case (56/56). `#` headers only in `base_train.py:42-79`, `chat_sft.py:36-66`, `chat_rl.py:33-59`. `infer_bench.py:105` `-t/--temperature` has no `help=`. `-1` and `None` both mean auto | `base_train.py:56-77`, `chat_sft.py:47-53` |
| `user_config = vars(args).copy()` goes to wandb *and* into checkpoint meta | `base_train.py:81,487` |
| Two status-line shapes: `base_train.py:567` logs `bf16_mfu`; `chat_sft.py:469` logs `mfu` | `base_train.py:567`, `chat_sft.py:469` |
| Shell sweeps grep those lines. `miniseries.sh:85` greps `Number of parameters:`, which `base_train.py:253` never prints (`Parameter counts:`). `scaling_laws.sh:101-106` is fine | `runs/miniseries.sh:85` |
| Pretty table is hand-aligned f-strings (`infer_bench.py:202-226`). The last-line JSON at `:241` is printed **unconditionally** (no `--json` flag) | `scripts/infer_bench.py:241` |
| Inherited or derived values are printed one per line: `Inherited max_seq_len=2048 from pretrained checkpoint` | `chat_sft.py:97-115` |
| One canonical end-to-end script (`runs/speedrun.sh`) is the living quickstart | `runs/speedrun.sh` |

## Performance facts (gpu.md builds on these)

| Technique | Evidence |
|---|---|
| Pinned CPU buffer, one contiguous `non_blocking` H2D copy per step, allocated once | `dataloader.py:111-120,160` |
| First batch prefetched before the loop; next batch fetched *inside* the grad-accum loop | `base_train.py:333,518` |
| `synchronize()` only around the timer. `.item()` once per step, commented as a sync point | `base_train.py:88,508-544,541` |
| `gc.collect(); gc.freeze(); gc.disable()` after the first step (GC was causing ~500ms stalls) | `base_train.py:586-594` |
| `torch.compile(model, dynamic=False)` for training; the uncompiled `orig_model` for eval/sampling/saving | `base_train.py:245-246` |
| Fused, compiled optimizer steps; 0-D CPU tensors for LR so changing values don't recompile | `optim.py:23,111,262-272` |
| Meta-device init + `to_empty` + `load_state_dict(assign=True)` | `gpt.py:157-162`, `checkpoint_manager.py:99-104` |
| Vocab padded to a multiple of 64 for tensor cores | `gpt.py:168-172` |
| MFU computed from a peak-FLOPs table per GPU and logged every step | `common.py:232-279`, `base_train.py:554` |
| Benchmarks: a warmup call before every timed measurement (cublas autotune, allocator, kernels) | `infer_bench.py:185,206-209` |
| `@torch.inference_mode()` on all generation paths | `gpt.py:526`, `engine.py:140,175` |

## Project practice

| Practice | Evidence | nanoUNet |
|---|---|---|
| `dev/LOG.md`: dated entries, free prose, every entry ends in a verdict. Only `:1077` uses `**Hypothesis:**`. Suffixes `(negative)` / `(Reverted)` | `dev/LOG.md` (34 entries) | adopted for `docs/dev-notes/` (docs.md) |
| `LEADERBOARD.md`: wall-clock to a quality target is the headline metric, every record pinned to a commit and command, noise quantified by repeated runs | `dev/LEADERBOARD.md` | spirit adopted in G4 (median of epochs, fixed config) |
| **Permanent** small `tests/`: hermetic, testing code-path equivalence (FA3 vs SDPA), determinism, and named regressions; GPU tests `skipif` | `tests/*.py` (6 files, incl. `test_attention_fallback.py`) | **deviation:** R16 keeps tests temporary |
| Deps: minimal, only `torch` hard-pinned, `uv` with cpu/gpu extras | `pyproject.toml:7-69` | similar floors, no uv extras |
| Agent skill format: `name` + `description` frontmatter, short numbered imperative steps | `.claude/skills/read-arxiv-paper/SKILL.md` | this skill follows it, plus references/ and a checker |

## Known nanochat inconsistencies (don't copy them)

- `checkpoint_manager.py:18-20` re-implements a rank-0 `log0` instead of reusing `print0`.
- `chat_rl.py` mixes a bare `print` (`:323`) with `print0` calls.
- `base_train.py:339` has an assert with no message for a user-facing flag combination. That case should be an E1 error. Same class: `base_train.py:409`, `chat_sft.py:123`.
- Import-time side effects are load-bearing: `common.py:32,68-69` (COMPUTE_DTYPE, logging). `flash_attention.py:23-50` downloads a kernel at import under a broad `except` (`:45`).
- Three output mechanisms: `print0`, bare `print` (`dataset.py`, `tokenizer.py:138`), `logger.info` (`checkpoint_manager.py`).
- Determinism is split: global `torch.manual_seed(42)` (`common.py:184-190`; `use_deterministic_algorithms` is commented out) plus a local `Generator` per sampling call (`engine.py:182-183`, `gpt.py:536-539`).
- Tests use `torch.equal` for determinism (`test_optim.py:64-77`, `test_engine.py:201-223`) and `allclose` only across implementations (`test_attention_fallback.py:39-45`).
- Deletions are logged with numbers: `LOG.md:72-105`, `:1075-1089`.
- Reuse is cross-script import, not a shared module: `chat_sft.py:26`, `base_train.py:36`.
- Ckpt quirks stay, with a comment: `gpt.py:58-59` "kept for checkpoint compatibility".
- Don't copy: duplicated dispatch `core_eval.py:184-194` and `:232-239`; `all_reduce` without a barrier `loss_eval.py:54-58` vs `core_eval.py:256-259`; broad swallow `engine.py:41-44`; `input()` at `chat_cli.py:53`.
