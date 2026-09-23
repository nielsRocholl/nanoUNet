# Docs: small, structured, never stale

Load this when editing `docs/`, `README.md`, or any change that adds, renames, or removes a CLI flag, command, output path,
or log line. Rule IDs refer to SKILL.md. `scripts/check.py` enforces D3, D4, and D6 mechanically.

## Tree

```
docs/
├── index.md       pipeline overview: mermaid flow, quickstart sequence, links into steps/   (D1)
├── steps/         one file per pipeline stage, user-facing                                  (D2)
│   └── preprocess, plan, valset, pretrain, train, predict, track
├── reference/     config fields, losses, patch size, instance targets, track ids
├── dev-notes/     plans, investigations, experiment logs: dated scratch, not user-facing  (D6)
└── handoffs/      session handoffs and decision records: dated scratch                   (D6)
```

User docs are `index.md`, `steps/`, `reference/`, and `README.md`. The 200-line cap and the staleness checks apply to them.
`dev-notes/` and `handoffs/` are exempt, but they must open with a date and a status line (see the log format below).

## Step doc template (D2): this exact section order

````markdown
# Preprocess

Fingerprint the raw dataset, run the ResEnc planner, and resample cases to blosc2 (`3d_fullres`).
(≤3 lines: what it does, what it needs, what it produces.)

## Command

```bash
nanounet_preprocess -d 501 --planner nnUNetPlannerResEncL -np 8
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d, --dataset_id` | int+ | required | Dataset id(s), e.g. 501. Several ids merge into `--merged-id`. |
| `-np, --num_processes` | int | 8 | Worker processes for resampling |

## Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `$NANOUNET_RAW/Dataset501_*/imagesTr/*.nii.gz` | NIfTI | you |
| `$NANOUNET_PREPROCESSED/Dataset501_*/nnUNetResEncUNetLPlans.json` | JSON | this step |

## Common errors

| Message starts with | Fix |
|---|---|
| `No preprocessing plan at` | `nanounet_preprocess -d 501` first |
````

- **D3.** Every argparse flag in `nanounet/cli/` appears in some argument table, in backticks, with its spelling exactly as in code.
  The checker matches literally, so `--val-frac` in code must be `` `--val-frac` `` in the doc.
- **D4.** A removed or renamed flag or command is deleted from the docs *in the same change*. The checker flags any
  backticked `--flag` in a table row that no CLI defines, and any `nanounet_*` command that isn't in `pyproject.toml`.
- **D5.** Use real placeholders (`-d 501`, `$NANOUNET_RESULTS/...`), not `<dataset>`, whenever a literal works.
- **Common errors** tables quote the first words of the real error message, so a user can grep for it. When you add an E1 error,
  add a row.

## Experiment log format (dev-notes): borrowed from nanochat `dev/LOG.md` (example numbers are illustrative)

One entry per experiment. Record negative results with the same detail as positive ones: they stop the next agent
from re-running a dead idea.

```markdown
## 2026-09-23: bf16-mixed vs 16-mixed on H200 (Negative Result)

**Hypothesis:** bf16 drops GradScaler overhead, so it should give ~5% faster epochs at the same dice.
**Change:** `--precision bf16-mixed`, everything else fixed (d900, fold 0, bucket l, bs 8, 250 it/ep).
**Result:**

| arm | epoch_wall_time_sec (med e1-e3) | val_dice @e50 |
|---|---|---|
| 16-mixed | 212.4 | 0.712 |
| bf16-mixed | 210.9 (-0.7%) | 0.709 |

**Verdict:** not adopted. The gain is within noise. Revisit if torch.compile lands.
```

Titles take a suffix when useful: `(Negative Result)`, `(Reverted)`, or `(commit abc1234)`. Every entry ends with a verdict:
**adopted / not adopted / reverted / open**.

## Handoff format (handoffs/)

Open with `Date:`, `Status:` (in progress / blocked / done), `Branch/commit:`, then: what was done, what is
verified (with commands and numbers), what is next (literal commands), and open questions. Keep it disposable: once the
work lands, fold anything durable into `steps/` or `reference/` and leave the handoff as history.
