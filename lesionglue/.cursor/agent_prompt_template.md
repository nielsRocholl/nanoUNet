# New-agent prompt template — lesion tracking

Copy ONE block below into a fresh agent session. Block A = continue an in-flight round
(a handoff exists). Block B = start a brand-new development round (research + new plan).
Fill the `<...>` placeholders. Keep "caveman mode" if you want terse output.

---

## Block A — continue current round (USE THIS FIRST; a handoff exists)

```
caveman mode

This repo is our graph-based lesion tracking algorithm, under active development.

Read `.cursor/HANDOFF.md` first — it is the source of truth for where we left off.
Then read the active plan it points to in `.cursor/plans/`, and the coding philosophy
in `.cursor/rules/nanochat-style.mdc` (HARD RULE, do not assume you already know it).

Context you must internalise before acting:
- Run everything from the repo root with `PYTHONPATH=.`; the interpreter is `python3`
  (there is no `python` in this container).
- Batch-size-independence (step clock + EMA-by-update-count) is a keeper from R8 — never reintroduce
  batch-size dependence; it is what makes our W&B curves comparable across changes.
- Metrics come from patient-level 5-fold CV (mean±std), monitored on `val_match_score_ema`.
  Judge changes on separated-vs-overlapping CIs, NOT single-split point estimates.
- Peaks, not finals: there is heavy overfitting; the best value is not the last value.
- The held-out test split is the final gate — touch it ONCE, only via the RUN_FINAL stage.

Your task: <e.g. run scripts/round9.sh sweep and report the CV ranking with mean±std> /
<continue the next pending todo in the plan>. Confirm the current state from the handoff
before changing anything; do not redo completed work. Obey nanochat-style in every file you touch.
```

---

## Block B — start a new development round (research a new plan)

```
caveman mode

This repo is our graph-based lesion tracking algorithm, under active development.

Task 1 — understand the current algorithm deeply: its switches and knobs (data/features,
matcher architecture, losses, Sinkhorn/dustbin, augmentation, EMA, decode, CV harness).
Start from `.cursor/HANDOFF.md` and the latest plan in `.cursor/plans/`, then read the code.

Task 2 — read the plan files in `.cursor/plans/` (rounds 2..N). Carry-overs that MUST stay:
batch-size-independence (R8) and patient-level k-fold CV with EMA-best monitoring (R9).
Treat unproven knobs skeptically — prefer ablation over assumption.

Task 3 — use the wandb MCP server (project `hyper-alignment/lesion-tracking`) to inspect
the curves for the recent experiments. Look at PEAKS not finals (overfitting is heavy, so the
best values are NOT the last values). Sample enough history points (sparse logging — use a high
samples count). Note where `val_acc_disappeared`/`newly_appearing` decay and where
`val_acc_unchanged_split` plateaus.

Then: identify current failure points, what to roll back, and what to introduce to make this the
best-performing lesion tracker for the current problem statement. Write it up as the next round
plan in `.cursor/plans/`.

For the plan, reason as a panel of: Ilya Sutskever (representation/regularization in the small-data
regime), Jure Leskovec (GNN/matching structure), Fabian Isensee (medical validation rigor).
All three agree: measurement before modeling.

HARD RULE — embed the coding philosophy from `.cursor/rules/nanochat-style.mdc` directly into the
plan as a "Coding philosophy" section; do not assume the implementing agent already knows it.
```

---

## Notes for whoever writes the prompt
- Always point at `.cursor/HANDOFF.md` — update it at the end of every session.
- Keep the two carry-overs explicit (batch-size-independence, CV+EMA-best); agents tend to "simplify" them away.
- "peaks not finals" and "test touched once" are the two mistakes new agents make most.
- The plan must inline the nanochat rules; a link is not enough (agents skip linked rules under load).
