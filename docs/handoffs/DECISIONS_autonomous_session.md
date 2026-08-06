# Decisions taken without the human — review these

The human authorised autonomous work to completion, on the condition that every choice normally
reserved for them is recorded here with the reasoning, for review afterwards.

Each entry: what was decided, why, how to reverse it, and how much it matters.

---

## D-A1 — Cohort names are bare prefixes (`d013`), not underscored (`d013_`)

**Decision.** `sampling.cohorts` keys are `"d013"`, not `"d013_"` as the parent handoff's example
showed.

**Why.** `cohort_of()` in `nanounet/plan/splits.py` splits a case id on the first underscore and
already returns `d013`. Accepting the underscored form would need a second normalisation path for
no benefit, and the same function is what the split builder and the val manifest already use — one
spelling everywhere.

**Reversible.** Yes, trivially: it is a config key. An unknown name raises at startup naming the
available cohorts, so a stale `"d013_"` fails loudly rather than silently sampling nothing.

**Stakes.** Low. Cosmetic, caught at startup.

**Note.** `--only-prefix` still takes the underscored form. They are different flags; the train doc
says so explicitly.

---

## D-A2 — Click dropout rate, rebalanced against boundary clipping

**Pending — filled in once C7 finishes.** The human asked for the total suppression signal to land
near 20% rather than 20% plus whatever boundary clipping adds on top.

---

## D-A3 — Cohort weights for the training run

**Pending.** The human explicitly parked this "until the last step before the training run". It is
now that step, so a choice has to be made to proceed; it is recorded here rather than assumed.

---

## Standing constraints honoured

- `configs/default.json` untouched, so the 600-epoch baseline stays reproducible.
- No permanent `tests/` folder (R16); verification scripts are written, run, reported, deleted.
- Every file under 200 LOC except `fit.py`, which the human has explicitly allowed to run over.
- Explicit paths staged, never `git add -A` (this caused a mixed commit last session).
