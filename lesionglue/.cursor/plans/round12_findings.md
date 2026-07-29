# Round 12 — Findings log

Running record of measured results. Written as each stage completes.
Plan: `round12_measurement_features_data.md`.

---

## Stage A.1 — Repairs and accounting (2026-07-29)

### A.1.1 `configs/base.json` (D5) — FIXED

`git mv configs/r9_base.json configs/base.json`. `scripts/round9.sh` and
`scripts/lesion-round9-cv.sh` both default to `configs/base.json` and now resolve. Verified
`load_config('configs/base.json')` succeeds; `CKPT_MONITOR = val_match_score_ema`.

Remaining `r9_base.json` mentions are confined to historical plan files (`round-10.md`) and are left
as written — they are a record of what was true at the time.

### A.1.2 The 240-vs-224 gap — EXPLAINED, and larger than the plan assumed

Cached graphs vs `data_split.json`:

| split | in split | cached | missing |
|---|---:|---:|---:|
| train | 240 | **224** | 16 |
| val | 30 | **28** | 2 |
| test | 30 | **29** | 1 |

(The plan quoted 225 for `cv/train`; the true count is 224. `final/train` holds 225 — a different
split scheme.)

**Reasons for the 19 missing patients:**

| reason | n | meaning |
|---|---:|---|
| `EMPTY_FU` | 14 | every baseline lesion disappeared — **complete responders** |
| `EMPTY_BL` | 2 | `cog_propagated` is `None` for *all* rows — **registration failure** |
| both sides non-empty | 3 | dropped by the dominant-FU filter (see A.1.3) |

Missing patient IDs, for the record:
- `EMPTY_FU` (14): `06eb133bbf`, `0777d5c17d`, `0aa1883c64`, `19f3cd308f`, `5878a7ab84`,
  `a3c65c2974`, `b73ce398c3`, `d296c101da`, `d947bf06a8`, `e56954b4f6`, `fc22130974` (train);
  `6883966fd8`, `cfa0860e83` (val); `07e1cd7dca` (test).
- `EMPTY_BL` (2): `06eb61b839` (8 UNCHANGED + 1 NEWLYAPPEARING rows, none with `cog_propagated`),
  `8e98d81f82` (3 UNCHANGED + 8 DISAPPEARED, none with `cog_propagated`).
- dominant-FU casualties (3): `432aca3a1e`, `c6f057b865`, `eb160de1de`.

**On `EMPTY_BL`:** these are *not* genuinely empty baselines. `06eb61b839` has 8 UNCHANGED rows —
real, annotated correspondences — but `cog_propagated` is missing for every one, so
`_node_rows` drops them all (`graph.py:84-86`). This is registration failure feeding directly into
data loss, and it is the same mechanism as **D3** (registration-imputed positions). Counted here;
handled in the Stage A.3 audit.

**On `EMPTY_FU`:** the matcher cannot represent a graph with zero FU nodes (no cross edges, no
Sinkhorn support), so these are structurally excluded. Two consequences worth stating:
1. The deployed tracker has **no defined behaviour for a complete responder** — a real clinical
   case. Documented as a known limitation; fixing it needs matcher changes that are out of scope
   for R12 (§10).
2. Excluding them is *conservative* for the reported metric, not inflationary: with zero FU
   distractors, every BL lesion is trivially "disappeared", so including them would have **raised**
   `val_acc_disappeared`. No correction to past numbers is needed.

### A.1.3 NEW FINDING — `_dom_fu` silently discards 10% of all transitions

`tracking/data/graph.py:57-62` picks the single most common `img_id_fu` per patient, and line 100
(`rows = [r for r in rows if r.img_id_fu == dom]`) throws away every other transition. **The pipeline
builds at most one graph per patient**, but the dataset contains more:

| quantity | count |
|---|---:|
| patients with meta | 300 |
| `(pid, img_id_fu)` transitions in the data | **335** |
| graphs built (1 per patient, capped) | 300 attempted / **281 succeed** |
| transitions discarded by `_dom_fu` | **35 (10.4%)** |
| transitions that are **viable** (both sides non-empty) | **306** |
| viable transitions currently **unused** | **25** (23 train / 1 val / 1 test) |
| lesion rows discarded | 107 of 4,503 (2.4%) |

**So the usable graph count is 306, not 281 — a free +8.9% of training data**, in a problem whose
one historically successful lever (R6) was a data lever. This was not visible in any prior round.

The plan (§2 A.1.3) authorizes fixing dropped-for-fixable-reason cases. Scheduled as **Stage A.5**
below, deliberately *after* the Stage B selector bake-off so the two effects are not confounded.

**Leakage guard (mandatory when implemented):** a patient contributing 2 graphs must have **both**
graphs in the same CV fold. `tracking/data/splits.py::fold_map` keys on patient id, so this holds
provided the datamodule filters by `pid` and not by index — **verify explicitly, do not assume.**

### A.1.4 Corrections to the plan's own numbers

- Plan said `cv/train` = 225; actual = **224** (`final/train` = 225).
- Plan said "15 missing"; actual = **16 train, 19 total** across splits.
- Plan said 359 BL→FU transitions (from the initial scout); recomputed after `linking_unclear`
  filtering and split membership, the figure is **335** total / **306** viable. The 359 figure counted
  `(pid, img_id_bl, img_id_fu)` triples before row filtering.

---

## Stage A.2 / A.3 — provenance + audit (2026-07-29)

New files: `tracking/data/provenance.py` (79 LOC), `tracking/cli/audit.py` (146 LOC).
`tracking/data/meta.py` gained a `cog_backpropagated` field (2-line additive change).

### A.3.1 `clickfix_report.csv` — real schema

Header: `case,status,n_lesions,n_flagged,n_sanity_bad,n_bl_filled,n_fu_filled`.
**`case` is `{pid}_{img_id_fu:02d}`** — one row per BL→FU transition, **not** per patient
(307 rows / 292 unique pids). Keying by bare pid silently collapses cases. Totals reconcile with
D3 exactly: `n_bl_filled` 678, `n_fu_filled` 1364, `n_sanity_bad` 137, 20 rows `status=="partial"`.

### A.3.2 Class balance and imputation, per split

| | train | val | test |
|---|---|---|---|
| UNCHANGED | 2026 | 200 | 277 |
| DISAPPEARED | 1074 | 97 | 104 |
| NEWLYAPPEARING | 492 | 19 | 48 |
| MERGED | 147 | 8 | 11 |
| **native SPLIT** | **0** | **0** | **0** |
| BL "imputed" rate | 12.41% | 5.86% | 10.91% |
| FU "imputed" rate | 28.30% | 29.50% | 22.94% |
| `sanity_bad` patients | 18/240 | 1/30 | 0/30 |
| `volume_bl` quartiles (mm³) | 115 / 357 / 1497 | 125 / 406 / 1650 | 83 / 210 / 988 |
| `volume_fu` quartiles (mm³) | 16 / 203 / 1150 | 0 / 284 / 2524 | 21 / 123 / 539 |

**D2 re-confirmed independently: zero native SPLIT in every split.**

### A.3.3 **GATE A AS SPECIFIED IS UNDEFINED — the plan was wrong, and here is why**

The plan's rule ("imputed if `cog_bl`/`cog_fu` is empty") **never fires for a node the model sees**.
Verified directly: of **3,815 graph-eligible BL nodes, 0 lack `cog_bl`**; every graph-eligible FU node
has `cog_fu`. The imputed stratum has **n = 0**, so `acc_observed − acc_imputed` has no denominator.

What `n_bl_filled`/`n_fu_filled` actually count (reconstructed, matching 291/307 and 298/307 cases
exactly) is registration-projected *guess* coordinates handed to the **missing** side of a row — a
NEWLYAPPEARING lesion's absent BL position, or a DISAPPEARED lesion's absent FU position — purely for
QC bookkeeping. **Those guesses never become graph nodes.**

**The real registration dependence is elsewhere, and it is uniform rather than stratified:**
- `graph.py:123` — the descriptor is computed at `cog_bl`, the **real annotated** centroid.
- `graph.py:124` — but `bl.pos` is `cog_propagated * sp_fu`, i.e. **every BL node's position is a
  registration projection**, without exception.

So registration error is a *global* property of the BL geometry channel, not a per-node attribute.
That is why a per-node observed/imputed split cannot exist — and it is the same conclusion the R11
critique reached from the other direction ("registration error is pervasive, not localized").

### A.3.4 Gate A, reformulated so it is answerable

Stratify by **case-level registration quality**, which does vary across patients. Two independent
sources, now both available:

1. `derivatives/registration_error_table.json` → `excluded.{backend}.case_level_failure_patients`
   — a named list of patients where registration failed: **26** for backend `original`, **45** for
   `unigradicon`. The same file carries measured per-lesion offsets binned by lesion size (the
   empirical basis for `PROP_SIGMA` in `tracking/common.py`).
2. `clickfix_report.csv` → `status != "ok"` or `n_sanity_bad > 0` — **19** patients.

Union, restricted to patients that actually have cached graphs:

| split | flagged / cached | % |
|---|---|---|
| train | 33 / 224 | 15% |
| val | 5 / 28 | 18% |
| test | 2 / 29 | 7% |

**Revised Gate A:** compare pooled out-of-fold per-patient `match_score` on the **33 flagged** vs
**191 clean** training-pool patients, with the patient bootstrap from Stage B.2.
- If flagged patients are materially worse → the geometry channel is carrying registration error into
  predictions; report a registration-clean subset alongside the full number, and geometry modelling
  stays dead (as R11 already found empirically).
- If not → registration quality is not the binding constraint and the ceiling is elsewhere.

**Note the dependency this creates:** the global val split alone (28 patients, 5 flagged) is far too
small to answer this. Only the **pooled out-of-fold** estimator from Stage B.2 makes Gate A
answerable at all — so B.2 is now a prerequisite for Gate A, not an independent nicety.

---

## Stage A.5 (scheduled) — recover the 25 unused viable transitions

Deferred until after Stage B.3 to avoid confounding the selector measurement with a data change.
Requires: `build_hetero_data` to return a *list* of graphs (one per viable transition) rather than
one; `dataset.py` to flatten; fold assignment to stay keyed on `pid`. Cache tag bump.
