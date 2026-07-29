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

## Stage A.5 (scheduled) — recover the 25 unused viable transitions

Deferred until after Stage B.3 to avoid confounding the selector measurement with a data change.
Requires: `build_hetero_data` to return a *list* of graphs (one per viable transition) rather than
one; `dataset.py` to flatten; fold assignment to stay keyed on `pid`. Cache tag bump.
