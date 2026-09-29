# Round 10 — Expert Critique + Alternative Plan

Reviewer: external expert pass on the agent's "Assignment-Anchored Match Refinement" plan.
Goal restated: **a model with significantly better performance**, where the real ceiling is
`val_acc_unchanged_split ≈ 0.90` (same-anatomy, co-located, same-type decoy disambiguation).

No code was changed for this document. It is analysis + a redirected plan.

---

## TL;DR

The R10 block is **safe but aimed slightly off the bullseye**, and most of its capacity is
**redundant with parts already in the model**. Only ~1 of its 3 "new" mechanisms is genuinely
novel, and that one (single-shot assignment anchoring) is the weakest form of the SuperGlue idea
on graphs this small. Its own predicted failure mode ("gate ≈ 0, R10 ≈ R9, flat") has a high
prior. We are spending a full paired 5-fold sweep on a likely null.

**The better lever is the user's own instinct**: a **registration-invariant, second-order
(pairwise-distance) consistency signal**. It (a) is the principled answer to the
registration-error pain point, (b) targets the *actual* failure mode (co-located same-type
decoys), and (c) is a signal **genuinely absent** from the current model. It reuses the R10
scaffold (zero-init gate, paired CV, safety) but swaps the weak signal for a strong one.

---

## Part 1 — Critique of the R10 plan

### What is good and should be kept
- **Zero-init gated residual** ⇒ exact R9 at init, cannot regress at step 0. Correct R7-avoidance.
- **Paired 5-fold CV + explicit decision rule.** Right response to the 30-patient noise floor.
- **The framing premise is true**: the GNN never sees the current soft assignment. Adding an
  assignment-conditioned signal is legitimately non-redundant *in principle*.

### Where it is weak (grounded in the code)

1. **Two of three "new axes" are redundant with existing components.**
   - `edge_bias = Linear(cross_attr → heads)` is the *same* conditioning the GNN already does:
     `TransformerConv(..., edge_dim=CROSS_DIM)` at [matcher.py:70](tracking/matcher.py:70). This is
     precisely the R7 mistake (more edge-conditioned attention on top of edge-conditioned attention).
   - `score(m_bl * m_fu)` duplicates the existing bilinear identity term `self.bilin`
     ([matcher.py:94](tracking/matcher.py:94)). Two bilinear pair terms now.
   - So only **assignment anchoring** (`assign_w * logp`) is truly new. The block carries a lot of
     redundant capacity — on 240 patients that is overfit surface, even when gated.

2. **The novel signal is the weakest form of SuperGlue, at the wrong scale.**
   SuperGlue's assignment-anchoring pays off with hundreds–thousands of keypoints and **iterated**
   re-assignment between layers. Here graphs are tiny (≈46×40 max, often <10 nodes) and the anchor
   is **single-shot** (`logp` from `pair0.detach()`, [matcher.py:145](tracking/matcher.py:145)),
   feeding **one** block. With 3 FU candidates, "attend to my current best candidate" adds little
   that the bilinear head doesn't already encode. The mechanism shines at scale; here it is marginal.

3. **It does not target the actual failure mode.** `unchanged_split = 0.90` is co-located,
   same-type decoys (two lung mets compete for one BL). What breaks that tie is either appearance
   (already in nodes + `desc_cos`/`desc_l2`) or **relative geometric configuration** — *which* BL
   sits where relative to its neighbors, and whether the surrounding constellation is preserved.
   Assignment anchoring only adds soft winner-take-all competition; it does **not** add the
   surrounding-configuration signal that actually resolves the tie. That signal is absent from the
   model entirely.

4. **High prior on a null by the plan's own logic.** The plan lists "gate stays ~0 ⇒ R10 ≈ R9" as
   the *designed-in safe failure*. Combined with (1)–(3), that is the **most likely** outcome. A
   full paired CV sweep for a predicted null is poor EV.

5. **It picks the architecture lever to dodge a cache rebuild, not because it is highest-EV.**
   The plan explicitly defers the geometry-feature levers (v6 registration uncertainty; synthetic
   decoy augmentation) because they touch the cache — yet *those* target the real ceiling. R6's win
   was a **data** lever (node-drop aug). On 240 patients the ceiling-mover is far more likely to be
   data/feature than another gated block.

### Minor correctness notes (if R10 ships anyway)
- `_refine_logp` recomputes a full Sinkhorn separate from `_dust_graph`'s pass. Plan acknowledges;
  share one pass if profiled.
- `logp` aligns to the row-major dense `edge_index` — verified the convention matches
  `dense_pair_index` ([pairs.py:11](tracking/data/pairs.py:11)). Good, but keep the alignment test.

**Verdict on R10:** safe, well-engineered, low expected gain. Don't burn the sweep on it as-is.
Redirect the same scaffold to the signal below.

---

## Part 2 — The registration pain point, answered

### The trap in the naive Chamfer idea
You cannot compute Chamfer (or any Euclidean) distance between a **BL point** and an **FU point**
without a transform between the two frames — `cog_bl` lives in BL space, `cog_fu` in FU space.
Naive cross-space Chamfer **does not escape registration**; it silently *requires* it. Today the
code applies that transform as `cog_propagated` and bakes its error into:
- BL node positions ([graph.py:124](tracking/data/graph.py:124)),
- the **BL intra-kNN graph topology** ([graph.py:143](tracking/data/graph.py:143)),
- and every geometric cross feature `dp / dist / log dist` ([pairs.py:29](tracking/data/pairs.py:29)).

Registration error is therefore *pervasive*, not localized.

### What IS registration-free: intra-cloud geometry
A rigid (and approximately, affine) registration **preserves pairwise distances and angles within
a cloud**. So these quantities need **no** registration and carry real matching signal:
- `D_BL[i,i'] = ‖cog_bl[i] − cog_bl[i']‖` in **native BL mm space**.
- `D_FU[j,j'] = ‖cog_fu[j] − cog_fu[j']‖` in **native FU mm space**.

The correct reframe of your gut feeling: **don't match points across spaces — require the
*matching* to preserve each cloud's internal geometry.** If BL lesions `i,i'` are 40 mm apart and
their true FU matches `j,j'` are also ≈40 mm apart, that is a registration-free constraint. This is
classic **graph matching / quadratic assignment** (Leordeanu–Hebert spectral matching), and it is
exactly the surrounding-configuration signal missing in Part 1, point 3.

`cog_bl` and the BL spacing `sb` are already loaded ([graph.py:112,122](tracking/data/graph.py:112)),
so native-space distances are cheap — **no cache rebuild** for the distance matrices.

### Honest limits (so we complement, not replace)
- **Degeneracy at small n.** 2–3 lesions give too few distance constraints; the term adds little.
- **Symmetry ambiguity.** Symmetric constellations have reflection/relabel ambiguities that pure
  distances cannot break.
- **Non-rigid change.** Real growth/shrinkage and deformable anatomy mean distances are only
  *approximately* preserved — use a soft, robust penalty, not a hard equality.

Conclusion: the registration prior is still **useful to break ties** — so keep it as a **soft
prior**, not a hard coordinate. The new term is a complement to appearance + a downweighted
registration cue.

---

## Part 3 — Alternative plan (reuses the R10 scaffold)

Same safety envelope as R10 (zero-init gate, paired 5-fold CV, single test gate). Swap the weak
signal for the registration-free second-order one. Three pieces, increasing ambition; ship A+B,
defer C.

### Piece A — native-space BL intra graph (registration-free, ~5 LOC, no cache rebuild)
- Build BL `intra_knn` on **`cog_bl * sb`** (native BL mm), not on propagated positions.
- Effect: the GNN's intra message passing now encodes each cloud's *internal* geometry cleanly;
  removes registration error from half the graph topology. Keep a separate `pos_prop` for the
  cross features so nothing else breaks.
- This alone is a clean, near-free ablation worth its own CV cell.

### Piece B — distance-consistency attention bias (the real lever)
Replace the R10 block's redundant `edge_bias(cross_attr)` with a **registration-invariant
quadratic-assignment bias**, using the current soft assignment `P` (already computed in the R10
scaffold) and the two native distance matrices:

```
bias(i, j) = − Σ_{i'≠i} P(i' → j')-weighted  | D_BL[i,i'] − D_FU[j,j'] |   (robust, e.g. Huber)
```

i.e. BL `i` matching FU `j` is *encouraged* when, for `i`'s neighbors `i'`, their currently
preferred FU `j'` sits at an FU-distance consistent with the BL-distance `D_BL[i,i']`. Fold this as
the attention/logit bias in place of `assign_w * logp` alone. Properties:
- **Registration-free** (only intra-cloud distances + soft assignment).
- **Directly resolves co-located same-type decoys**: the decoy that breaks the surrounding
  constellation is penalized — exactly the 0.90 ceiling case.
- **Genuinely new**: no first-order or second-order distance-consistency term exists in the model.
- Keep zero-init gate ⇒ starts == R9, cannot regress.

This is the differentiable, soft, single-pass version of spectral graph matching, anchored on the
Sinkhorn assignment the model already has.

### Piece C — registration as soft prior only (defer)
- Downweight or replace the absolute `dp/dist` cross features with registration **uncertainty**
  features (the deferred v6 `sigma_mm`, `Mahalanobis`) so the network learns to *trust geometry
  less where registration is unreliable*. Needs a cache rebuild → defer behind A+B results.

### Evaluation (same rigor as R10)
Paired 5-fold CV, additive cells to isolate signal:
1. `R9-base` (refine_blocks=0).
2. `A` (native BL intra only).
3. `A + B` (native intra + distance-consistency bias).
4. (defer) `A + B + C`.

Ship rule unchanged (Fabian): 5-fold mean `val_match_score_ema` clears base with non-overlapping
mean±std, `val_acc_unchanged_split` ≥ +2 pp, disappeared/newly do not regress. Then tau sweep the
winner; single test-split gate.

### Why this beats R10 on expected value
| | R10 (anchoring block) | Alternative (A+B) |
|---|---|---|
| Targets `unchanged_split` ceiling | Indirectly (competition only) | **Directly** (constellation consistency) |
| Novel vs. existing model | ~1/3 (rest redundant) | **Fully new signal** |
| Registration-error answer | No (still uses laden cross_attr) | **Yes — the principled fix** |
| Most likely outcome | Flat (gate→0), predicted null | Real shot at the ceiling |
| Cost | Full sweep | Same scaffold, same sweep |
| Cache rebuild | No | No (A+B); C deferred |

---

## Recommendation
1. **Do not run R10 as the headline experiment.** Keep its scaffold (zero-init gate, CV harness).
2. **Implement Piece A + Piece B** on that scaffold; run the additive paired 5-fold CV.
3. Hold **Piece C** (registration-uncertainty cache v6) as the next pull if A+B is flat.
4. If you want one cheap sanity ablation first: **Piece A alone** — native-space BL intra graph —
   is ~5 LOC and tests whether removing registration error from the graph topology already moves
   the needle.
