# Add "centered" inference mode to nanoUNet

## Context

nanoUNet has one inference mode today: **clustered**. It greedily packs nearby
clickpoints into shared patches (each patch placed to cover a cluster bbox),
encodes *all* clicks visible in that patch, runs the net, and gaussian-merges
the logits. Optional per-cluster border-expansion BFS grows large lesions past
the patch edge.

We want a **second mode, "centered"**, to test a hypothesis: *the model predicts
better when each lesion is presented centered in its patch.* In centered mode we
run **one patch per clickpoint, centered on that click**, and encode **only that
single click** as the prompt (not other visible clicks). Border expansion must
still work. All downstream machinery (batched forward, TTA, gaussian
accumulation, border BFS, export) is identical to clustered and is reused
verbatim.

The two modes differ in exactly one place: **how seed patches and their prompt
points are constructed**. Per nanochat style R3 ("two cases is not a registry"),
this is an `if/else`, not a new class/factory. The change stays inside the
existing files and under the 200-LOC limit (R1).

**Style is a hard rule.** Obey `.cursor/rules/nanochat-style.mdc`: no new
abstractions, no ABCs/factories/registries, `if cfg==a/else`, comments explain
*why* not *what*, `assert` on invariants, raise at boundaries, <200 LOC/file.

## Design (one decision the user made)

- **Prompt scope = only the centered click.** Each centered patch encodes a
  single-element point list `[p_i]`. Reuses `_encode_row` unchanged: pass the
  cluster `[p_i]`; `cluster_prompts_patch_local` returns it in patch-local coords
  (always inside, since the patch is centered on it). If border-expansion moves
  the patch so `p_i` is no longer inside, the existing fallback
  `local_prompt_points_for_patch(cl[0], ...)` puts the prompt at the patch
  center — same behavior as clustered.
- **Dedup identical patches.** Skip a seed whose `(clamped_slice_key, click)`
  pair was already added. Note: because each seed carries its own single-click
  prompt, two seeds are truly identical only when the input JSON has duplicate
  click coordinates that clamp to the same slice. Cheap + robust; keep it, but it
  won't save much. Cross-seed border-expansion patches are **not** deduped (they
  carry different prompts → genuinely different) — keep the existing per-seed
  `vis` set in the border loop.

## Files to change

### 1. `nanounet/infer/predict_case.py` (currently 181 LOC — target ~195)

Add a `mode` parameter and branch the seed-construction block. Everything else
(rows build, batch forward, gaussian accumulate, border BFS, safe-divide, unpad)
is **unchanged** apart from two variable renames.

**a. Add `mode` to the signature** (after `cluster_margin_frac`):
```python
    cluster_margin_frac: float = 0.1,
    mode: str = "clustered",
) -> torch.Tensor:
```
And assert the invariant near the top of the body (R5/R15):
```python
    assert mode in ("clustered", "centered")
```

**b. Replace the seed-construction block** (current lines 120-123):
```python
    clusters = cluster_points_for_patch_size(pts_pad, patch_size, cluster_margin_frac)
    cluster_slices = [
        spatial_slices_covering_points(cl, patch_size, padded_shape) for cl in clusters
    ]
```
with:
```python
    if mode == "clustered":
        seeds_pts = cluster_points_for_patch_size(pts_pad, patch_size, cluster_margin_frac)
        seed_slices = [
            spatial_slices_covering_points(cl, patch_size, padded_shape) for cl in seeds_pts
        ]
    else:  # centered: one patch per click, prompt = that single click only
        seen: set = set()
        seeds_pts, seed_slices = [], []
        for p in pts_pad:
            sl = centered_spatial_slices_at_point(p[0], p[1], p[2], patch_size, padded_shape)
            key = (spatial_slices_to_tuple(*sl), p)
            if key in seen:  # exact-duplicate input clicks only
                continue
            seen.add(key)
            seeds_pts.append([p])
            seed_slices.append(sl)
```

**c. Rename the two locals throughout the rest of the function**
(`clusters` → `seeds_pts`, `cluster_slices` → `seed_slices`). Affected lines:
- L126 `for cl, (sz, sy, sx) in zip(clusters, cluster_slices):`
- L141 `for i, (sz, sy, sx) in enumerate(cluster_slices):`
- L146 `for i, cl in enumerate(clusters):`
- L147 `sz, sy, sx = cluster_slices[i]`

No other change to the border-expansion block — it already iterates
`seeds_pts`/`seed_slices` and uses `cl` (now `[p_i]` in centered mode) for prompt
encoding, exactly what we want.

**Imports**: `centered_spatial_slices_at_point` and `spatial_slices_to_tuple` are
**already imported** (lines 13-20). `cluster_points_for_patch_size` /
`spatial_slices_covering_points` stay (still used by the clustered branch). No
import changes needed.

### 2. `nanounet/cli/predict.py`

**a. Add the flag** (next to the other inference args, after line 62
`--cluster-margin-frac`):
```python
    ap.add_argument("--inference-mode", choices=("clustered", "centered"), default="clustered",
                    help="patch placement: 'clustered' packs clicks, 'centered' = one patch per click")
```

**b. Pass it through** in the `predict_case_logits(...)` call inside
`gpu_export` (currently lines 126-133): add `mode=args.inference_mode,` to the
kwargs. `--cluster-margin-frac` is simply ignored in centered mode (harmless).

### 3. `README.md` (inference section, ~lines 165-215)

Add one line documenting `--inference-mode {clustered,centered}` and a single
example invocation:
```bash
# centered: one patch per click, each prompt = its own click (lesion-centered)
nanounet_predict -i case.nii.gz -o seg.nii.gz --points case.json \
  -m /path/to/run --ckpt last.ckpt --inference-mode centered --border-expand
```

## Why this is fast / robust / efficient

- **Zero new compute paths.** Centered reuses the exact batched-forward + TTA +
  gaussian-merge + border-BFS engine; only seed construction differs. No risk of
  a divergent, under-tested code path.
- **Batching unchanged.** Seeds are mini-batched by `--batch-size` regardless of
  mode, so an H200 and a 3080ti both saturate via the same knob. More clicks →
  more seeds → more batches; tune `--batch-size`/`--num-workers` per node.
- **Accumulation dtype** still honors `NANOUNET_SINGLE_PATCH_ACCUM_DTYPE`
  (fp16 accumulation on big GPUs) — set it for memory-tight nodes.
- **Dedup** removes exact-duplicate work; gaussian `n_pred` normalization already
  guarantees correctness under arbitrary patch overlap, so even without dedup the
  output is well-defined.
- **No fallbacks (R12).** Empty `points_xyz` still returns the all-background
  logits early-return (shared, unchanged). Missing points JSON still raises at
  the CLI boundary.

## Verification (end-to-end, then delete any scratch)

1. **LOC check** (R1): `wc -l nanounet/infer/predict_case.py` must be `< 200`.
2. **Import/smoke**: `python -c "from nanounet.infer.predict_case import predict_case_logits"`.
3. **Clustered unchanged (regression)**: run a single case in default mode and
   confirm the output `.nii.gz` is byte-identical to a pre-change run on the same
   case (or that the seg matches a saved baseline). The renames + new branch must
   not alter clustered output.
   ```bash
   nanounet_predict -i case.nii.gz -o /tmp/clustered.nii.gz --points case.json \
     -m /path/to/run --ckpt last.ckpt
   ```
4. **Centered runs**: same case with `--inference-mode centered`; confirm it
   produces a plausible seg and logs per-case timing.
   ```bash
   nanounet_predict -i case.nii.gz -o /tmp/centered.nii.gz --points case.json \
     -m /path/to/run --ckpt last.ckpt --inference-mode centered
   ```
5. **Centered + border expansion**: add `--border-expand`; confirm it completes
   and large lesions grow past patch edges.
6. **Multi-click sanity**: pick a case with several nearby clicks; confirm
   centered produces (deduped) one seed per distinct click and the merged seg is
   coherent where patches overlap.
7. **Device matrix (optional)**: rerun centered with `--device cpu` and (if
   available) cuda to confirm both paths execute.

No new tests committed (R16) — validate, then remove scratch files.