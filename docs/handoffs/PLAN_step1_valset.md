# Plan — Step 1+2: Fixed stratified validation set & layered metrics

**Audience:** a coding agent with no access to the conversation that produced this plan. Everything
needed is here. Read it fully before writing code.

**Job in one sentence:** replace the randomly re-drawn per-epoch validation set with a fixed,
seeded, offline manifest of 1500 patches across 4 prompt scenarios, and report metrics per scenario
and per source dataset.

**Parent document:** `docs/handoffs/HANDOFF_training_overhaul.md`. That doc's §1 and §2 are
authoritative for *why*; this doc is authoritative for *what to type*. Where they disagree on
detail, this doc wins — it was written after the human resolved the open decisions.

**Style:** `.claude/skills/nanochat-style` is binding. <200 LOC per file, no `utils/`, no ABCs,
no factories, rich CLI output, errors that name the fix, no defensive try/except.

---

## 0. Decisions already made — do not reopen

| # | Decision | Rationale |
|---|---|---|
| D1 | **One split**, not K-fold. `splits_final.json` becomes a 1-element list. | Only fold 0 is ever trained; a full run takes days. |
| D2 | Val = **15% of cases, applied within each source dataset**. | The old 5-fold split was dataset-blind and drifted 13–25% per cohort. |
| D3 | Val patch count = **1500, fixed**, listed in an offline manifest. | ~600 random patches/epoch is too noisy to read per-stratum curves. |
| D4 | **4 mutually exclusive scenarios**: `all_clicked` 600, `lesion_free_decoy` 375, `subset_clicked` 300, `none_clicked` 225. | Every patch is exactly one scenario. |
| D5 | **Per-dataset floor of 40 patches**, remainder proportional. | Without it MSD_Lung gets 17 patches and ±0.087 Dice error bars. |
| D6 | Per-dataset reporting = **2 metrics only** (`val_dice`, `val_prompt_agreement`). | 17 datasets × 9 metrics is unreadable. |
| D7 | Headline aggregates are **weighted by true dataset proportions**, cancelling the floor's skew. | The floor must not distort the global number. |
| D8 | Size (small/large) and click-inside/outside are **tags**, not scenarios. | A patch can be `all_clicked` *and* small *and* click-missed. |
| D9 | Old `splits_final.json` is backed up, never silently overwritten. | It is on persistent storage and is the only record of the old split. |

### Decisions made while writing this plan (stated, not hidden)

| # | Decision | Rationale |
|---|---|---|
| D10 | `none_clicked` and `lesion_free_decoy` are scored by **predicted foreground fraction**, not Dice. | Correct output is empty; Dice against an empty target is undefined. This is exactly what the existing `val_fp` measures. |
| D11 | Aggregate `val_dice` keeps its **formula** byte-identical but its **composition changes**, so it is not comparable to the historical 0.8030. `val/all_clicked/val_dice` is the closest analogue. | Composition change is the point of the step. Formula change would break `pooled_fg_dice`. |
| D12 | Subset targets are **precomputed offline into a packed-bit sidecar** (~92 MB), so there is **zero `cc3d` on the validation path**. | The parent handoff §2 forbids per-patch connected components on any hot path. |
| D13 | Both prompt draws (for `val_prompt_agreement`) are **stored in the manifest**. No RNG at validation time at all. | Makes two runs on the same manifest bit-identical, which is an acceptance criterion. |

---

## 1. Verified facts about the current code

Read out of the source on 2026-08-05. Re-confirm line numbers before editing; treat semantics as
established.

| Fact | Location |
|---|---|
| `val_transforms` is `RemoveLabelTansform(-1→0)` + optional deep-supervision downsample. **No spatial transform.** | `nanounet/data/augment.py` |
| Therefore a precomputed target aligns voxel-for-voxel with the crop. This is what makes D12 possible. | — |
| `crop_patch` pads seg outside the volume with `-1`, which `RemoveLabelTansform` turns into 0. | `nanounet/data/patch_bbox.py:64` |
| `build_iter_dataloader` is a thin `DataLoader` wrapper and works unchanged for a map-style dataset. | `nanounet/dataloader_prefs.py:76` |
| `fold_keys(splits, 0)` indexes `splits[0]`, so a 1-element list is valid. | `nanounet/plan/splits.py:43` |
| `load_or_create_splits` returns the file verbatim if it exists — writing a new file is enough to switch. | `nanounet/plan/splits.py:31` |
| Val forces `fg_patch_prob = 1 - no_lesion_frac` and `false_pos_probability = 1.0`. | `nanounet/train/data_module.py:75-80` |
| Val already runs 3 forward passes per batch: normal, prompt-ablated, second-prompt. | `nanounet/train/lightning_module.py:126-151` |
| `val_step_row` pools `tp/fp/fn` over the batch with `.sum(0)`, losing per-row identity. | `nanounet/model/dice_helpers.py:33-39` |
| Centroid sidecars carry `centroids_zyx`, `bboxes_zyx`, `seed_zyx`, `volume_vox` for **every** case (verified: 200/200 sampled). | `nanounet/prompt/centroids.py:79-99` |
| Patch size is `[96, 160, 160]`; batch size 6. | `nnUNetResEncUNetLPlans_h200_smallpv.json` |
| Dataset999_Merged has **5866** cases across **17** prefixes (`d010`…`d027`, no `d026`). | verified on disk |
| `d026` is absent deliberately — the raw dataset was corrupted. Do not "fix" this. | human, confirmed |

### File LOC budget — several targets are at the ceiling

| File | LOC now | Budget |
|---|---|---|
| `nanounet/train/fit.py` | 211 | **already over — do not touch** |
| `nanounet/train/lightning_module.py` | 199 | must **shrink**; see §7 |
| `nanounet/train/data_module.py` | 192 | ≤ 8 new lines |
| `nanounet/train/patch_iterable.py` | 188 | **do not touch** |
| `nanounet/train/patch_render.py` | 124 | ≤ 20 new lines |
| `nanounet/model/dice_helpers.py` | 115 | ≤ 15 new lines |
| `nanounet/plan/splits.py` | 47 | ample |

Everything else goes in new files.

---

## 2. What gets built

| # | Path | Kind | Purpose |
|---|---|---|---|
| 1 | `nanounet/plan/splits.py` | edit | `make_balanced_split`, fold validation |
| 2 | `nanounet/cli/build_splits.py` | **new** | writes the balanced 15% split |
| 3 | `nanounet/data/valset.py` | **new** | manifest schema, loader, map-style dataset |
| 4 | `nanounet/cli/build_valset.py` | **new** | offline manifest builder |
| 5 | `nanounet/train/patch_render.py` | edit | carry meta through collate |
| 6 | `nanounet/model/dice_helpers.py` | edit | per-row tp/fp/fn |
| 7 | `nanounet/train/val_metrics.py` | **new** | all bucketing + logging |
| 8 | `nanounet/train/lightning_module.py` | edit | buffer meta, delegate logging |
| 9 | `nanounet/train/data_module.py` | edit | use the manifest when given |
| 10 | `nanounet/cli/train_parser.py` | edit | `--val-manifest` |
| 11 | `pyproject.toml` | edit | 2 new console scripts |
| 12 | `docs/steps/valset.md` | **new** | step doc + argument tables |
| 13 | `docs/steps/train.md` | edit | document `--val-manifest` |

Implement in this order. Items 1–2 are independently testable, 3–4 likewise, 5–9 are one wiring
change, 10–13 finish it.

---

## 3. Item 1–2 — the balanced split

### 3.1 `nanounet/plan/splits.py` — add two functions

Append to the existing file. Do not modify `make_splits`, `load_or_create_splits`, or `parse_fold`.

```python
def cohort_of(identifier: str) -> str:
    """Source-dataset prefix of a merged-pool case id: 'd010_CECT_P0001_ct_C1' -> 'd010'."""
    return identifier.split("_")[0]


def make_balanced_split(identifiers: List[str], val_frac: float, seed: int) -> List[dict]:
    """One train/val split with `val_frac` applied WITHIN each source dataset.

    The plain KFold above is dataset-blind: on the 17-cohort merged pool it drifted to 13-25% val
    per cohort, so small cohorts got val sets too small to plot. Returns a ONE-element list so it
    stays format-compatible with splits_final.json; fold 0 is the only valid fold."""
    assert 0.0 < val_frac < 1.0, val_frac
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for cid in sorted(identifiers):
        groups.setdefault(cohort_of(cid), []).append(cid)
    train: list[str] = []
    val: list[str] = []
    for _, ids in sorted(groups.items()):
        perm = rng.permutation(len(ids))
        n_val = int(round(len(ids) * val_frac))
        n_val = min(max(n_val, 1), len(ids) - 1)  # every cohort appears in BOTH sides
        val += [ids[i] for i in perm[:n_val]]
        train += [ids[i] for i in perm[n_val:]]
    return [{"train": sorted(train), "val": sorted(val)}]
```

Then harden `fold_keys` so a stale `--fold 3` fails at startup instead of raising `IndexError`
mid-run. Replace the existing body:

```python
def fold_keys(splits: List[dict], fold: int | str) -> tuple[list[str], list[str]]:
    if fold == ALL_FOLD:
        ids = sorted(splits[0]["train"] + splits[0]["val"])
        return ids, ids
    if not 0 <= fold < len(splits):
        raise IndexError(
            f"--fold {fold} but splits_final.json holds {len(splits)} split(s) (valid: 0"
            f"{'' if len(splits) == 1 else f'-{len(splits) - 1}'}).\n"
            f"This dataset uses a single balanced split (see docs/steps/valset.md).\n"
            f"Fix: pass --fold 0"
        )
    return splits[fold]["train"], splits[fold]["val"]
```

`import numpy as np` is already at the top of the file.

### 3.2 `nanounet/cli/build_splits.py` — new, target ~90 LOC

```python
"""Write a single train/val split balanced within each source dataset.

Replaces the dataset-blind 5-fold splits_final.json on merged pools: with 17 source datasets a
plain KFold drifts to 13-25% val per cohort, so small cohorts get unplottable val sets. The old
file is backed up, never silently overwritten."""
```

Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d, --dataset_id` | int | required | Dataset ID (e.g. 999) |
| `--plans` | str | required | Plans identifier, no `.json` |
| `--val-frac` | float | 0.15 | Validation share, applied per source dataset |
| `--seed` | int | 12345 | RNG seed, recorded in the printed table |
| `--force` | flag | off | Required to overwrite an existing `splits_final.json` |

Procedure, top to bottom:

1. `ds = convert_id_to_dataset_name(args.dataset_id)` (from `nanounet.plan.dataset_id`).
2. `nano_header(f"nanoUNet build-splits  {ds}  val_frac {args.val_frac}", color="green")`.
3. `pp = preprocessed_dir()`; `pm = Plans(join(pp, ds, args.plans + ".json"))`;
   `cm = pm.get_configuration("3d_fullres")`; `case_dir = join(pp, ds, cm.data_identifier)`.
4. `all_ids = Blosc2Folder.get_identifiers(case_dir)`. Then apply the **same** `numTraining`
   truncation the datamodule uses, so the key list is identical:
   `dj = load_json(join(raw_dir(), ds, "dataset.json"))`; `ntr = dj.get("numTraining")`;
   `ids = all_ids[:int(ntr)] if ntr is not None else list(all_ids)`.
5. `out = join(pp, ds, "splits_final.json")`. If it exists and not `--force`:
   ```python
   raise FileExistsError(
       f"{out} already exists ({len(load_json(out))} split(s)).\n"
       f"Overwriting changes which cases are held out and invalidates every existing val curve.\n"
       f"Fix: re-run with --force (the old file is backed up automatically)"
   )
   ```
6. If `--force` and the file exists, copy it to
   `join(pp, ds, f"splits_final.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")` and `cprint` the
   backup path.
7. `splits = make_balanced_split(ids, args.val_frac, args.seed)`.
8. Print a rich `Table` (import `Table` from `rich.table`, render with the console via `cprint`, or
   follow whatever pattern `config_table` uses in `nanounet/common.py:151`) with columns
   `dataset | total | train | val | val %`, one row per cohort, sorted by total descending, plus a
   bold `TOTAL` row.
9. Write `out` with `json.dump(splits, f)`.
10. Closing summary via `cprint`:
    ```
    wrote <out>  (1 split, <ntrain> train / <nval> val)
    next: nanounet_build_valset -d <id> --plans <plans> --config <cfg>
    ```

**Acceptance:**
- Every cohort's val share is `0.15 ± 1 case` of its own size.
- Every cohort appears in both `train` and `val`.
- `train` and `val` are disjoint and together equal `ids`.
- Re-running with the same seed produces a byte-identical file.

---

## 4. Item 3 — `nanounet/data/valset.py`

Target ~180 LOC. Module docstring:

```python
"""Fixed validation manifest: schema, load, and a deterministic map-style patch dataset.

The training val loader re-samples patches every epoch, so per-scenario curves drown in resampling
noise. This module reads a manifest built offline by nanounet_build_valset instead: every patch is
pinned by (case, bbox, click coordinates), both prompt draws are stored, and nothing is randomised
at validation time -- two runs on one manifest give bit-identical metrics.

Scenario targets: all_clicked and subset_clicked score against seg; none_clicked and
lesion_free_decoy score predicted-foreground fraction (correct output is empty, so Dice is
undefined). subset_clicked additionally carries a precomputed clicked-subset target, packed to
bits in a sidecar .npz -- that is what keeps cc3d off the validation path entirely."""
```

### 4.1 Constants and schema

```python
SCHEMA_VERSION = 1
SCENARIOS = ("all_clicked", "subset_clicked", "none_clicked", "lesion_free_decoy")
SIZE_BUCKETS = ("small", "large")
SMALL_LESION_MAX_VOX = 500  # ~10mm diameter at the plans spacing; see docs/steps/valset.md
```

Manifest JSON layout (one file, UTF-8, `json.dump(..., indent=None)`):

```json
{
  "schema": 1,
  "dataset": "Dataset999_Merged",
  "plans": "nnUNetResEncUNetLPlans_h200_smallpv",
  "config_path": "/nanoUNet/configs/default.json",
  "seed": 1234,
  "patch_size": [96, 160, 160],
  "small_lesion_max_vox": 500,
  "scenario_counts": {"all_clicked": 600, "lesion_free_decoy": 375,
                      "subset_clicked": 300, "none_clicked": 225},
  "cohort_weights": {"d010": 0.1839, "d011": 0.0336, "...": 0.0},
  "entries": [ ... ]
}
```

`cohort_weights` are the **true** case-count proportions over the whole dataset (train+val), used
by D7 to undo the floor's skew. They must sum to 1.0 within 1e-6.

One entry:

```json
{
  "case": "d010_CECT_P0001_ct_C1",
  "cohort": "d010",
  "scenario": "subset_clicked",
  "bbox": [[12, 108], [64, 224], [80, 240]],
  "clicks_zyx":  [[40, 71, 88]],
  "clicks2_zyx": [[43, 66, 91]],
  "n_false_pos": 0,
  "size_bucket": "large",
  "click_inside": 1,
  "subset_target_index": 17
}
```

| Field | Meaning |
|---|---|
| `bbox` | `[[z0,z1],[y0,y1],[x0,x1]]`, half-open, in the preprocessed case frame. May be negative or exceed the volume — `crop_patch` pads. |
| `clicks_zyx` | **Patch-local** integer coordinates, post-displacement, already filtered into the patch. Draw 1. |
| `clicks2_zyx` | Draw 2 — same lesions, independently displaced, **same decoy**. Feeds `val_prompt_agreement`. |
| `n_false_pos` | How many trailing entries of each click list are decoys. `click_inside_flags` relies on decoys being last. |
| `size_bucket` | `small` if the largest lesion **present in the patch** has `volume_vox <= SMALL_LESION_MAX_VOX`, else `large`. `"large"` for lesion-free patches (never read). |
| `click_inside` | Precomputed 1/0/-1, same semantics as `patch_render.click_inside_flags`, for draw 1. |
| `subset_target_index` | Row index into the `.npz` sidecar, or `-1` when the scenario has no subset target. |

### 4.2 Sidecar

Path: manifest path with `.json` replaced by `.targets.npz`. Written with
`np.savez_compressed(path, packed=arr, shape=np.array(patch_size))` where `arr` has dtype `uint8`
and shape `(n_subset_entries, prod(patch_size) // 8)`, produced by `np.packbits(mask.ravel())`.

At 300 entries × 96·160·160 bits this is ~92 MB uncompressed and far less compressed.

### 4.3 Loading

```python
@dataclass(frozen=True)
class ValManifest:
    path: str
    header: dict
    entries: list[dict]
    packed: np.ndarray | None   # (n, nbytes) uint8, or None when no subset entries
    patch_size: tuple[int, int, int]


def load_manifest(path: str) -> ValManifest:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No validation manifest at {path}.\n"
            f"Expected the output of nanounet_build_valset.\n"
            f"Fix: nanounet_build_valset -d 999 --plans <plans> --config configs/default.json "
            f"--out {path}   (see docs/steps/valset.md)"
        )
    header = load_json(path)
    if header.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema {header.get('schema')}, this build expects {SCHEMA_VERSION}.\n"
            f"Fix: rebuild it with nanounet_build_valset"
        )
    ...
```

Load the `.npz` only if any entry has `subset_target_index >= 0`; if such entries exist and the
sidecar is missing, raise with the rebuild command. No fallback (R12).

### 4.4 `ValPatchDataset`

Map-style `torch.utils.data.Dataset`. Constructor takes
`(manifest: ValManifest, case_folder: str, roi_cfg: RoiPromptConfig, val_tf, final_patch_size, longi: bool)`.

`__len__` returns `len(manifest.entries)`.

`__getitem__(i)` — no RNG anywhere in this method:

```python
e = self.manifest.entries[i]
with self.ds.open_case(e["case"], need_seg=True) as (data, seg, _, _):
    data_crop, seg_crop, _shape, _pslc = crop_patch(data, seg, e["bbox"])
im = torch.from_numpy(data_crop.astype(np.float32)).float()
se = torch.from_numpy(seg_crop.astype(np.int16)).short()

# Both draws ride ONE augmentation pass, exactly as PatchIterable does, so the pair differs only
# in click placement. Build variant dicts in the shape split_variant_keypoints expects.
variants = [
    {"points_pos": np.asarray(e["clicks_zyx"], np.float32).reshape(-1, 3),
     "points_neg": np.zeros((0, 3), np.float32),
     "n_false_pos": e["n_false_pos"]},
    {"points_pos": np.asarray(e["clicks2_zyx"], np.float32).reshape(-1, 3),
     "points_neg": np.zeros((0, 3), np.float32),
     "n_false_pos": e["n_false_pos"]},
]
kp = concat_variant_keypoints(variants, self.longi)
with torch.no_grad():
    o = self.tf(**{"image": im, "segmentation": se, "keypoints": kp})
    entries = split_variant_keypoints(o["keypoints"], variants, self.longi)
    v1 = render_variant(o, entries[0], {"null_baseline": False}, self.longi, self.final_ps, self.pr)
    v2 = render_variant(o, entries[1], {"null_baseline": False}, self.longi, self.final_ps, self.pr)

item = {
    "data_variants": [v1],
    "data_prompt2": v2,
    "target": o["segmentation"],
    "click_inside": [e["click_inside"]],
    "scenario": SCENARIOS.index(e["scenario"]),
    "cohort": self.cohort_index[e["cohort"]],
    "size_bucket": SIZE_BUCKETS.index(e["size_bucket"]),
}
if e["subset_target_index"] >= 0:
    bits = np.unpackbits(self.manifest.packed[e["subset_target_index"]])
    m = bits[: int(np.prod(self.patch_size))].reshape(self.patch_size)
    item["target_subset"] = torch.from_numpy(m.astype(np.int16))[None]
else:
    item["target_subset"] = torch.zeros((1, *self.patch_size), dtype=torch.int16)
item["has_subset"] = int(e["subset_target_index"] >= 0)
return item
```

`self.cohort_index` is `{cohort: i}` built once in `__init__` from
`sorted(manifest.header["cohort_weights"])`, so the integer↔name mapping is stable and shared with
`val_metrics.py` (which reads it back off the manifest header the same way).

`self.pr` is `roi_cfg.prompt`. `self.ds` is a `Blosc2Folder(case_folder, identifiers=<cases in manifest>)`.

**`target_subset` is always present** (zeros when unused) so `collate_patches` can stack
unconditionally — a batch mixing scenarios must not have ragged keys.

### 4.5 `build_val_dataloader`

Keeps `data_module.py` inside its 8-line budget.

```python
def build_val_dataloader(manifest, case_folder, roi_cfg, val_tf, final_ps, longi,
                         batch_size, bucket, pin_memory, persistent_workers) -> DataLoader:
    """Deterministic val loader over a fixed manifest. shuffle stays False and no sampler is
    passed: Lightning injects a DistributedSampler under DDP, and order does not affect any
    metric here because every bucket is pooled before reduction."""
```

Body calls `build_iter_dataloader(ds, batch_size=batch_size, bucket=bucket, nw=bucket.nw_val,
prefetch=bucket.prefetch_val, collate_fn=collate_patches, pin_memory=pin_memory,
worker_init_fn=worker_init if bucket.nw_val else None, persistent_workers=persistent_workers)`.

---

## 5. Item 4 — `nanounet/cli/build_valset.py`

Target ~190 LOC. If it will not fit, move the per-scenario patch search into
`nanounet/data/valset_build.py` — **do not** exceed 200 LOC in either file.

Module docstring:

```python
"""Offline build of the fixed validation manifest: 1500 patches over 4 prompt scenarios.

Everything expensive happens HERE, once, so validation stays pure tensor work: connected
components, the clicked-subset targets, both prompt draws, and the click-inside flags are all
resolved and written to disk. Nothing in nanounet/data/valset.py randomises or recomputes."""
```

Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-d, --dataset_id` | int | required | Dataset ID (e.g. 999) |
| `--plans` | str | required | Plans identifier, no `.json` |
| `--config` | str | required | ROI/prompt JSON (e.g. `configs/default.json`) |
| `--out` | str | required | Manifest output path (`.json`) |
| `--n-patches` | int | 1500 | Total validation patches |
| `--floor` | int | 40 | Minimum patches per source dataset |
| `--mix` | str | `0.40,0.25,0.20,0.15` | Scenario shares in the order `all_clicked, lesion_free_decoy, subset_clicked, none_clicked` |
| `--seed` | int | 1234 | RNG seed, recorded in the manifest |
| `--max-tries` | int | 60 | Rejection-sampling attempts per patch before giving up |

### 5.1 Budget allocation

```python
def allocate(cohort_val_counts: dict[str, int], n_patches: int, floor: int) -> dict[str, int]:
    """floor per cohort, remainder proportional to val-case count. Largest-remainder rounding so
    the totals add up exactly."""
```

Guard first, because it is a user error with an obvious fix:

```python
if floor * len(cohort_val_counts) > n_patches:
    raise ValueError(
        f"--floor {floor} x {len(cohort_val_counts)} cohorts = "
        f"{floor * len(cohort_val_counts)} > --n-patches {n_patches}.\n"
        f"Fix: lower --floor or raise --n-patches "
        f"(>= {floor * len(cohort_val_counts)})"
    )
```

Then split each cohort's allocation across the 4 scenarios using `--mix`, again with
largest-remainder rounding so per-cohort totals are exact.

### 5.2 Per-case preparation (once per case, cached)

```python
labels, n_inst = cc3d.connected_components(seg[0] > 0, connectivity=26, return_N=True)
```

`cc3d` is already a dependency (`nanounet/prompt/centroids.py:33` uses it). Match the
`connectivity` used there — read it and use the same value; a mismatch would make instance ids
disagree with the sidecar centroid order.

The sidecar `centroids_zyx` / `bboxes_zyx` / `volume_vox` are in `cc3d` label order (label `k+1`
is entry `k`). Assert this once per case:
`assert n_inst == len(props["centroids_zyx"]), (cid, n_inst, len(...))`.

### 5.3 Drawing one patch

Common geometry — identical to validation today, so patches are drawn from the same distribution:

```python
patch_size = np.array(cm.patch_size)
need_to_pad = np.zeros(3, dtype=int)   # val uses patch_size == final_patch_size
shape = np.array(case_spatial_shape(case_dir, cid))
```

Per scenario, with rejection sampling up to `--max-tries`:

| Scenario | bbox draw | Accept when | Clicks |
|---|---|---|---|
| `all_clicked` | `_sample_bbox(..., fg_patch_prob=1.0, ...)` | ≥1 instance centroid inside the patch | every in-patch centroid, displaced |
| `subset_clicked` | `_sample_bbox(..., fg_patch_prob=1.0, ...)` | **≥2** instance centroids inside the patch | strict subset: `k = rng.integers(1, n)` instances chosen without replacement, displaced |
| `none_clicked` | `_sample_bbox(..., fg_patch_prob=1.0, ...)` | ≥1 instance centroid inside the patch | **none** |
| `lesion_free_decoy` | `_sample_bbox(..., fg_patch_prob=0.0, ...)` | `seg_crop.max() <= 0` | exactly one decoy from `_sample_false_pos(seg_crop, rng)` |

"instance centroid inside the patch" means `filter_centroids_in_patch` returns it, evaluated on the
**undisplaced** centroids — acceptance must not depend on the random displacement, or the two draws
could accept different lesion sets.

Displacement uses the production model, so val clicks are as noisy as deployment:

```python
from nanounet.data.error_table import draw_propagated_offset
prop = cfg.sampling.propagated
d1 = [draw_propagated_offset(c, v, prop, rng1) for c, v in zip(chosen_cts, chosen_vols)]
d2 = [draw_propagated_offset(c, v, prop, rng2) for c, v in zip(chosen_cts, chosen_vols)]
```

`rng1` and `rng2` are two independent generators derived from `--seed`, e.g.
`np.random.default_rng(seed)` and `np.random.default_rng(seed + 777_777)`. Both draws use the
**same** `chosen_cts` and the **same** decoy — that is the whole point of the Step 0 fix, and
re-introducing a per-draw decoy would silently undo it.

Convert to patch-local: `filter_centroids_in_patch(displaced, pslc)` returns global coordinates
that fall inside; subtract `bbox_lbs` to get patch-local. Decoys from `_sample_false_pos` are
**already patch-local** — do not subtract twice. Append decoys last.

**Reject the patch if either draw ends up with zero lesion clicks** in a scenario that requires
clicks (`all_clicked`, `subset_clicked`) — displacement can push every click out of the patch, and
such a patch is silently a `none_clicked` patch. Count these and report the count in the summary.

If `--max-tries` is exhausted:

```python
raise RuntimeError(
    f"Could not fill scenario '{scenario}' for cohort {cohort}: needed {want}, got {have} "
    f"after {tries} attempts across {len(ids)} cases.\n"
    f"Most likely this cohort has too few multi-lesion cases for 'subset_clicked'.\n"
    f"Fix: lower its share with --mix, or raise --max-tries"
)
```

### 5.4 Subset target

For `subset_clicked` only:

```python
lab_crop = crop_and_pad_nd(labels[None], bbox, 0)[0]      # same bbox, pad with 0 = background
mask = np.isin(lab_crop, chosen_instance_ids).astype(np.uint8)
packed_rows.append(np.packbits(mask.ravel()))
```

`crop_and_pad_nd` comes from `acvl_utils.cropping_and_padding.bounding_boxes`, the same import
`patch_bbox.py` uses. Pad value **0**, not −1: this target never passes through
`RemoveLabelTansform`.

### 5.5 `click_inside` precomputation

Reuse the real function so the definition cannot drift:

```python
from nanounet.train.patch_render import click_inside_flags
entry = {"pp": torch.tensor(clicks_local, dtype=torch.float32).reshape(-1, 3),
         "pn": torch.zeros((0, 3)), "n_fp": n_fp}
flag = click_inside_flags([entry], torch.from_numpy(seg_crop.astype(np.int16)))[0]
```

### 5.6 Output

1. Write the `.npz` sidecar first, then the manifest — so a manifest on disk always has its
   targets.
2. Print a rich `Table`: rows = cohorts, columns = the 4 scenario counts + total.
3. Print a second `Table`: realised tag counts — `click_inside` 1/0/−1, `size_bucket` small/large —
   so the human can see whether any tag bucket is too small to plot.
4. Closing summary:
   ```
   wrote <out>  (<n> patches, <k> subset targets, <mb> MB sidecar)
   next: nanounet_train -d 999 --plans <plans> --val-manifest <out> ...
   ```

**Acceptance:**
- Re-running with the same seed produces byte-identical `.json` and `.npz`.
- Every entry's `case` is in `splits_final.json[0]["val"]`.
- Scenario counts match the requested mix exactly.
- Every cohort has ≥ `--floor` patches.
- `sum(cohort_weights.values()) == 1.0 ± 1e-6`.
- For every `subset_clicked` entry: `0 < len(instances_clicked) < len(instances_in_patch)`.

---

## 6. Items 5–6 — carry meta through, keep per-row Dice

### 6.1 `nanounet/train/patch_render.py` — extend `collate_patches`

Meta is per **item**; rows are per **variant**. Validation always has one variant per item, but
build the lists in the row loop anyway so the function stays correct if that changes.

Inside the existing `for pid, item in enumerate(batch)` loop, in the inner
`for v, ci in zip(...)` body, append `item[k]` to a per-key list for
`k in ("scenario", "cohort", "size_bucket", "has_subset")` when present in `batch[0]`.

After the existing `out = {...}`:

```python
    for k in ("scenario", "cohort", "size_bucket", "has_subset"):
        if k in batch[0]:
            out[k] = torch.tensor(meta[k], dtype=torch.long)
    if "target_subset" in batch[0]:
        out["target_subset"] = torch.stack([item["target_subset"] for item in batch])
```

Extend the docstring with one sentence explaining that `scenario`/`cohort`/`size_bucket` are
row-aligned integer codes from the fixed val manifest and absent during training.

### 6.2 `nanounet/model/dice_helpers.py` — per-row tp/fp/fn

`val_step_row` currently keeps only batch-pooled sums, so nothing can be re-pooled per bucket.
**Keep the existing `tp`/`fp`/`fn` keys exactly as they are** — the aggregate `val_dice` must stay
byte-identical — and *add* three per-row keys.

In `val_step_row`, after the existing `row = {...}`:

```python
    # Per-row fg tp/fp/fn kept alongside the batch-pooled sums so val_metrics can re-pool by
    # scenario/cohort. The pooled keys above are untouched: val_dice must stay byte-identical.
    row["tp_row"] = tp_fg.detach().cpu()
    row["fp_row"] = fp_fg.detach().cpu()
    row["fn_row"] = fn_fg.detach().cpu()
    row["pred_fg_row"] = (output_seg > 0).float().flatten(1).mean(1).detach().cpu()
```

Shapes: `tp_row`/`fp_row`/`fn_row` are `[B, Cfg]` (Cfg == 1 for this binary task);
`pred_fg_row` is `[B]`.

Add one helper:

```python
def pooled_dice_from_rows(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> float:
    """pooled_fg_dice for an arbitrary row subset: same formula, rows selected by a mask."""
    if tp.numel() == 0:
        return float("nan")
    a, b, c = tp.sum(0).numpy(), fp.sum(0).numpy(), fn.sum(0).numpy()
    dg = [2 * x / (2 * x + y + z) if (2 * x + y + z) > 0 else np.nan for x, y, z in zip(a, b, c)]
    return float(np.nanmean(dg))
```

---

## 7. Items 7–8 — metrics module and the Lightning module

### 7.1 `nanounet/train/val_metrics.py` — new, target ~170 LOC

```python
"""Validation metric logging: aggregate, per scenario, per source dataset, per tag.

Lives outside lightning_module.py, which is at the 200-LOC ceiling. Everything here is tensor
arithmetic over buffers already in host memory -- no forward passes, no device syncs beyond the
ones the caller already paid for.

Two reporting layers, deliberately asymmetric:
  * scenario  -- the full metric set, because each scenario tests a different behaviour;
  * cohort    -- val_dice and val_prompt_agreement only, because 17 datasets x 9 metrics is
                 unreadable and those two answer "how accurate" and "how stable".
Headline aggregates are re-weighted to the true cohort proportions, undoing the per-cohort patch
floor the manifest builder applies to keep small cohorts plottable."""
```

Public surface — exactly one function:

```python
def log_val_metrics(lm) -> None:
    """Called from NanoUNetLM.on_validation_epoch_end. Reads lm._val_buf, lm._val_buf_ablated,
    lm._agreement_buf, lm._meta_buf and calls lm.log(...)."""
```

Behaviour, in order:

**(a) Aggregate — unchanged names, unchanged formulas.** Move the existing body of
`on_validation_epoch_end` here verbatim: `val_dice`, `val_dice_macro`, `val_fp`, `val_n_a`,
`val_n_b`, `val_loss`, `val_dice_prompt_ablated`, `val_prompt_gap`, `val_dice_click_inside`,
`val_dice_click_outside`, `val_prompt_agreement`. Do not alter a single expression.

**(b) Early return** if `lm._meta_buf` is empty — that is a training run without a manifest, and
everything below is manifest-only. This keeps the non-manifest path bit-identical to today.

**(c) Per scenario** — for each of the 4, with `s` the scenario name:

| Key | Value | Logged for |
|---|---|---|
| `val/{s}/val_dice` | `pooled_dice_from_rows` over that scenario's rows | `all_clicked`, `subset_clicked` |
| `val/{s}/val_dice_macro` | mean per-row Dice over rows with GT foreground | `all_clicked`, `subset_clicked` |
| `val/{s}/val_pred_fg` | mean `pred_fg_row` | **all four** |
| `val/{s}/val_prompt_agreement` | mean of that scenario's agreement values | all four |
| `val/{s}/val_dice_prompt_ablated` | pooled Dice from the ablated buffer, same row mask | `all_clicked`, `subset_clicked` |
| `val/{s}/val_prompt_gap` | the two above, subtracted | `all_clicked`, `subset_clicked` |
| `val/{s}/n` | row count | all four |

`val/none_clicked/val_pred_fg` and `val/lesion_free_decoy/val_pred_fg` are the D10 metrics —
**lower is better, target ≈ 0.**

**(d) The subset diagnostic** — only over rows with `has_subset == 1`:

| Key | Meaning |
|---|---|
| `val/subset_clicked/val_dice_vs_clicked_subset` | prediction vs the clicked-subset target |
| `val/subset_clicked/val_dice_vs_all_lesions` | prediction vs the full seg |
| `val/subset_clicked/val_selectivity_margin` | the first minus the second |

**`val_selectivity_margin` is the headline number of this step.** Negative means the model scores
better against *all* lesions than against the ones it was pointed at — i.e. it ignores the click.
Both are computed with `pooled_dice_from_rows`; `vs_all_lesions` reuses the existing `tp_row`
family, and `vs_clicked_subset` needs a second `get_tp_fp_fn_tn` call against `target_subset`,
done in `validation_step` (see 7.2) so this module stays free of forward-pass concerns.

**(e) Per cohort** — for each cohort present in the manifest, two keys only:

```
val/cohort/{name}/val_dice
val/cohort/{name}/val_prompt_agreement
val/cohort/{name}/n
```

**(f) Tags:**

```
val/tag/click_inside/val_dice     val/tag/click_outside/val_dice
val/tag/small/val_dice            val/tag/large/val_dice
val/tag/small/n                   val/tag/large/n
```

**(g) Weighted headline (D7):**

```
val_dice_weighted               # cohort val_dice, weighted by header cohort_weights
val_prompt_agreement_weighted
```

Skip cohorts whose value is NaN and renormalise the weights over the rest. Guard against an
all-NaN epoch by logging `float("nan")`.

Every `lm.log(...)` uses `sync_dist=True`, matching the existing calls, so DDP ranks reduce.
Log `n` counts with `reduce_fx="sum"`, matching the existing `val_n_a` / `val_n_b` treatment.

**A bucket with zero rows logs `float("nan")`, never 0.0.** A fabricated zero looks like a
catastrophic regression on a W&B chart.

### 7.2 `nanounet/train/lightning_module.py` — must end up **under 199 LOC**

Three edits. The third one removes more lines than the first two add.

**Edit 1 — `__init__`, after `self._agreement_buf`:**

```python
        self._meta_buf: List[Dict[str, Any]] = []
```

**Edit 2 — `validation_step`.** After the existing `self._val_buf_ablated.append(...)`, add:

```python
        if "scenario" in batch:
            meta = {k: batch[k].cpu() for k in ("scenario", "cohort", "size_bucket", "has_subset")}
            if bool(meta["has_subset"].any()):
                ys = batch["target_subset"].to(self.device, non_blocking=True)
                meta.update(subset_row=subset_dice_row(out, ys, lm, ds))
            self._meta_buf.append(meta)
```

`subset_dice_row` is a new ~12-line helper in `dice_helpers.py` returning per-row `tp/fp/fn`
against `target_subset`, using the same `get_tp_fp_fn_tn` path as `val_step_row`. It runs **no
forward pass** — `out` is already computed. Cost is one extra one-hot scatter per val batch.

Also move the agreement append so the per-row values are stored (they already are — 
`prompt_pair_dice` returns a per-row tensor), and record the row count so `val_metrics` can align
agreement rows with meta rows. Alignment rule: **the agreement buffer and the meta buffer are
appended once per batch, in the same order, with the same row count** — assert this in
`val_metrics` with `assert sum(len(m["scenario"]) for m in buf) == agree.numel()`.

**Edit 3 — `on_validation_epoch_end` becomes:**

```python
    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "_epoch_t0") and not self.trainer.sanity_checking:
            self.log("epoch_wall_time_sec", float(time.perf_counter() - self._epoch_t0))
        if not self._val_buf:
            return
        log_val_metrics(self)
```

and `on_validation_epoch_start` also clears `self._meta_buf`.

Delete the now-unused imports (`agreement_mean`, `click_split_means`, `pooled_fg_dice`) from
`lightning_module.py` and import them in `val_metrics.py` instead. Net LOC change must be
**negative**; verify with `wc -l`.

---

## 8. Items 9–11 — wiring

### 8.1 `nanounet/cli/train_parser.py`

```python
    ap.add_argument("--val-manifest", default=None,
                    help="fixed validation manifest from nanounet_build_valset; "
                         "omit for the legacy per-epoch random val sampling")
```

In `validate_train_args`, fail at startup (E3) if the path is set and missing:

```python
    if args.val_manifest and not os.path.isfile(args.val_manifest):
        raise FileNotFoundError(
            f"--val-manifest {args.val_manifest} does not exist.\n"
            f"Fix: nanounet_build_valset -d {args.dataset_id} --plans {args.plans} "
            f"--config {args.config} --out {args.val_manifest}"
        )
```

Add a row to `train_config_rows` so the resolved value shows in the startup config table (U3).

### 8.2 `nanounet/train/data_module.py` — at most 8 new lines

Add `val_manifest: str | None = None` to `__init__` and store it. In `setup`, after `self.val_tf`
is built:

```python
        self.val_manifest = load_manifest(self.val_manifest_path) if self.val_manifest_path else None
```

Replace the body of `val_dataloader` with a two-branch `if` (R3 — an `if`, not a strategy object):

```python
    def val_dataloader(self) -> DataLoader:
        init_dataloader_ipc()
        if self.val_manifest is not None:
            return build_val_dataloader(
                self.val_manifest, self.case_folder, self.roi_cfg, self.val_tf, self.final_ps,
                self.longi, self.batch_size, self.dl_bucket, self.pin_memory,
                self.persistent_workers,
            )
        ... existing body unchanged ...
```

`fit.py` is **already over budget at 211 LOC** — pass `val_manifest=args.val_manifest` through the
existing `NanoDataModule(...)` call by adding it to the existing keyword list. That is one line, on
an existing call. Do not add anything else to that file.

### 8.3 `pyproject.toml`

```toml
nanounet_build_splits = "nanounet.cli.build_splits:main"
nanounet_build_valset = "nanounet.cli.build_valset:main"
```

Reinstall with `pip install -e .` afterwards, or the console scripts will not exist.

---

## 9. Items 12–13 — docs (D4: same change, not later)

`docs/steps/valset.md`, under 200 lines, following the D2 structure: 3-line summary, copy-paste
command block, argument tables for **both** new CLIs, inputs/outputs with paths and formats, the
manifest schema table from §4.1, the scenario table with what each one tests, and common errors
with fixes.

Copy-paste block, literal:

```bash
nanounet_build_splits -d 999 --plans nnUNetResEncUNetLPlans_h200_smallpv --val-frac 0.15 --force
```

```bash
nanounet_build_valset -d 999 --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --config configs/default.json \
  --out /nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/valset_1500.json
```

`docs/steps/train.md` gains a `--val-manifest` row in its argument table.

---

## 10. Verification — run all of it, report the numbers

### 10.1 Correctness

| # | Check | How |
|---|---|---|
| V1 | Split balance | Every cohort's val share is 0.15 ± 1 case; train/val disjoint; union == all ids |
| V2 | Split reproducibility | Same seed → byte-identical file |
| V3 | Manifest reproducibility | Same seed → byte-identical `.json` and `.npz` |
| V4 | Manifest cases are val-only | Every `case` ∈ `splits_final.json[0]["val"]` |
| V5 | **Determinism** | Two `Trainer.validate` runs, same manifest, same checkpoint → **identical** metrics. This is the acceptance criterion from the parent handoff. |
| V6 | Aggregate unchanged | Run once **without** `--val-manifest`: every legacy metric matches the pre-change code to float noise |
| V7 | Subset targets sane | For 5 random `subset_clicked` entries: subset mask is a strict non-empty subset of `seg_crop > 0` |
| V8 | Scenario purity | `none_clicked` entries have 0 clicks; `lesion_free_decoy` entries have `seg_crop.max() <= 0` and exactly 1 click |

For V5 you need a validate-only path. It does not exist — `fit.py` only ever calls `Trainer.fit`.
**Do not add one to `fit.py` (211 LOC).** Write a throwaway script under
`/tmp/claude-*/scratchpad/`, run it, report the numbers, delete it (R16).

### 10.2 GPU utilisation — hard gate, not optional

The parent handoff §2 makes this a blocker: **average GPU utilisation must stay above 95%.**
Measure before and after, same node, same `--dl-bucket`, same batch size.

| Signal | How |
|---|---|
| `epoch_wall_time_sec` | already logged, `lightning_module.py:160` |
| GPU utilisation | `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` for ≥3 epochs, report the mean |

Run 5 epochs on the old path, 5 on the manifest path, report a table of both signals.

**Expected: no change.** This step adds no per-patch CPU work — `cc3d`, the subset targets, both
prompt draws and the click-inside flags are all precomputed offline (D12/D13). The only new
runtime cost is one extra one-hot scatter per val batch for the subset Dice, on tensors already on
the GPU. If utilisation drops below 95%, **stop and report the measurement** rather than shipping.

### 10.3 Report to the human

1. The split table (cohort × train/val).
2. The manifest composition tables (cohort × scenario, and the tag counts).
3. V5/V6 results.
4. The before/after GPU table.
5. A first read of the new metrics on the baseline checkpoint
   `/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200/checkpoints/best-epoch=570-val_dice=0.8030.ckpt`
   — especially `val/subset_clicked/val_selectivity_margin` and
   `val/none_clicked/val_pred_fg`.

Expect those last two to look **bad**. The model has never been trained to segment selectively or
to stay quiet; the parent handoff §Step 6 says so explicitly. Bad baselines here are the evidence
that motivates Step 6, not a defect in this work.

---

## 11. Do not do

- Do not run `cc3d`, EDT, or connected components anywhere in `nanounet/data/valset.py` or the
  validation loop. Offline only.
- Do not change the formula of `pooled_fg_dice` or the aggregate `val_dice`.
- Do not grow `fit.py`, `patch_iterable.py`, or push `lightning_module.py` back over 199 LOC.
- Do not draw the decoy per prompt draw. One decoy per patch, shared by both — undoing this
  silently reverts the Step 0 fix.
- Do not add an `EarlyStopping`, an LR change, or any training-side change. This step is
  measurement only.
- Do not create a permanent `tests/` folder (R16). Validate, report, delete.
- Do not invent a fallback when the manifest, its sidecar, or a centroid file is missing. Raise
  with the command that fixes it (R12).
- If a file or field this plan describes does not exist or is named differently, **stop and report
  what you found**. Do not guess a substitute.
