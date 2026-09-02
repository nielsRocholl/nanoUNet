# segtrack propagated-coordinate contract

Status: implementation-ready after
[instance labels and EMA](segtrack_instances_ema_plan.md).

## 1. Frames

| Boundary | Order |
|----------|-------|
| SimpleITK array / `*_zyx` instance mask | `(z,y,x)` |
| nibabel volume / matcher centroid / metadata `cog_*` | `(x,y,z)` |
| propagated JSON `point` | `(x,y,z)` |
| ITK continuous index and physical point | `(x,y,z)` |
| documented slim CSV columns | external `(z,y,x)` |

`segtrack.py` correctly converts instance arrays to matcher arrays with
`transpose(2,1,0)`. `tracking.data.appearance.centroids` then returns `(x,y,z)`.
Do not transpose metadata or propagated JSON values. Do reorder slim CSV values at load.

## 2. Rename the misleading parser

In `/lesion-tracking/tracking/data/meta.py`, rename `parse_zyx` to `parse_xyz` and change
its error to:

```python
def parse_xyz(s: object) -> tuple[float, float, float] | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).strip()
    if not t:
        return None
    parts = t.split()
    if len(parts) != 3:
        raise ValueError(f"expected x y z triple, got {s!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])
```

Update the four `parse_meta_csv` calls and the import/use in `propagate.py`. There are no
other repository call sites.

## 3. Keep the documented `cog_fu` fallback — do not remove it

`/lesion-tracking/tracking/data/propagate.py::_from_meta` falls back from
`cog_propagated` to `cog_fu` when propagation is empty (23 real BL-mask IDs on the
held-out set). The FU-click JSON has the same character: of 683+90 checked points, 90
equal `cog_fu` instead of a registration warp.

This is **not leakage to fix**. Confirmed with the dataset owner: the dataset team
registered most lesions but ran out of time for some, and recorded the true FU annotation
as the substitute for both the meta-CSV `cog_propagated` column and the FU-click JSON.
That is a permanent, documented characteristic of this dataset, not a bug this pipeline
introduces. Dropping those lesions would reduce prediction coverage for no benefit — there
is no better estimate available for them.

**Leave `_from_meta` and the FU-click JSON path unchanged on this point.** No code change
in this section. `nanoUNet`'s existing
`prop = case.meta_csv if case.meta_csv is not None else case.fu_clicks` also stays as-is —
do not restrict it to `case.meta_csv` only.

The fallback stays silent: no per-lesion `imputed` tagging in output for this pass.
`tracking/data/provenance.py` already computes an `imputed` flag (registration guess vs.
real annotation) if stratified reporting is wanted later; it is not wired into the segtrack
CLI or scorer here.

Before live registration exists, a geo checkpoint with **neither** `case.meta_csv` nor
`case.fu_clicks` present (`prop is None`) must fail with an actionable error. A `drop_dp`
checkpoint must continue to pass `None` and work.

## 4. Fix slim CSV order

`_from_slim` documents columns `lesion_id,z,y,x` but returns them unchanged to an internal
`(x,y,z)` consumer. Replace:

```python
np.asarray([float(r["z"]), float(r["y"]), float(r["x"])], dtype=np.float64)
```

with:

```python
np.asarray([float(r["x"]), float(r["y"]), float(r["z"])], dtype=np.float64)
```

Do not change the external schema in this patch.

## 5. Preserve graph-partition exclusions

A single patient's meta CSV can cover more than one BL/FU scan pair (different
`img_id_bl`/`img_id_fu`, e.g. different anatomical regions or follow-up timepoints). One
`nanounet_segtrack` run works on exactly one such pair. Without filtering lookups by
`img_id`, "no `cog_propagated` found for this BL id" conflates two different situations:

1. the lesion ID belongs only to a *different* scan pair for this patient; it was never
   part of this run's volumes, and there is nothing to propagate — exclude it, do not fill;
2. the lesion ID *is* in this run's scan pair, but has no coordinate; this is the same
   documented registration gap as section 3, and after the `cog_fu` fallback in section 3
   it should already have a coordinate in nearly all cases.

Across 68 held-out BL masks the split is 39 (category 1) versus six (category 2). Any
future live registration must only ever target category 2 IDs, never category 1.

Add to `propagate.py`:

```python
def outside_region_ids(path: Path, bl_ids: list[int], img_id: int) -> set[int]:
    df = pd.read_csv(path)
    if not {"lesion_id", "img_id_fu"} <= set(df.columns):
        return set()
    want = set(map(int, bl_ids))
    any_region = set(map(int, df["lesion_id"])) & want
    here = set(map(int, df.loc[df["img_id_fu"].astype(int) == img_id, "lesion_id"])) & want
    return any_region - here
```

Call this only for a metadata CSV with a known `img_id`. A slim CSV and explicit JSON have
no region-partition semantics.

The post-gate merge is:

```python
prop, _ = load_propagated(meta_csv, all_bl_ids, img_id=region)
outside = outside_region_ids(meta_csv, all_bl_ids, region)
missing = sorted(set(all_bl_ids) - set(prop) - outside)
```

Then register `missing`; never register `outside`. A case with no metadata has
`outside = set()` and all BL IDs are registration candidates.

## 6. Verification

Use temporary assertions against real data:

```python
meta = Path("/nnunet_data/Longitudinal-CT/meta/307fd7f231.csv")
prop, _ = load_propagated(meta, [1, 2], img_id=0)
assert np.allclose(prop[1], [207.95331503893644, 286.0173397756782, 119.03847670485004])
assert np.allclose(prop[2], [332.98329748071467, 330.1652544079212, 108.33511925423443])
```

Expected behavior:

- metadata rows with empty `cog_propagated` fall back to `cog_fu` (documented dataset
  limitation, kept intentionally — see section 3);
- an explicit JSON preserves `[x,y,z]`;
- slim row `z=1,y=2,x=3` returns `[3,2,1]`;
- malformed triples say `expected x y z triple`;
- the 68-case classification totals 39 `other_region_only` and six
  `same_region_no_coord`;
- `drop_dp=true` succeeds with `propagated=None`;
- a geo checkpoint with neither `case.meta_csv` nor `case.fu_clicks` present fails before
  matcher inference with an actionable error.

Delete temporary scripts after validation.
