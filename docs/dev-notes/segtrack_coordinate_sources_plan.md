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

## 3. Remove ground-truth leakage

`/lesion-tracking/tracking/data/propagate.py::_from_meta` currently falls back from
`cog_propagated` to `cog_fu`. For the held-out BL masks this leaks 23 real lesion
locations into matcher geometry. Replace:

```python
c = parse_xyz(r["cog_propagated"]) or parse_xyz(r.get("cog_fu"))
```

with:

```python
c = parse_xyz(r["cog_propagated"])
```

Update the module docstring to `Meta CSV: optional img_id_fu filter; cog_propagated only.
Missing BL ids omitted.`

Keep JSON support because an explicitly supplied JSON may contain registration-warped BL
points. Its contract is not a generic FU-click file. In `nanoUNet` specifically, stop using
`case.fu_clicks` as a propagated source:

```python
prop = case.meta_csv
```

Before live registration exists, a geo checkpoint with `prop is None` must fail with an
actionable error. A `drop_dp` checkpoint must continue to pass `None` and work.

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

Missing `cog_propagated` has two meanings:

1. the lesion is assigned only to another `img_id_fu`; exclude it from this graph;
2. a same-region row exists without a coordinate; this is a registration gap.

Across 68 held-out BL masks the split is 39 versus six IDs. Live registration must fill
only category 2.

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

- metadata rows with empty `cog_propagated` are omitted even when `cog_fu` exists;
- an explicit JSON preserves `[x,y,z]`;
- slim row `z=1,y=2,x=3` returns `[3,2,1]`;
- malformed triples say `expected x y z triple`;
- the 68-case classification totals 39 `other_region_only` and six
  `same_region_no_coord`;
- `drop_dp=true` succeeds with `propagated=None`;
- a geo checkpoint with no metadata fails before matcher inference and does not inspect
  `fu_clicks`.

Delete temporary scripts after validation.
