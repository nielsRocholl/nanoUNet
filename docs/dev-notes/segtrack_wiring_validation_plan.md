# segtrack post-gate wiring and validation

Status: implement only after
[live-registration gate](segtrack_registration_gate_plan.md) passes.

## 1. Matcher API: in-memory propagated points

In `/lesion-tracking/tracking/data/masks.py::build_mask_graph`, add:

```python
extra_propagated: dict[int, np.ndarray] | None = None,
```

Replace the non-`drop_dp` source block with:

```python
if propagated_csv is None and extra_propagated is None:
    raise FileNotFoundError(
        "No propagated coordinates for a geo matcher.\n"
        "Expected a propagated file or live registration output.\n"
        "Fix: pass --propagated, enable live registration, or use a drop_dp checkpoint"
    )
prop, types = load_propagated(propagated_csv, all_bl, img_id=img_id) \
    if propagated_csv is not None else ({}, {})
if extra_propagated is not None:
    assert not set(prop) & set(extra_propagated)
    prop.update(extra_propagated)
bl_ids = [i for i in all_bl if i in prop]
```

An empty `{}` is meaningful: coordinate input was intentionally empty after
`--drop-uncovered`. Do not test it by truthiness.

In `/lesion-tracking/tracking/infer.py::track`, add the same keyword argument. Validate
geo inputs once, before the `volumes` branch:

```python
if not gcfg.drop_dp:
    if propagated is None and extra_propagated is None:
        raise FileNotFoundError("No propagated coordinates for a geo matcher ...")
    if propagated is not None and not Path(propagated).is_file():
        raise FileNotFoundError(f"No propagated at {propagated} ...")
```

For `drop_dp`, reject either source:

```python
if propagated is not None or extra_propagated is not None:
    raise SystemExit("drop_dp checkpoint does not use propagated coordinates ...")
```

File checks inside both `volumes` branches must not call `Path(None)`. Pass:

```python
Path(propagated) if propagated is not None else None
```

to `build_mask_graph`, and forward `extra_propagated`. Keep `_empty_result`; an empty
geo graph is valid after explicit dropping. Keep `tracking/infer.py` below 200 lines by
replacing its duplicate propagated guards rather than appending another guard.

## 2. `run_case` source merge

After Frame A masks are transposed to `mk_bl`/`mk_fu`, compute `all_bl_ids` as specified
in the instance plan. Read metadata points (including the kept `cog_fu` fallback, see
[the coordinate plan](segtrack_coordinate_sources_plan.md) section 3) only when
`case.meta_csv` exists; this draft does not change the existing
`case.meta_csv or case.fu_clicks` behavior for the no-metadata case, so `missing` below is
expected to be empty or near-empty for metadata cases in practice.

```python
prop_path = case.meta_csv
prop, outside = {}, set()
if not drop_dp and prop_path is not None:
    prop, _ = load_propagated(prop_path, all_bl_ids, img_id=region)
    outside = outside_region_ids(prop_path, all_bl_ids, region)
missing = sorted(set(all_bl_ids) - set(prop) - outside) if not drop_dp else []
extra = None
if missing and drop_uncovered:
    cprint(f"[dim]drop {case.stem}  BL ids {missing} (no propagated coverage)[/dim]")
elif missing:
    from tracking.data.appearance import centroids
    extra = propagate_points(
        case.bl_img, case.fu_img, centroids(mk_bl, missing),
        io_iterations=io_iterations,
    )
elif not drop_dp and prop_path is None:
    extra = {}
```

The last branch is reached only when there are no BL IDs; normal `run_case` returns earlier
for an empty BL mask. Keep it so source semantics remain explicit.

For no metadata plus `--drop-uncovered`, `missing` is all BL IDs but `extra` must be `{}`,
not `None`, before calling `track`. Set that in the drop branch:

```python
if prop_path is None:
    extra = {}
```

Pass `prop_path`, `extra_propagated=extra`, and `img_id=region` to `track`.
Use `centroids(mk_bl, missing)`, never `centroids(bl_zyx, missing)`.

Any registration exception fails the case. No automatic `drop_dp`, FU-click, identity,
or native-BL-centroid fallback.

## 3. Case collection and CLI

In `/nanoUNet/nanounet/cli/segtrack_cases.py::_folder`, missing per-patient metadata:

- `--drop-uncovered`: preserve current skip and report `no meta csv`;
- otherwise: keep the case with `meta_csv=None` for live registration.

Do not make `meta_dir` itself mandatory. Single-case no-metadata follows the same policy.

Add to `nanounet/cli/segtrack.py`:

```python
ap.add_argument(
    "--drop-uncovered", action="store_true",
    help="drop BL ids without propagated coordinates instead of live registration",
)
ap.add_argument(
    "--io-iterations", type=int, default=FROZEN_IO_ITERATIONS,
    help="uniGradICON instance-optimization steps; 0 disables",
)
```

`FROZEN_IO_ITERATIONS` is the exact winner from the registration gate; do not choose a
value while wiring. Reject negative values immediately. Add config-table rows for
registration policy and IO iterations. Update `docs/steps/track.md` in the same change.

## 4. Registration model lifecycle

The gate must specify the measured GPU-safe lifecycle. Apply it verbatim:

- whether segmentation/matcher remain on GPU during registration;
- whether uniGradICON remains cached on GPU between cases;
- where `torch.cuda.reset_peak_memory_stats()` is called for measurement only.

Do not add `empty_cache()` as an unmeasured ritual. Do not preload weights for
`drop_dp` or `--drop-uncovered` runs.

## 5. Correct scorer

Add `/nanoUNet/scripts/score_segtrack_fu.py` as a reproducible evaluation tool, not a
temporary test. It must emit these distinct fields:

- `n_cases`, `n_gt_lesions`, `n_pred_lesions`;
- `n_id_intersections`: GT IDs present anywhere in prediction;
- `n_positive_overlaps`: GT IDs whose same-ID Dice is >0;
- pooled `mean_all`, `median_all`, and `mean_positive`;
- per-case binary foreground Dice;
- cases with empty GT, empty prediction, or both.

Compute aggregates from one pooled Dice list:

```python
all_dice = [d for case in cases for d in case["lesion_dice"]]
positive = [d for d in all_dice if d > 0]
summary = {
    "mean_all": float(np.mean(all_dice)) if all_dice else None,
    "median_all": float(np.median(all_dice)) if all_dice else None,
    "mean_positive": float(np.mean(positive)) if positive else None,
}
```

Never call `mean_positive` `mean_detected`; ID presence and spatial overlap are different.
For binary Dice, define both-empty as 1.0 and exactly-one-empty as 0.0. Preserve per-case
records so every aggregate is auditable.

## 6. End-to-end matrix

Run in separate output roots:

1. complete metadata, geo checkpoint, no live call;
2. partial metadata, live only for same-region missing IDs;
3. no metadata, live for every BL ID;
4. no metadata + `--drop-uncovered`, folder case skipped;
5. partial metadata + `--drop-uncovered`, uncovered IDs absent from graph;
6. no metadata, explicit `drop_dp` checkpoint, no registration;
7. empty BL and empty FU cases;
8. BL ID larger than every provisional FU ID, verifying no output collision.

For every completed case assert:

- `fu.mha > 0` equals `pred_fu.mha > 0`;
- every shared nonzero BL/FU mask ID appears as `track_id` in `matches.csv`;
- live coordinates are finite Frame-B `(x,y,z)` points inside expected bounds;
- other-region-only BL IDs never enter the graph;
- live-registered coordinates (case 3/5 above) never equal the stored `cog_fu` value for
  that lesion — that would mean the research alignment peeked at ground truth, not that a
  metadata `cog_fu` fallback occurred (metadata `cog_fu` fallback is expected and fine,
  per [the coordinate plan](segtrack_coordinate_sources_plan.md) section 3).

Finally score the full fixed cohort before/after. Report segmentation retention, ID-linked
Dice, pair precision/recall/F1, complete-match score, runtime, and peak GPU memory with
denominators. A gain on one case is not a release gate.
