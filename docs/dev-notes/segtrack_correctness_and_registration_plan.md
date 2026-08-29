# segtrack correctness + live registration plan

Status: **plan only. Nothing below is implemented.** Second-audited on 2026-08-29
against `/nanoUNet`, `/lesion-tracking`, installed ITK/ICON source, real checkpoints,
and held-out volumes. Do not implement the deleted 909-line draft from commit `58c78e7`.

## Verdict

Instance retention, tracking-ID reservation, EMA default, and coordinate-source cleanup
are ready to implement. Automatic live registration is **blocked**: transform direction is
correct, but current image alignment is not robust enough to become the default.

Detailed implementation documents:

1. [Instance labels, tracking IDs, EMA](segtrack_instances_ema_plan.md)
2. [Propagated-coordinate contract](segtrack_coordinate_sources_plan.md)
3. [Live-registration research gate](segtrack_registration_gate_plan.md)
4. [Post-gate wiring and verification](segtrack_wiring_validation_plan.md)

## Repositories

- `/nanoUNet`: segmentation CLI, orchestration, uniGradICON wrapper, user docs.
- `/lesion-tracking`: instance labels, coordinate loading, graph construction, matcher.

Both repositories are required. `nanoUNet` already depends on `unigradicon>=1.0.4`;
do not add uniGradICON to `lesion-tracking`.

## Verified evidence

Legacy score artifact:
`/nnunet_data/NanoUNet_results/segtrack/followup/dice_vs_targetsTrFU.json`.
Active `$NANOUNET_RESULTS` is `/nanounet_data/NanoUNet_results`; these are different
directories. Never describe the legacy artifact as living under the active environment path.

| Measurement | Verified value |
|-------------|----------------|
| paired scored cases | 63 |
| GT lesions | 549 |
| final predicted IDs | 284 |
| GT/pred ID intersections | 266 |
| GT lesions with positive ID-matched Dice | 254 |
| median Dice over all GT lesions | 0.0 |
| mean Dice over positive overlaps | 0.730646 |
| EMA FU prediction CCs | 687 |
| CCs retained by exact-click labeling | 311 |
| CCs deleted before matching | 376 |
| GT lesions with any binary-prediction overlap | 478 / 549 |
| GT lesions whose best CC was deleted | 163 |
| deleted best CC with Dice >= 0.5 | 135 |
| click collisions on one predicted CC | 38 |

Conclusion: exact-click instance labeling is a major proven failure mode, but not the sole
cause of zero Dice. Seventy-one GT lesions have no binary overlap, and connected predictions
hit by multiple clicks still cannot represent multiple instances.

FU JSON is not uniformly a registration product. Across the 63 paired cases, 683 points
equal `cog_propagated` and 90 equal `cog_fu`. `load_propagated::_from_meta` also uses
`cog_fu` when propagation is missing; 23 real BL-mask IDs receive this ground-truth
location. Remove that leakage.

Across 68 held-out BL masks, 45 IDs are absent after current region-filtered loading:
39 belong only to another `img_id_fu`; six have a same-region row but no coordinate.
The 39 are intentional graph-partition exclusions, not registration failures.

## Coordinate contract

Three representations exist:

- Frame A: SimpleITK/numpy instance arrays `(z,y,x)`.
- Frame B: matcher/nibabel/ITK continuous indices `(x,y,z)`.
- Physical ITK points: millimetres `(x,y,z)`.

`load_clicks` converts JSON `[x,y,z]` to Frame A. `segtrack.py` transposes Frame A masks
with `(2,1,0)` before matching. `centroids(mk_bl, ...)`, metadata `cog_*`, propagated JSON,
and live-registration output are Frame B.

The documented slim CSV is external `(z,y,x)`. `_from_slim` must reorder it to internal
Frame B `[x,y,z]`; current code does not.

For `register_pair(model, BL, FU)`, `phi_AB` maps FU physical points to BL and `phi_BA`
maps BL physical points to FU. This was proved from installed ICON source and a synthetic
ITK transform. Metadata-only `SetOrigin` pre-alignment cancels when the result is converted
back to an index; it cannot improve registration.

## Required order

1. Implement instance retention and reserve **all** BL IDs during FU painting.
2. Make segmentation EMA default without removing the existing `--ema` spelling.
3. Enforce strict propagated-coordinate sources and preserve other-region exclusions.
4. Run registration research gate. Do not wire live fallback before it passes.
5. Wire `extra_propagated`, case collection, and CLI only after the gate.
6. Re-run the exact scorer and compare both segmentation and tracking metrics.

Each numbered document is a gate. A failed gate stops later work.

## Non-negotiable invariants

- Every predicted 18-connected foreground component reaches the matcher.
- Every integer shared by `bl.mha` and `fu.mha` is backed by a match row.
- Metadata coordinates used as BL-in-FU positions come only from `cog_propagated`.
- IDs assigned exclusively to another FU region remain excluded.
- No bare `(z,y,x)`/`(x,y,z)` triple crosses a module boundary undocumented.
- No origin-only alignment is introduced.
- Live registration does not become default from a one-point smoke test.
- New flags update `docs/steps/track.md` in the same change.
- Every touched source or documentation file remains below 200 lines.

## Safe behavior before registration passes

Removing the FU-click and `cog_fu` fallbacks may make a geo matcher reject cases without
propagated coordinates. That explicit failure is safer than silently using a false frame.
Users needing CSV-free tracking can explicitly pass the existing
`/nnunet_data/lesion_tracking/runs/v7_nodp_complete/last.ckpt`
(`drop_dp=true`; measured 57-graph match score 0.8929 versus 0.9558 for geo complete).
Do not switch checkpoints automatically.

## Out of scope

- Changing `nanounet_predict` EMA defaults.
- Changing the standalone `lesion_track` registration behavior before segtrack is proven.
- Matcher retraining, MRI registration, or biological split/merge decoding policy.
- Renumbering canonical BL IDs.
- Treating the held-out test set as a registration-method development set.

## Persistence

Workspace disk may vanish. Copies of the next-agent prompt and canvases:

- `/nnunet_data/segtrack_audit/`
- `/home/nielsrocholl/segtrack_audit/`
