# segtrack live-registration research gate

Status: **blocked research phase**. Do not wire registration into `nanounet_segtrack` yet.

## 1. Why this is blocked

Existing uniGradICON derivatives prove the backend works often, but not reliably enough for
an automatic default. Recomputed from the source records used by
`registration_error_table.json`:

| Backend | lesions | patients | raw median | raw p90 | lesion outliers | failed patients |
|---------|---------|----------|------------|---------|-----------------|-----------------|
| original | 2,571 | 282 | 4.70 mm | 25.17 mm | 20 | 26 |
| uniGradICON | 2,563 | 283 | 7.13 mm | 53.10 mm | 119 | 45 |

The stored `offsets_zyx` exclude outliers and failed patients; computing a 5.63 mm median
from that filtered array does not measure deployment reliability. Conversely, 92.7 mm is
not the pooled median of this artifact and must not appear in reports.

The uniGradICON exclusion rules were >100 resampled voxels per lesion and >20-voxel
patient median after lesion filtering. Forty-five failed patients are 15.9% of measured
patients. This blocks default live use.

## 2. Reuse the existing backend

Use `/nanoUNet/nanounet/register/unigradicon.py`. `get_model()` already imports the package,
downloads/caches weights, and caches the model. `nanoUNet/pyproject.toml` already declares
`unigradicon>=1.0.4`. Do not create a registration module or dependency in
`lesion-tracking`.

Append an experimental point API:

```python
def propagate_points(
    bl_path: Path,
    fu_path: Path,
    points_xyz: dict[int, np.ndarray],
    *,
    io_iterations: int = 0,
) -> dict[int, np.ndarray]:
    import icon_registration.itk_wrapper as itk_wrapper
    import itk
    import numpy as np
    from unigradicon import preprocess

    bl = itk.imread(str(bl_path), itk.F)
    fu = itk.imread(str(fu_path), itk.F)
    steps = io_iterations if io_iterations > 0 else None
    _, phi_ba = itk_wrapper.register_pair(
        get_model(), preprocess(bl, "ct"), preprocess(fu, "ct"),
        finetune_steps=steps,
    )
    out: dict[int, np.ndarray] = {}
    for lid, xyz in points_xyz.items():
        idx = itk.ContinuousIndex[itk.D, 3]()
        for axis in range(3):
            idx[axis] = float(xyz[axis])
        p_bl = bl.TransformContinuousIndexToPhysicalPoint(idx)
        p_fu = phi_ba.TransformPoint(p_bl)
        out[lid] = np.asarray(fu.TransformPhysicalPointToContinuousIndex(p_fu))
    return out
```

Add `from pathlib import Path` and `import numpy as np` at module scope instead of the two
local imports shown if that keeps the file clearer.

Why `phi_ba`: installed `itk_wrapper.create_itk_transform` states
`warp(image_A, phi_AB) is close to image_B`; ITK resampling is a pull map, so `phi_AB`
maps B coordinates to A. With `register_pair(model, BL, FU)`, `phi_BA` maps a BL point
into FU.

ITK 5.4.7 rejects a Python list in
`TransformContinuousIndexToPhysicalPoint`. The explicit
`itk.ContinuousIndex[itk.D,3]` object is mandatory.

Reject `io_iterations < 0` at the CLI boundary. Zero means `finetune_steps=None`; passing
zero directly makes `icon_registration` raise.

## 3. Unit gates before running the network

Write a temporary script that:

1. builds BL and FU ITK images with non-zero origins, anisotropic spacing, and non-identity
   direction;
2. uses an exact known ITK transform;
3. maps several subvoxel BL indices through index -> physical -> transform -> FU index;
4. checks each result against an independently calculated physical-space answer;
5. proves swapping `phi_AB` and `phi_BA` fails the assertion.

Also monkeypatch `itk_wrapper.register_pair` so `propagate_points` itself is exercised
without weights. Include a non-integer point; integer-only tests miss the API bug.

## 4. Reproduce the existing derivative

Before changing alignment, compare direct point propagation with successful
`derivatives/unigrad-icon-registration/*/lesions/*.json` records whose
`fill_source.bl == "warped_bl"`.

For each point:

- direct `phi_BA` result should be close to the centroid of the warped stamped ID;
- report differences, not only registration error against `cog_fu`;
- investigate large differences as transform-direction, metadata, or centroid-resampling
  bugs before quality tuning.

Run `io_iterations=0` and the derivative-generation setting separately. Never infer the
setting from the current CLI default.

## 5. Research alignment without target leakage

`warp_case` uses BL/FU lesion correspondence for disjoint physical frames. That is valid
for dataset construction but unavailable in deployment. Live segtrack must not use
`fu_clicks`, `cog_fu`, shared lesion IDs, or a target mask for alignment or model choice.

Evaluate on a fixed development split:

1. raw physical metadata;
2. body-mask geometric-center translation applied as a real resampling transform;
3. image-only rigid body registration followed by uniGradICON;
4. body-masked uniGradICON if supported by the installed model.

An origin edit alone is invalid: converting the transformed physical point back through
the edited FU origin cancels the shift. Any pre-transform must resample BL or be explicitly
composed with the learned point transform.

Freeze the method and thresholds on development data. Use held-out data once for the final
estimate.

## 6. Required measurements

For every candidate, persist:

- point error median, p75, p90, p95 in millimetres;
- fraction >25, >50, and >100 mm;
- points outside FU continuous-index bounds;
- case failure count and reason;
- runtime median/p95;
- CUDA peak allocated and reserved memory;
- downstream matcher pair precision, recall, F1, and complete-match score.

Stratify by overlapping/disjoint physical frame, lesion diameter, time interval, and body
region. Report denominators. Do not remove failures before headline metrics.

## 7. Pass criteria

Live wiring may start only when one frozen, leakage-free candidate:

- has no coordinate-direction or ITK API failures;
- has fewer lesion outliers and failed patients than the existing uniGradICON derivative;
- is no worse than the original registration backend on raw median and p90 by more than
  10%;
- is no worse than the deployed geo matcher with metadata on downstream complete-match
  score by more than one percentage point;
- beats the explicit `drop_dp` checkpoint on the same no-metadata cases;
- fits alongside loaded segmentation and matcher models without OOM.

If no candidate passes, keep live registration unavailable. Use strict metadata or an
explicit `drop_dp` checkpoint; do not silently degrade.
