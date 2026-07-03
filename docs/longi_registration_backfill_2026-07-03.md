# BL→FU registration backfill — 2026-07-03

Goal: recover from a CIFS write bug that left the `register_longi` batch job (`unprocessed-universal-lesion-segmentation-registered`, 316 BL/FU pairs) mostly incomplete, and understand what's left. This is a self-contained record; see [[longi_registration_landmark_align_plan]] and [[longi_registration_refine_plan]] for the registration algorithm itself.

## TL;DR

- **272/316 pairs now fully registered** (up from 14 at the start of this session), with zero re-registration needed for 256 of those — a plumbing bug, not a registration bug.
- **8/316** are legitimately unregisterable: both BL and FU lesion clicks vanish by follow-up (no `cog_fu` in the meta CSV), so there's no landmark to warp onto. Not a bug — exclude from the dataset.
- **36/316** are genuine elastix registration failures (`"Too many samples map outside moving image buffer"`). Investigated but not fixed — see [below](#the-36-elastix-failures). These need dedicated registration R&D, not a quick parameter tweak.

## What we register, and why

Longitudinal lesion segmentation needs the baseline (BL) scan warped into the follow-up (FU) frame so a two-timepoint model can see both on a common grid, with BL lesion clicks propagated to FU coordinates via the meta CSV's `cog_bl`/`cog_fu` correspondences. `nanounet/cli/register_longi.py` (itk-elastix rigid→affine→bspline, MI metric, optional body-mask + per-lesion click refinement) does this per BL/FU pair and writes a 7-file mini-dataset per case to `/nnunet_data/unprocessed-universal-lesion-segmentation-registered/`:

```
inputsTrBL/{pid}_{idx}.nii.gz   warped BL image
inputsTrBL/{pid}_{idx}.json     BL clicks (in FU frame)
targetsTrBL/{pid}_{idx}.nii.gz  warped BL lesion mask
inputsTrFU/{pid}_{idx}.nii.gz   FU image (copied)
inputsTrFU/{pid}_{idx}.json     FU clicks (from meta CSV cog_fu, or copied)
targetsTrFU/{pid}_{idx}.nii.gz  FU lesion mask (copied)
meta/{pid}.csv                  lesion correspondence table (copied)
```

Downstream this feeds `nanounet_preprocess` → the d013 two-stream longitudinal finetune ([[longitudinal_dwb_design]]).

## What went wrong

The original batch (run interactively, never committed as a script — only in `~/.bash_history`) used `NANOUNET_REG_THREADS=1`, `J=6` concurrent cases, resuming on `inputsTrBL/*.nii.gz` existing. Two bugs compounded:

1. **`shutil.copy2` on CIFS.** `nanounet/register/output.py`'s `_copy_if_exists()` used `shutil.copy2`, whose `copystat`/`utime` call raises `PermissionError` on the CIFS mount (`//blissey.umcn.nl/oncology` → `/nnunet_data`). The crash happened on the *first* FU copy call — after the warped BL (the expensive elastix output) had already been written, but before any FU/meta file was copied. Fix: `shutil.copyfile` (data only, no metadata), matching the existing pattern in `nnunetv2/utilities/shutil_sol.py`.
2. **Resume logic only checked `inputsTrBL/*.nii.gz`.** Since that file was always written before the crash, every partial case looked "done" to the resume check and was never retried.

Result: only 14/316 cases were fully complete; 257 had a warped BL but nothing else; 1 case (`f2fc990265_00`) never ran at all (empty log).

## The fix

- [`nanounet/register/output.py`](../nanounet/register/output.py) — `copy2` → `copyfile`.
- [`scripts/slurm_nanounet_register_longi.sh`](../scripts/slurm_nanounet_register_longi.sh) — the driver loop, now actually committed (previously only in bash history), with a resume check requiring **all 7 output files**, not just the warped BL.
- [`nanounet/cli/repair_longi_fu.py`](../nanounet/cli/repair_longi_fu.py) — a fast repair pass for the 257 already-partial cases. Key insight: `WarpResult.fu_out` (the FU click coordinates) is derived purely from the meta CSV ([`warp_case.py:84`](../nanounet/register/warp_case.py:84)), *before* the expensive elastix step — it doesn't depend on the registration result at all. So for cases where the warped BL already existed (registration had already succeeded), the missing FU json / FU mask / meta CSV could be re-derived/copied directly, without re-running elastix. Ran once: **256 cases repaired in seconds** (vs. an estimated ~10 hours to re-register them all).
- 1 never-ran case (`f2fc990265_00`) needed a real registration attempt (no shortcut available) — ran to completion normally (~65 min, single-threaded; slower than the typical 15–40 min).

Net: **271 → 272 additional complete cases**, using seconds of compute instead of hours.

## The 8 legitimate CSV-skip cases

Confirmed via [`nanounet/register/landmarks.py`](../nanounet/register/landmarks.py) (`read_pairs`, `correspondence`): a case is skipped with `"no CSV lesion pairs with both cog_bl and cog_fu"` only when *every* lesion's `cog_fu` field is empty. Checked one case's raw CSV directly — `cog_fu` is genuinely blank for lesions marked `DISAPPEARING` with `volume_fu=0.0`. These lesions vanished by follow-up, so there is no FU landmark to register against. This is correct behavior, not a parsing bug. Exclude these 8 from the registered dataset:

`06eb133bbf_00`, `07e1cd7dca_00`, `0aa1883c64_00`, `5878a7ab84_00`, `b73ce398c3_00`, `cfa0860e83_00`, `d947bf06a8_00`, `fc22130974_00`

## The 36 elastix failures

All fail with the same error, and fail **fast** (5–120s, not 15–40min — this made iteration cheap):

```
ITK ERROR: AdvancedMattesMutualInformationMetric: Too many samples map outside moving image buffer
```

This means the *initial* transform (before any optimization) already puts most sampled points from the fixed (FU) image outside the moving (BL) image's physical bounds — i.e. the coarse pre-alignment is bad, not the fine registration.

**Diagnostic pass** (image-header-only, no elastix — see breakdown logic in `nanounet/register/elastix.py:frame_z_overlap_mm` and the ad hoc bounding-box overlap check used for this investigation):

- 20/36 have `frame_z_overlap_mm <= 0` (BL and FU physical z-ranges don't overlap at all in the raw DICOM frames) — these already go through the landmark-based rigid pre-alignment in `warp_case.py`, and still fail.
- 16/36 have positive z-overlap (bounding-box overlap fraction 0.00–1.00) and go through elastix's `GeometricalCenter` initialization (align image bounding-box centers) — a much cruder init that ignores where the patient's anatomy actually sits within the FOV.
- Raw physical distance between paired `cog_bl`/`cog_fu` landmarks (before any alignment) ranges 14–200mm across these cases — large but not obviously separable into "will fail" vs "will succeed" by this metric alone; scan-coverage mismatch (very different z-extents between BL and FU, e.g. `fu_sz=(512,512,441)` vs `bl_sz=(512,512,55)`) is a better visual predictor but wasn't reduced to a clean automatic rule.

**Tried and rejected** (all tested against all 36 failing cases in parallel, ~48 CPUs, since each attempt fails/times out in under 2 minutes — no 20+ minute waits needed):

- `--no-refine`: no effect, same failure.
- `--no-body-mask`: **causes a native segfault** (core dump) instead of a clean failure — worse, not better.
- **Automatic retry with landmark-based pre-alignment on elastix failure** (implemented, tested, then reverted): for the 16 zov>0 cases with usable CSV landmarks, retrying with the same landmark-rigid pre-alignment used for the zov≤0 cases got 7/13 applicable cases past the *immediate* sampling error. But on full runs, none of those 7 reached a real success: 3 failed again later (same underlying issue, just delayed ~1–2min instead of ~1min), and 4 **segfaulted** (confirmed reproducible for at least one case, `21addf5898_00`). Net result: zero confirmed recoveries, plus a worse failure mode (native crash vs. clean skip) for over half the cases it touched. Reverted — not committed.

**Conclusion:** these 36 cases have a real geometric registration difficulty (large/variable mismatch in scan coverage between BL and FU), not a parameter-tuning problem solvable with the flags this CLI already exposes. Fixing them would need either a fundamentally different coarse-alignment strategy (e.g. a multi-start/global search, or a learned initializer) or manual per-case QC — both out of scope for this backfill. Flagging as follow-up work, not attempting further without a clearer signal.

## Final tally

| Status | Count |
|---|---:|
| Fully registered | 272 |
| Legitimately unregisterable (no FU landmarks) | 8 |
| Elastix registration failures (unresolved) | 36 |
| **Total pairs** | **316** |

Effective ceiling for this dataset without further registration R&D: **272/308** registerable pairs (excluding the 8 that can never be registered) = **88%**.
