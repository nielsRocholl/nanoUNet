# Handoff — `nanounet_segtrack` (26 Aug 2026)

**Machine:** `dlc-mewtwo`. **Audience:** next agent / human with no session context.

**Status:** RAM + `.mha` product is on `origin/main`. BL-mask mode (skip BL UNet, GT ids) is **implemented and gated, not committed**.

Living spec: `/lesion-tracking/.cursor/plans/segtrack_e2e.md` (Parts 1–3). CLI docs: `/nanoUNet/docs/steps/track.md`.

---

## Env (this box)

```bash
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results
# already installed:
#   pip install -e /nanoUNet
#   pip install -e /lesion-tracking
```

| Path | What |
|------|------|
| `/nanoUNet` | seg CLI + UNet |
| `/lesion-tracking` | matcher (`tracking.*`) |
| `/nnunet_data/Longitudinal-CT` | CIFS (`blissey`): `inputsTrBL/FU`, `targetsTrBL/FU`, `meta/`, `test_patients.csv` |
| `/scratch` | local ZFS — use for timed runs |
| `/tmp` | tmpfs — fine for one-case gates |

Defaults (no `-m` / `--track-ckpt` needed):

- seg: `$NANOUNET_RESULTS/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_instance_1200ep`
- matcher: `/nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt`

`inputsTrBL` ∩ `inputsTrFU` is **316** stems, not the full folders. Folder mode without `--patients-csv` dies on stem mismatch. Use the CSV.

nanoUNet `origin` is HTTPS; push via `git@github.com:nielsRocholl/nanoUNet.git` (SSH). Do not `git remote set-url` unless asked.

---

## What landed today

### 1. Product = RAM + `.mha` (pushed)

| Repo | Commit | Message |
|------|--------|---------|
| nanoUNet | `b4f435c` | Keep segtrack in RAM and write linked masks as `.mha` |
| lesion-tracking | `fa75f74` | Pass volumes into `track()` and paint with a LUT |

Per case dir:

| File | Meaning |
|------|---------|
| `bl.mha` | BL instance mask, ids **unchanged** (sitk zyx, uint8/int16) |
| `fu.mha` | FU instance mask, ids **painted** so same integer = same lesion |
| `matches.csv` | `bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id` |

`--keep-pred` → `pred_bl.mha` / `pred_fu.mha` (binary FG). Mask mode writes **only** `pred_fu.mha`.

Nets load **once**. Default `--decode hungarian`. FU click JSON = matcher BL coordinates (`cog_propagated`). Paint: `fu_track_map` then LUT `paint_fu`. Empty side → zeros + CSV header, no `track()`.

**Axis (do not mix):** sitk / native pred / CC / `write_seg` / clicks = **zyx**. nib `_load_vol` / L0 / centroids = **xyz**. Convert once: `np.ascontiguousarray(inst_zyx.transpose(2, 1, 0))` before `track()`.

### 2. `--bl-mask` / `--bl-mask-dir` (uncommitted)

Skip BL UNet. Load a native BL **instance** mask (voxel = `lesion_id`). Predict FU only. Match. Paint FU with those BL ids → `fu.mha` shares the GT id namespace (`targetsTrFU`).

- Single: `--bl-img --bl-mask --fu-img --fu-clicks`. **No** `--bl-clicks`.
- Folder: `--bl-dir --fu-dir --bl-mask-dir`. BL dir is CT-only (JSON not required). FU still needs sibling JSON.
- BL **CT** still required (L0). Matcher still needs FU JSON points for every BL mask id (already true on this dataset: GT ⊂ FU JSON names).

Files: `nanounet/infer/segtrack.py`, `segtrack_case.py`, `cli/segtrack.py`, `cli/segtrack_cases.py`. **No** lesion-tracking code change.

Gate (`01161aaa0b_00`, `/tmp`, hungarian, no meta): `bl.mha` sitk-equal to `targetsTrBL`; 1 pair; `fu.mha` id `7`; **16s** (one UNet). Output left at `/tmp/segtrack_blmask/01161aaa0b_00`.

---

## Run on this machine

### Predict both timepoints (click-CC BL ids — **not** GT namespace)

```bash
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results

nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.nii.gz \
  --bl-clicks /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.json \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.json \
  -o /tmp/segtrack_both/01161aaa0b_00 --overwrite
```

Writes `/tmp/segtrack_both/01161aaa0b_00/{bl,fu}.mha` + `matches.csv`.

Crowded smoke (52 clicks; ~2 min one-pass FU+BL): stem `03b90eb112_00`, same flag pattern.

### GT BL mask + predict FU (ids comparable to `targetsTrFU`)

```bash
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results

nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.nii.gz \
  --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/01161aaa0b_00.nii.gz \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.json \
  -o /tmp/segtrack_blmask/01161aaa0b_00 --overwrite
```

Then open `fu.mha` vs `/nnunet_data/Longitudinal-CT/targetsTrFU/01161aaa0b_00.nii.gz`. Same integer = same lesion. Pred ≠ GT voxels; namespace is the point.

### Holdout folder (GT BL)

```bash
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results

nanounet_segtrack \
  --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL \
  --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU \
  --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL \
  --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv
```

Default out: `$NANOUNET_RESULTS/segtrack/inputsTrFU/{stem}/`. Resume skips cases that already have `matches.csv` unless `--overwrite`.

---

## Git (now)

**nanoUNet `main`:** HEAD `b4f435c` (RAM/`.mha`) is on GitHub; local `origin/main` may lag because push used the SSH URL, not HTTPS `origin`. **Uncommitted** (BL-mask):

- modified: `cli/segtrack.py`, `infer/segtrack.py`, `docs/steps/track.md`, `docs/reference/track_ids.md`, `docs/index.md`
- new: `cli/segtrack_cases.py`, `infer/segtrack_case.py`

**lesion-tracking `main`:** `fa75f74` = `origin`. Uncommitted: `.cursor/plans/segtrack_e2e.md` Part 3.

Do not push nanoUNet HTTPS; use SSH URL.

---

## Do not

- Merge the two repos
- Auto-load `meta/*.csv`
- Add `--gt-dir` scoring inside `nanounet_segtrack` (compare `fu.mha` vs `targetsTrFU` outside)
- Prefetch / second CUDA net / `--disable-tta` as a “speedup”
- Mix sitk zyx with nib xyz
- Quote CIFS read times as “our gzip”
- Pass `--bl-clicks` with `--bl-mask`
- Point `--bl-mask` at a **registered/warped** BL mask (must be native `targetsTrBL` grid)

`graphify` is **not** on PATH on this host.
