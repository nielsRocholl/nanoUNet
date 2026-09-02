# Handoff: dataset cleaning (speck filter + MSWAL/MCT_LTDiag cancer-only) — 2026-09-02

You have no memory of the conversation that produced this. Read it fully before touching anything —
two workstreams are mid-flight on the same training data, one has a live background agent still
running, and there is a shared doc both workstreams write to.

## Goal

nanoUNet trains a promptable universal lesion segmentation model (nnU-Net-based) for a MedIA
submission. Training data: `/nnunet_data/NanoUNet_raw/DatasetXXX_Name` (21 datasets, ids 011–031,
all binary `{background:0, lesion:1}`). Two data-quality problems were found and are being fixed:

1. **Tiny connected-component "specks"** in labels across most of the 21 datasets — almost
   certainly annotation/segmentation-tool artifacts, not real disease.
2. **Non-cancer classes mislabeled as "lesion"** in two specific datasets (MSWAL, MCT_LTDiag).

## Key fact that motivates all of this

`nanounet/prompt/centroids.py` (`_one_case`) runs `cc3d.connected_components` on every label file
with **zero minimum-size filter**, and every component becomes a first-class clickable "lesion
instance" (centroid + EDT seed + bbox + volume) consumed by `nanounet/data/instance_target.py` to
build the promptable click-training curriculum. A 1-voxel artifact isn't just noise sitting in a
mask — it's an active training example teaching the model "a click here should segment ~nothing."
This is why the speck problem matters more than ordinary label noise.

---

## Workstream 1: voxel-count speck filter (all 21 datasets)

### Decisions already made, don't re-litigate

- **Filter by voxel count, not mm³.** Spacing varies 5–10x across these 21 datasets
  (~0.25–4.5 mm³/voxel). The artifact mechanism (annotation-tool noise, resampling/morphology
  slivers, thin-neck CC splits) is a voxel-space phenomenon, not anatomical — a fixed voxel floor
  is scale-invariant; mm³ conflates real small anatomy at coarse resolution with pure noise at
  fine resolution. User explicitly rejected an organ-mask-gating (TotalSegmentator) approach
  earlier in the conversation — don't revisit that.
- **cc3d, 26-connectivity** — matches `centroids.py`/`instance_target.py`'s own definition of "a
  component," so cleaning uses the same definition the training pipeline will apply downstream.
- **Floor = 3 voxels** tested as the primary candidate (5 and 10 also scanned for comparison).
- **Mandatory backup before any write, always**, verified by file count (not by trusting library
  return values — see gotcha below). Re-running `apply` always re-reads from the backup, never
  from a possibly-already-modified live file, so it's idempotent.

### Tooling (all already written and working)

- `/nnunet_data/NanoUNet_raw/speck_filter.py` — `backup` / `dry_run` / `apply` subcommands.
  Parallelized (`ProcessPoolExecutor`, 18 workers). Full docstring at the top of the file explains
  the design.
- `/nnunet_data/raw/labelsTr_backup_pre_speck_filter/DatasetXXX_Name/labelsTr/` — **all 21
  datasets' original label files already backed up** (6,016 files verified by count).
- `/nnunet_data/NanoUNet_raw/speck_filter_dry_run.json` — full-corpus (not sampled) dry-run result,
  all 6,016 cases, floors 3/5/10. Took 5.3 minutes.
- `/nnunet_data/NanoUNet_raw/find_flagged.py` — recomputes per-file flagged-component detail
  (case, centroid, voxel count) for one dataset+floor at a time; the summary JSON above only has
  aggregate counts, not per-case detail.
- `/nnunet_data/NanoUNet_raw/flagged_Dataset012_LNDb_floor3.json` and
  `flagged_Dataset027_MCT_LTDiag_floor3.json` — per-case detail already computed for these two
  (see "open questions" below for why).
- `/nnunet_data/NanoUNet_raw/dataset_provenance.md` — source, lesion/cancer type, phase notes,
  documented clinical size thresholds, and a small-lesion risk flag for all 21 datasets. Built
  specifically to judge whether a given dataset's small components are real disease (e.g. LNDb,
  LIDC, LiTS, Mediastinal-LN, PanTS all have documented reasons to contain real small lesions) or
  not (GIST, Adrenal_ACC, KiTS23, MCT_LTDiag have no such documented rationale).

### CIFS-mount gotcha — will bite you again if you forget it

`/nnunet_data` is a CIFS network mount (`//blissey.umcn.nl/oncology`). It rejects `chmod`/`utime`
(`EPERM`) even though the actual data write succeeds. `shutil.copytree`'s default `copy2` (and even
its final directory-level `copystat` call) will raise `shutil.Error` on **every file**, even though
the bytes landed correctly. Fix already applied in `speck_filter.py`:
`copy_function=shutil.copyfile` (skips metadata), plus a `try/except shutil.Error: pass` around the
whole `copytree` call, with correctness verified independently by comparing file counts
src-vs-backup afterward. If you write any new copy logic against this mount, use the same pattern —
don't trust a clean return from `shutil`/`os` copy calls here, verify by count/hash instead.

### Dry-run findings — what's safe, what isn't

At floor 3, **zero cases go empty in 20 of 21 datasets** (CLM, Longitudinal_CT, MSD_Colon,
MSD_Lung, MSD_Pancreas, MSD_Liver, MSWAL, WAW_TACE, WORC_CRLM, WORC_GIST, KiTS23, LiTS, LIDC,
RUMC_Bone, RUMC_Pancreas, MCT_LTDiag, PanTS, RIDER_LungCT, Mediastinal_LN, Adrenal_ACC) — safe to
apply. Per-dataset case/CC/voxel counts are in `speck_filter_dry_run.json`.

**`Dataset012_LNDb` is held out and NOT safe to blanket-apply.** 20 of 236 cases would become
*fully* empty at floor 3 — every single component in those cases is ≤3 voxels. That's suspicious:
LNDb's own protocol (per `dataset_provenance.md`) only fully segments nodules ≥3mm diameter, which
at LNDb's resolution should be ~38+ voxels, not 1–3. Visual QC (orthogonal-slice crops, sent to the
user, case list + centroid voxel coords in `flagged_Dataset012_LNDb_floor3.json`) showed a mixed
picture: many flagged foci sit at vessel branch points (classic false-positive shape — a vessel
cross-section looks like a tiny round nodule on one slice), a couple looked like plausible isolated
real tiny nodules. **This needs an actual radiologist/expert look at the 20 cases before any
decision — do not auto-apply or auto-exclude.**

**`MCT_LTDiag` floor is undecided, leaning toward "raise it above 3."** 151–211/516 cases affected
depending on floor (3/5/10), but **0 cases go empty at any tested floor** — every flagged case
keeps a larger surviving component. A 12-case visual sample (case list + coords in
`flagged_Dataset027_MCT_LTDiag_floor3.json`) showed several flagged tiny components sitting right
at the edge of an already-large annotated tumor in the same slice — consistent with "thin-neck
fragment split off the main tumor by cc3d," not an independent small lesion, which would justify a
higher floor here specifically. Not conclusively proven from 12 cases though — user wanted a bigger
sample before committing to a number. **Do this next**, then decide (candidates discussed: try 10,
maybe higher).

### Not yet applied

`speck_filter.py apply` **has not been run on anything**. Only `backup` and `dry_run` (both
read-only w.r.t. the live dataset) have executed. Nothing in `/nnunet_data/NanoUNet_raw` has been
modified by this workstream yet.

### What's left to do here

1. Get a bigger MCT_LTDiag visual sample (or radiologist input) and settle a floor for it
   (candidate: something above 3, maybe 10 — not decided).
2. Get LNDb's 20 would-go-empty cases reviewed by someone qualified to read chest CT. Decide:
   drop those cases' tiny nodules as noise, or keep them (and if kept, `Dataset012_LNDb` may need
   a different/no floor entirely, or per-case handling).
3. Run `python speck_filter.py apply --floor 3 --datasets <comma-separated 3-digit ids>` for the 20
   clean datasets (all except `012`) — **but check workstream 2 first**: if MSWAL/MCT_LTDiag's
   cancer-only relabel (below) has already replaced `Dataset018_MSWAL`/`Dataset027_MCT_LTDiag`'s
   content, the existing backup+dry-run for those two is now **stale** (backed up the *old*
   pre-cancer-only labels). Re-run `speck_filter.py backup` then `dry_run` for just those two
   datasets against the new content before applying speck-filter to them. The `backup` subcommand
   currently skips re-backup if file *count* matches the source — that's not a strong enough check
   if content changed but count happens to coincide; verify manually or strengthen the check.
4. After `apply` actually runs: it will print any case that became fully empty (it does NOT
   auto-drop them). Apply the same "drop empty case, fix `dataset.json` numTraining" treatment
   already established for `Dataset028_PanTS` (see `/nnunet_data/raw/UCLP_CONVERSION_PLAN.md`).
5. Once any label file changes (from this workstream or workstream 2), `Dataset999_Merged` and the
   preprocessed centroid sidecars (built by `nanounet_preprocess`) are stale and must be rebuilt —
   this is a deliberately separate, bigger step, not done automatically. Reuse the
   mandatory-backup-refuse-if-exists pattern already in
   `/nanoUNet/scripts/run_preprocess_sidecars.sh`.

---

## Workstream 2: MSWAL + MCT_LTDiag cancer-only relabel

### Why

- **`Dataset018_MSWAL`**: source `labelsTr` carries separate per-class integers (0=bg, 1=gallstone,
  2=kidney stone, 3=liver tumor, 4=kidney tumor, 5=pancreatic cancer, 6=liver cyst, 7=kidney cyst).
  The shipped `uclp-pro` config (`configs/MSWAL/mswal.yaml`) used `labels_to_keep: [3,4,5,6,7]` —
  it dropped stones but **wrongly kept both cyst classes**. Fix: `labels_to_keep: [3,4,5]` only.
- **`Dataset027_MCT_LTDiag`**: mixes 4 malignant subtypes (HCC, ICC, CRLM, BCLM) with **benign
  hepatic hemangioma (HH)** under one binary label. Harder problem: the HF-mirrored source
  (`MCT-LTDiag/{case}/mask_pvp.nii.gz`) is **already a flat binary merge** — no per-class or
  per-case diagnosis metadata anywhere in the HF repo. Filtering can only be case-level (drop whole
  hemangioma-diagnosis cases), never voxel-level. True original source: Peking Union Medical
  College Hospital, *Scientific Data* 2025
  (https://www.nature.com/articles/s41597-025-06343-4), data on Harvard Dataverse — the fix
  requires that paper/dataset publishing a per-case diagnosis table, and the case IDs matching
  what's in the local mirror (e.g. `230218a1`, `230218a10`).

### Governing rule — already established project convention

`/nnunet_data/raw/UCLP_CONVERSION_PLAN.md` (a pre-existing decision log for a *different*, already
completed conversion — Adrenal-ACC/HCC-TACE/Mediastinal-LN/RIDER/PanTS → Dataset028-031) states:
*"if a future source fails this check, stop and flag it rather than guessing which component is the
lesion."* Apply this literally to MCT_LTDiag: no heuristic guessing (size/shape/location as a
hemangioma proxy) is acceptable. Only a verified, real per-case diagnosis mapping justifies
filtering; otherwise leave the dataset untouched and report the limitation.

### Tooling / locations

- `/uclp-pro` — separate git repo (github.com/nielsRocholl/uclp-pro), the conversion tool. Adapters
  `mswal` and `mct_ltdiag` already exist. Read `README.md`, `ADAPTERS.md`,
  `AGENT_ADD_DATASET.md` before touching configs. **Do not edit shipped configs in place, do not
  git-commit there unless asked.**
- Source data: private HF dataset repo `nielsRocholl/universal-lesion-segmentation`
  (`huggingface_hub` 0.34.3 installed, already authenticated as user nielsRocholl, scoped read
  token). Download convention: `/nnunet_data/raw/MSWAL/`, `/nnunet_data/raw/MCT-LTDiag/`.
- `/nnunet_data/raw/UCLP_CONVERSION_PLAN.md` — **shared doc, both workstreams append sections to
  it.** Always `Read` immediately before `Edit`/append — don't blind-write, another process may
  have appended since you last looked. Never overwrite existing content, only append new dated
  sections.

### Live background agent — verify before trusting

**Agent name/id: `ad096021fe5d1fd51`** (spawned via the `Agent` tool, `general-purpose` type). Use
`ListAgents` to check if it's still running. **This agent has twice needed correction** for
inaccurately reporting task state (it once claimed "a monitor is watching the download" when no
such monitor existed — just bare untracked OS processes that nothing would have resumed). Do not
trust its self-reports; verify directly:
```
ps aux | grep -E "snapshot_download|uclp_preprocess"
find /nnunet_data/raw/MSWAL -type f -not -path '*/.cache/*' | wc -l      # target: 971
find /nnunet_data/raw/MCT-LTDiag -type f -not -path '*/.cache/*' | wc -l # target: 3620
```
Last observed at handoff time: MSWAL download essentially done (~971/971 or very close),
MCT-LTDiag download ~23% done (832/3620) and slow (~34 files/min observed) — could need another
hour+ from here. If the agent looks stalled or the downloads finished but nothing progressed, use
`SendMessage` to `ad096021fe5d1fd51` with the real, verified state (not what it last claimed) and
tell it to continue the pipeline itself.

A duplicate/redundant `snapshot_download` process for MSWAL was found and killed mid-session (two
processes racing on the same target dir — harmless but wasteful). If you see duplicates again for
either dataset, same deal: check which one the agent's own wait-loop is actually watching (grep its
log files for what it polls) and kill the other.

### Outstanding question sent to the agent, not yet answered

We kept only portal-venous-phase (`pvp.nii.gz`) for MCT_LTDiag. A **different** dataset,
`HCC-TACE-Seg`, was already excluded from this same overall pipeline specifically because arterial-
phase rim enhancement is the defining HCC sign and fades/vanishes by portal venous phase (see the
"HCC-TACE-Seg — excluded" remark near the top of `UCLP_CONVERSION_PLAN.md`). MCT_LTDiag's malignant
subtypes include HCC too. Asked the agent to check whether the source paper documents masks being
drawn per-phase vs. drawn once and registered onto all four phases — if the latter, some (esp.
small/early) HCC could be genuinely invisible in the PVP image at the annotated voxels. Agent was
told not to block the main pipeline on this, just fold the answer into its final report. **Check
whether it answered this; if not, it still needs answering before MCT_LTDiag's cancer-only version
is trusted for training.**

### What's left to do here

1. Check on the agent (see above), verify real state, nudge if needed.
2. Get the PVP-visibility answer (above).
3. Once both conversions are done: verify `Dataset018_MSWAL`/`Dataset027_MCT_LTDiag` were
   backed up (refuse-if-exists pattern) before replacement, verify `UCLP_CONVERSION_PLAN.md` got a
   new appended section (not an overwrite), verify `dataset.json` `numTraining` was corrected for
   any zero-foreground cases dropped after the label filter.
4. Feed back into workstream 1: re-run speck-filter backup+dry_run for these two datasets against
   their new content (see workstream 1, item 3).
5. Flag (don't do automatically) that `Dataset999_Merged`/preprocessed sidecars need rebuilding.
