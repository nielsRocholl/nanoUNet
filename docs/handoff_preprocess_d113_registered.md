# Handoff: preprocess Dataset113_Longitudinal_CT_registered (DWB longi finetune)

**Written 2026-07-03, mid-task, on a slow-storage node. Picking this up on a node with a
faster connection to the storage node (`//blissey.umcn.nl/oncology` → `/nnunet_data`).**

## Goal (one line)

Finish CPU preprocessing of the 2-channel registered longitudinal CT dataset
(`Dataset113_Longitudinal_CT_registered`) so it's ready for the `nanounet_train --longi` DWB
finetune. **Steps 1–2 of the pipeline are already done. Only step 3 (preprocess) + step 4
(longi_clicks) remain.** The previous node did preprocessing at a crawl (~1 case/min); the whole
point of switching nodes is to run it much faster.

## Where we are exactly

| Thing | State |
|---|---|
| Raw 2-ch dataset built (`longi_build`) | ✅ done — `imagesTr` 544 files (272×`_0000`FU+`_0001`warpedBL), `labelsTr` 272, `clicksTr` 272 |
| Preprocessed dir staged (dataset.json + patched plans) | ✅ done — both files present |
| Preprocess (raw → blosc2 `.b2nd`/`.pkl` + centroids) | ⏳ **~20/272 done**, needs finishing |
| `longi_clicks` sidecars (`*_bl_clicks.json`) | ❌ not started (step 4) |

A slow fallback run (`nanounet_preprocess … -np 2 --resume`, pid on the old node) may still be
inching forward on CIFS. It writes into the **same** output dir, and preprocess is idempotent with
`--resume` (only wipes the dir when `--resume` is **absent**), so it's safe — but on the old node it
will likely be dead by the time you read this. Just rerun with `--resume` (below); it skips the
cases already done.

## Exact paths

```
DATASET_ID   = 113          # NOT 13: Dataset013_Longitudinal_CT (the unregistered uclp-pro
                            # dataset) owns id 13; convert_id_to_dataset_name(13) raises
                            # "ambiguous dataset id" once both exist.
DS_FOLDER    = Dataset113_Longitudinal_CT_registered
PLANS_NAME   = nnUNetResEncUNetLPlans_h200_smallpv

RAW_DS  = /nnunet_data/nnUNet_raw/Dataset113_Longitudinal_CT_registered
PRE_DS  = /nnunet_data/NanoUNet_preprocessed/Dataset113_Longitudinal_CT_registered
OUT_DIR = $PRE_DS/nnUNetPlans_3d_fullres          # where the .b2nd/.pkl land (data_identifier)
CLICKS  = $RAW_DS/clicksTr                         # warped BL clicks, one <stem>.json per case
```

Env (nanoUNet reads `NANOUNET_*`, NOT `nnUNet_*` — hard `EnvironmentError` if unset):

```bash
export NANOUNET_RAW=/nnunet_data/nnUNet_raw
export NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results
```

Repo: `git@github.com:nielsRocholl/nanoUNet.git`, branch `main`, HEAD `7409f86`. Use `python3`
(there is no `python` on PATH in the container). Committed reference script for all of this:
`scripts/slurm_nanounet_preprocess_d113_registered.sh` (do steps 3+4 by hand, or just run it — it
re-does steps 1–2 too but those are no-ops / cheap re-stages).

## Why it was slow + kept crashing (READ THIS before cranking `-np`)

- **Crash:** `nanounet_preprocess` with `-np 4`, `-np 8`, `-np 24` all died with
  `RuntimeError: preprocess worker died` (`nanounet/plan/preprocess.py:85`). No traceback, no core
  dump (`ulimit -c` = 0) → a native segfault in a worker, reproducible only at higher process
  counts. Only `-np 1` and `-np 2` ran stably. Strong suspicion: **concurrent CIFS I/O** (many
  spawn-pool workers reading big NIfTIs + writing blosc2 to the SMB mount at once), not RAM (node
  had 377 GB, barely touched) and not a per-case bug (a single case processes fine — verified).
- **Slow:** at `-np 2` each ~66 MB × 2-channel case (resampled volume ~578×520×515) took ~30–60 s,
  so 272 cases ≈ hours.

Cases are large. Verify with a quick `du -sh $RAW_DS` (was 72 GB).

## Recommended fast approach: preprocess on LOCAL disk, then copy back

The old node had a fast local disk (`/`, 5.8 TB free) but a slow CIFS link. If your new node also
has local scratch, this both (a) removes the CIFS-concurrency segfault and (b) is far faster.
**First just try higher `-np` straight against CIFS** — on a fast-storage node the segfault may not
reproduce. If it still dies above `-np 2`, fall back to local-disk staging below.

### Option A — straight to CIFS at higher parallelism (try this FIRST)

```bash
cd /root/nanounet        # or wherever the fresh clone is
export NANOUNET_RAW=/nnunet_data/nnUNet_raw
export NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

nanounet_preprocess -d 113 --skip-fingerprint --skip-plan \
  --plans-name nnUNetResEncUNetLPlans_h200_smallpv -np 16 --resume
```

Watch for `preprocess worker died` in the first ~2 min. If it survives, let it finish. If it dies,
drop to Option B (don't just keep lowering `-np` — go local).

### Option B — local-disk staging (reliable + fast if CIFS concurrency still segfaults)

```bash
LOCAL=/root/nano_local        # any fast local scratch with ~150 GB free
mkdir -p "$LOCAL/raw" "$LOCAL/pre/Dataset113_Longitudinal_CT_registered"

# input reads: symlink raw (reading from CIFS single-threaded-ish is fine) OR rsync it local for
# max speed. rsync is safest against the segfault:
rsync -a /nnunet_data/nnUNet_raw/Dataset113_Longitudinal_CT_registered/ \
        "$LOCAL/raw/Dataset113_Longitudinal_CT_registered/"

# stage dataset.json + already-patched plans into the LOCAL preprocessed dir
cp /nnunet_data/NanoUNet_preprocessed/Dataset113_Longitudinal_CT_registered/dataset.json \
   /nnunet_data/NanoUNet_preprocessed/Dataset113_Longitudinal_CT_registered/nnUNetResEncUNetLPlans_h200_smallpv.json \
   "$LOCAL/pre/Dataset113_Longitudinal_CT_registered/"

cd /root/nanounet
export NANOUNET_RAW="$LOCAL/raw"
export NANOUNET_PREPROCESSED="$LOCAL/pre"
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

nanounet_preprocess -d 113 --skip-fingerprint --skip-plan \
  --plans-name nnUNetResEncUNetLPlans_h200_smallpv -np 24 --resume

# copy the finished preprocessed case dir + centroids + gt back to CIFS
rsync -a "$LOCAL/pre/Dataset113_Longitudinal_CT_registered/" \
        /nnunet_data/NanoUNet_preprocessed/Dataset113_Longitudinal_CT_registered/
```

Note: `nanounet_preprocess` also writes `gt_segmentations/` and per-case centroid JSONs into the
data_identifier dir at the end — the `rsync -a` of the whole `Dataset113…` folder covers all of it.
`convert_id_to_dataset_name(113)` scans raw+preprocessed+results; with local raw+pre and CIFS
results all naming `Dataset113…` there's no conflict.

## Step 4 — longi_clicks (REQUIRED, do after preprocess finishes, points at CIFS)

Once `$OUT_DIR` on CIFS has 272 `.pkl` (i.e. after the copy-back in Option B, or directly in
Option A), map the warped BL clicks into preprocessed voxel space:

```bash
cd /root/nanounet
export NANOUNET_RAW=/nnunet_data/nnUNet_raw
export NANOUNET_PREPROCESSED=/nnunet_data/NanoUNet_preprocessed
export NANOUNET_RESULTS=/nnunet_data/NanoUNet_results

python3 -m nanounet.cli.longi_clicks -d 113 \
  --plans nnUNetResEncUNetLPlans_h200_smallpv \
  --clicks-dir /nnunet_data/nnUNet_raw/Dataset113_Longitudinal_CT_registered/clicksTr
```

Fast (no multiprocessing, just coordinate math). Writes `<case>_bl_clicks.json` next to each
preprocessed case: `{"bl_clicks_zyx":[...], "has_baseline": bool}`. It **asserts** every mapped
click lands in-bounds (crashes loudly on wrong axis order — that's expected safety, not a bug to
work around). Prints `cases: N  with baseline clicks: M` at the end — report those numbers.

## Verification (do all, report results)

1. `find $OUT_DIR -name '*.pkl' | wc -l` → **272**.
2. Sample case has 2 channels:
   ```python
   from nanounet.data.blosc2_dataset import Blosc2Folder
   d,s,p = Blosc2Folder.load_case('<OUT_DIR>', '<somecase>')  # data.shape[0] must == 2
   ```
   (`<somecase>` e.g. `006f52e910_00`.)
3. Every case has a sidecar: `ls $OUT_DIR/*_bl_clicks.json | wc -l` → **272**.
4. Plans still has both channel norm entries:
   `python3 -c "import json;p=json.load(open('$PRE_DS/nnUNetResEncUNetLPlans_h200_smallpv.json'));c=p['configurations']['3d_fullres'];print(c['normalization_schemes'],c['use_mask_for_norm'],list(p['foreground_intensity_properties_per_channel']))"`
   → `['CTNormalization','CTNormalization'] [0,0] ['0','1']`.

## Do NOT

- Do **not** rerun `nanounet_preprocess` **without** `--resume` — that wipes `$OUT_DIR` and restarts
  from zero.
- Do **not** re-run `longi_build` / re-plan / re-fingerprint — raw dataset + plans are final and
  correct. `--skip-fingerprint --skip-plan` are mandatory (reuses the Dataset999 plans verbatim
  except channel-1 = copy of channel-0 CT stats, already patched into `$PRE_DS`).
- Do **not** use id 13 (ambiguous — see above).

## After this: training (next task, not now)

`scripts/slurm_nanounet_finetune_d013_registered.sh` (needs its id/paths updated to 113 like the
preprocess script was) →
`nanounet_train -d 113 -f 0 --plans nnUNetResEncUNetLPlans_h200_smallpv --config
configs/finetune_d013.json --init-weights <Dataset999 stage-2 last.ckpt> --longi …`. See
`docs/longitudinal_dwb_design.md` §12 and the user's handoff notes.
