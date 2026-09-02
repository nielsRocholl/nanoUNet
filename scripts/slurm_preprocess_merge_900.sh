#!/bin/bash
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --mem=200G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=nanounet-preprocess-merge-900
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_preprocess_merge_900.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_preprocess_merge_900.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"
#
# Builds Dataset900_Merged from the 21 site-tagged raw datasets (011..031): fingerprint, ResEncL
# plan (--patch-vol small -> verified to reproduce patch size 96x160x160), then preprocess.cases.
#
# NO GPU REQUESTED. --gpu-memory-gb below is only arithmetic input to the planner's patch-shrink
# loop (it targets a hypothetical 48 GB card for patch sizing) -- preprocessing itself never
# touches a GPU, so this job asks for none.
#
# -np 12, NOT 38 (the cpus-per-task ceiling): this step is memory-bound, not cpu-bound. At 200G
# mem and 12 workers that is ~16.7 GB/worker steady-state, but large volumes can peak at ~50 GB/
# worker during resampling -- running all 38 cores as workers would blow the 200G ceiling on the
# first big volume. 16 leaves headroom for that peak.
#
# --resume makes an OOM-killed run cheap to restart: already-preprocessed cases are skipped, only
# the remainder is redone. If workers still die (dmesg / the .err log shows a killed worker),
# lower -np (e.g. 12 or 8) and resubmit the same command -- do not raise --mem past 200G, that is
# this user's hard reservation ceiling on this cluster.
#
# CRITICAL PATH: raw root is ${STORAGE}/NanoUNet_raw (capital N, capital U). The older 999 merge
# script wrongly pointed at ${STORAGE}/nnUNet_raw -- a DIFFERENT, unrelated directory. Datasets
# 011-031 live under NanoUNet_raw; get this wrong and the guards below catch it before the
# expensive step, not nanounet_preprocess itself.

set -euo pipefail

STORAGE=/nnunet_data
IDS=(11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
MERGED_ID=900
MERGED_NAME=Merged
PLANNER=nnUNetPlannerResEncL
GPU_MEM_GB=48
PATCH_VOL=small
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
NP=12

export NANOUNET_RAW="${STORAGE}/NanoUNet_raw"
export NANOUNET_PREPROCESSED="${STORAGE}/NanoUNet_preprocessed"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
# No nnUNet_* exports: nanounet never imports nnunetv2 and reads only NANOUNET_*.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Pin BLAS/OMP fan-out to 1 thread each so the 16 preprocess workers do not each spawn their own
# thread pool across the 38 available cores and thrash each other.

mkdir -p "$NANOUNET_PREPROCESSED" "$NANOUNET_RESULTS" \
  /data/oncology/experiments/universal-lesion-segmentation/logs

# ---- fail-fast guards: everything checkable at t=0, before the expensive step -------------------

if ! nanounet_preprocess --help &>/dev/null; then
  echo "FATAL: nanounet_preprocess not found or broken (--help failed)." >&2
  echo "Fix: confirm the nanounet package is installed in this container image." >&2
  exit 1
fi

if [ ! -d "$NANOUNET_RAW" ]; then
  echo "FATAL: NANOUNET_RAW does not exist: $NANOUNET_RAW" >&2
  echo "Fix: confirm the container mount and that ${STORAGE}/NanoUNet_raw (capital N, capital U)" >&2
  echo "     is the intended raw root -- NOT ${STORAGE}/nnUNet_raw." >&2
  exit 1
fi

MISSING=()
for id in "${IDS[@]}"; do
  padded=$(printf "%03d" "$id")
  if ! compgen -G "$NANOUNET_RAW/Dataset${padded}_*" > /dev/null; then
    MISSING+=("Dataset${padded}_*")
  fi
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "FATAL: missing source dataset folder(s) under $NANOUNET_RAW:" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  echo "Fix: confirm all of datasets 011-031 are present before merging." >&2
  exit 1
fi

merged_padded=$(printf "%03d" "$MERGED_ID")
for root_var in NANOUNET_RAW NANOUNET_PREPROCESSED NANOUNET_RESULTS; do
  root="${!root_var}"
  if compgen -G "${root}/Dataset${merged_padded}_*" > /dev/null; then
    echo "FATAL: merged id ${MERGED_ID} is not free under \$${root_var} (${root})." >&2
    echo "Fix: pick a different --merged-id, or remove the existing Dataset${merged_padded}_* there first." >&2
    exit 1
  fi
done

# Rough floor: 21 raw CT datasets resampled to 3d_fullres blosc2 comfortably needs >500 GB free
# on the preprocessed volume. Warn hard rather than fail 20 hours into resampling.
FREE_GB=$(df --output=avail -BG "$NANOUNET_PREPROCESSED" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 500 ]; then
  echo "FATAL: only ${FREE_GB}G free at $NANOUNET_PREPROCESSED, need >=500G for a 21-dataset merge." >&2
  echo "Fix: free space on that volume, or point NANOUNET_PREPROCESSED elsewhere, before resubmitting." >&2
  exit 1
fi

echo "== nanounet_preprocess merge -> Dataset${merged_padded}_${MERGED_NAME} =="
echo "sources: ${IDS[*]} (all present under $NANOUNET_RAW)"
echo "free space at \$NANOUNET_PREPROCESSED: ${FREE_GB}G"
echo "-np $NP workers (memory-bound: ~16.7G/worker steady-state at --mem=200G, ~50G/worker peak)"

# ---- the expensive step --------------------------------------------------------------------------

nanounet_preprocess \
  -d "${IDS[@]}" \
  --merged-id "$MERGED_ID" \
  --merged-name "$MERGED_NAME" \
  --planner "$PLANNER" \
  --gpu-memory-gb "$GPU_MEM_GB" \
  --patch-vol "$PATCH_VOL" \
  --plans-name "$PLANS_NAME" \
  -np "$NP" \
  --resume

echo "== done: Dataset${merged_padded}_${MERGED_NAME} preprocessed under \$NANOUNET_PREPROCESSED =="
echo "Produced: dataset_fingerprint.json, ${PLANS_NAME}.json, ${PLANS_NAME}/3d_fullres/*.b2nd + *_centroids.json,"
echo "          splits_final.json, cohorts.json (see docs/steps/preprocess.md)."
echo "Next: nanounet_train -d ${MERGED_ID} -f 0 --plans ${PLANS_NAME} --config configs/default.json"
