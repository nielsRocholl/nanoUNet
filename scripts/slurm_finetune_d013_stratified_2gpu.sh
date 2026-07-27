#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-ft-d013-strat-2gpu
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_ft_d013_stratified_2gpu.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_ft_d013_stratified_2gpu.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nanounet-sol-docker:latest"

# STAGE 3 -- STRATIFIED FINETUNE ON LONGITUDINAL-CT ONLY, SINGLE-TIMEPOINT.
# Successor to nanoUNet-finetrune_stratified.sh, which produced the previous best model.
#
# Three things to understand about this script:
#
# 1. It is NOT a separate dataset. It runs -d 999 with --only-prefix d013_, filtering the merged
#    pool down to the Longitudinal-CT cases. Each timepoint (BL, FU) is its own single-timepoint
#    case in that pool.
# 2. NO --longi. This is the ordinary one-stream prompted model, not the two-stream DWB
#    longitudinal model. The d114 *_longi_* scripts are a different line of work.
# 3. "Stratified" means the *_weights.json sidecars, which drive lesion-type-weighted sampling via
#    centroid_weights. COVERAGE IS PARTIAL: 426 of 537 d013 cases have weights (dated 2026-06-18).
#    The other 111 fall back to UNIFORM lesion sampling silently. If you want full stratification,
#    rerun nanounet_lesion_weights before this. The check below only fails when NONE are staged.
#
# Warm start from the stage-2 supervised checkpoint (slurm_supervised_999_2gpu.sh).
#
# 2 GPUs (DDP), effective batch 12 = 6 rows per rank. NOTE the LR comment below.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
ONLY_PREFIX=d013_
FT_EPOCHS=500
ITERS_PER_EPOCH=1000
VAL_ITERS=50
LR=1e-5                   # warm-start finetune. The original note said "do NOT scale with batch",
                          # written when effective batch was 6. It is 12 here (2 ranks x 6 rows).
PROMPTS_PER_PATCH=2       # two independent click draws per patch, pairing rows for the consistency term
CONSISTENCY_WEIGHT=0.02   # measured, not guessed: train_loss_seg averages ~0.047 while the raw
                          # consistency term sits at 0.79-0.82, so 0.02 puts consistency at ~20-25%
                          # of total loss magnitude. Revisit from the logged
                          # train_loss_seg / train_loss_consistency ratio after a few epochs;
                          # val_prompt_gap collapsing toward 0 means it is too high.
STORAGE=/nnunet_data

export PIP_CACHE_DIR=/root/.pip-cache
export NANOUNET_RAW="${STORAGE}/nnUNet_raw"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export nnUNet_raw="$NANOUNET_RAW"
export nnUNet_results="$NANOUNET_RESULTS"
mkdir -p "$PIP_CACHE_DIR" "$NANOUNET_RESULTS" "$NANOUNET_TMPDIR"

if ! nanounet_train --help &>/dev/null; then
  echo "FATAL: nanounet_train not found or broken."
  exit 1
fi

# /nnunet_data is a slow CIFS mount -- stage locally. Only the d013_ cases are needed here (~50 GB),
# not the whole 543 GB pool.
LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"
DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID   staging only ${ONLY_PREFIX}* cases"

if ! rclone copy "$REMOTE_PREP/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress --transfers 32 --multi-thread-streams 16 --no-update-modtime --retries 5 --copy-links \
  --include "${PLANS_NAME}.json" \
  --include "splits_final.json" \
  --include "${DATA_ID}/${ONLY_PREFIX}**"; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"
export nnUNet_preprocessed="$LOCAL_PREP"

CASE_DIR="$LOCAL_PREP/${DS_FOLDER}/${DATA_ID}"

# Stratification weights. Partial coverage is expected (see header); zero coverage is a mistake.
N_W=$(ls "$CASE_DIR/${ONLY_PREFIX}"*_weights.json 2>/dev/null | wc -l)
N_C=$(ls "$CASE_DIR/${ONLY_PREFIX}"*_seg.b2nd 2>/dev/null | wc -l)
if [ "$N_W" -eq 0 ]; then
  echo "FATAL: no *_weights.json staged -- this run would NOT be stratified."
  echo "Fix: run nanounet_lesion_weights for ${DS_FOLDER} before this job."
  exit 1
fi
echo "stratification: ${N_W}/${N_C} cases carry lesion-type weights; the rest sample uniformly."

# The empirical click model needs volume_vox in the centroid sidecars; fail loudly, not 400 steps in.
if ! python3 -c "
import glob, json, sys
f = sorted(glob.glob('$CASE_DIR/${ONLY_PREFIX}*_centroids.json'))[:20]
sys.exit(0 if f and all('volume_vox' in json.load(open(x)) for x in f) else 1)"; then
  echo "FATAL: centroid sidecars lack volume_vox. Fix: bash scripts/run_preprocess_sidecars.sh"
  exit 1
fi

INIT_CKPT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}/checkpoints/last.ckpt"
OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_finetune_${ONLY_PREFIX%_}"
if [ ! -f "$INIT_CKPT" ]; then
  echo "FATAL: stage-2 checkpoint not found: $INIT_CKPT"
  echo "Fix: run scripts/slurm_supervised_999_2gpu.sh first."
  exit 1
fi
rm -rf "$OUT"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/finetune_d013.json \
  --init-weights "$INIT_CKPT" \
  --only-prefix "$ONLY_PREFIX" \
  --out "$OUT" \
  --epochs "$FT_EPOCHS" \
  --optimizer adamw \
  --lr "$LR" \
  --wd 3e-5 \
  --grad-clip 1.0 \
  --lr-schedule poly \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --loss dc_ce \
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --dl-bucket l \
  --dl-persistent-workers \
  --devices 2 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_finetune_d013_stratified_2gpu_promptfix_500ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
