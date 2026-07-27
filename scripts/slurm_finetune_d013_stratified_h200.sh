#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-ft-d013-strat-h200
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_ft_d013_stratified_h200.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_ft_d013_stratified_h200.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nanounet-sol-docker:latest"

# STAGE 3 -- STRATIFIED FINETUNE ON LONGITUDINAL-CT ONLY, SINGLE-TIMEPOINT.
# Single H200 (dlc-slowpoke, 141 GB). Successor to nanoUNet-finetrune_stratified.sh, which produced
# the previous best model.
#
# Three things to understand:
#  1. NOT a separate dataset. This is -d 999 with --only-prefix d013_, filtering the merged pool to
#     the Longitudinal-CT cases. Each timepoint (BL, FU) is its own single-timepoint case there.
#  2. NO --longi. Ordinary one-stream prompted model, not the two-stream DWB longitudinal model.
#     The d114 *_longi_* scripts are a different line of work.
#  3. "Stratified" means the *_weights.json sidecars driving lesion-type-weighted sampling.
#     COVERAGE IS PARTIAL: 426 of 537 d013 cases have weights (dated 2026-06-18); the other 111
#     sample UNIFORMLY. Rerun nanounet_lesion_weights first if you want full stratification.
#     The check below prints coverage and only hard-fails when NONE are present.
#
# Single GPU means no DDP: no rank sharding, no cross-rank metric reduction, no unused-parameter
# handling, no effective-batch doubling. Four classes of subtle failure do not exist here.
#
# BATCH SIZE 12 (rows), --prompts-per-patch 2 -> 6 DISTINCT patches per step. batch_size counts ROWS;
# two prompts share one patch and one augmentation pass, so rows/2 is the distinct-patch count.
# Previously validated: 6 rows / 6 patches (1 prompt). Verified today: 6 rows / 3 patches.
# Activations measured 31.1 GiB at 6 rows and scale ~linearly -> expect ~62 GiB of 141.
#
# BUCKET xl, NOT l -- the load-bearing choice on this node. On the A100 a step took 0.96 s needing
# ~3.1 patches/s, which bucket l (8 workers) just met. An H200 step should need ~6 patches/s, so l
# would starve it. xl is 16 train / 8 val workers, ~32-40 cores at peak, hence cpus-per-task=64.
# If GPU util sags anyway, the bottleneck is the data path -- do not raise the batch to mask it.
#
# STORAGE: dlc-slowpoke has a slow link to /nnunet_data, but this stage stages only the d013_ cases
# (~50 GB, not 543 GB), so the one-time cost is modest.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
ONLY_PREFIX=d013_
FT_EPOCHS=500
LR=1e-5                   # warm-start finetune. 12 rows vs the 6 this LR was set for, but paired
                          # rows are correlated so the real change is much smaller than 2x.
ITERS_PER_EPOCH=1000
VAL_ITERS=50
BATCH_SIZE=12             # rows; must divide by PROMPTS_PER_PATCH. 12 rows / 2 prompts = 6 DISTINCT
                          # patches per step -- exactly the patch diversity of the validated runs.
                          # Paired rows are highly correlated (same crop, same augmentation, same
                          # target; only the click differs), so the effective batch increase is far
                          # less than 12/6 suggests. Activations ~62 GiB of 141 -- ample.
PROMPTS_PER_PATCH=2       # two independent click draws per patch, sharing one crop + one augmentation
CONSISTENCY_WEIGHT=0.02   # measured, not guessed: train_loss_seg averages ~0.047 while the raw
                          # consistency term sits at 0.79-0.82, so 0.02 puts consistency at ~20-25%
                          # of total loss. Revisit from the logged train_loss_seg /
                          # train_loss_consistency ratio; val_prompt_gap -> 0 means it is too high.
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

LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"
DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID   staging only ${ONLY_PREFIX}* cases (~50 GB)"

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

# The empirical click model needs volume_vox in the centroid sidecars; fail now, not 400 steps in.
if ! python3 -c "
import glob, json, sys
f = sorted(glob.glob('$CASE_DIR/${ONLY_PREFIX}*_centroids.json'))[:20]
sys.exit(0 if f and all('volume_vox' in json.load(open(x)) for x in f) else 1)"; then
  echo "FATAL: centroid sidecars lack volume_vox. Fix: bash scripts/run_preprocess_sidecars.sh"
  exit 1
fi

INIT_CKPT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_h200/checkpoints/last.ckpt"
if [ ! -f "$INIT_CKPT" ]; then
  echo "FATAL: stage-2 checkpoint not found: $INIT_CKPT"
  echo "Fix: run scripts/slurm_supervised_999_h200.sh first."
  exit 1
fi

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_h200_finetune_${ONLY_PREFIX%_}"
rm -rf "$OUT"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/finetune_d013.json \
  --init-weights "$INIT_CKPT" \
  --only-prefix "$ONLY_PREFIX" \
  --out "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$FT_EPOCHS" \
  --optimizer adamw \
  --lr "$LR" \
  --wd 3e-5 \
  --grad-clip 1.0 \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --lr-schedule poly \
  --loss dc_ce \
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --dl-bucket xl \
  --dl-persistent-workers \
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_finetune_d013_stratified_h200_bs12_promptfix_500ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
