#!/bin/bash
#SBATCH --qos=vram
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-finetune-d114-registered
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d114_registered.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d114_registered.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# Longi finetune of Dataset114 (unigradicon-registered union clicks), A100-40GB only.
#
# GPU-STARVATION NOTE (2026-07-10): this job is dataloader-bound, not compute-bound.
# Each longi sample decompresses a 2-channel volume and pushes a 6-channel patch through
# the CPU augment chain (SpatialTransform + GaussianBlur dominate, ~520 ms/patch). Bucket
# `m` (4 workers) under-feeds the A100 -> ~60% GPU util, ~1400 s/epoch. Bucket `l` (8
# workers) is the proven config that kept the d013 longi finetune at 95%+; on 48 cores it
# restores ~95% util and ~<=1000 s/epoch. Do NOT drop below `l` for longi.
# Full diagnosis: docs/dev-notes/longi_gpu_starvation.md

set -euo pipefail

FOLD=0
DATASET_ID=114
DS_FOLDER=Dataset114_longi
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
BASE_DS_FOLDER=Dataset999_Merged
FT_EPOCHS=500
ITERS_PER_EPOCH=1000
VAL_ITERS=50
LR=1e-5          # fixed on ALL nodes; do NOT scale with batch (warm-start finetune)
STORAGE=/nnunet_data

export NANOUNET_RAW="${STORAGE}/nnUNet_raw"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export nnUNet_raw="$NANOUNET_RAW"
export nnUNet_results="$NANOUNET_RESULTS"
mkdir -p "$NANOUNET_RESULTS" "$NANOUNET_TMPDIR"

LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"

DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")

# A100-40GB only: batch size straight from the plans file (=6). No H200/batch-10 branch.
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['batch_size'])")
echo "batch_size=$BATCH_SIZE lr=$LR"

rclone copy "$REMOTE_PREP/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress --transfers 32 --multi-thread-streams 16 --no-update-modtime --retries 5 --copy-links \
  --include "${PLANS_NAME}.json" \
  --include "splits_final.json" \
  --include "${DATA_ID}/**"

export NANOUNET_PREPROCESSED="$LOCAL_PREP"
export nnUNet_preprocessed="$LOCAL_PREP"

INIT_CKPT="${NANOUNET_RESULTS}/nanounet/${BASE_DS_FOLDER}_${PLANS_NAME}_f${FOLD}/checkpoints/last.ckpt"
OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_finetune_dwb"
rm -rf "$OUT"

WANDB_NAME="Dataset114_registered_f0_finetune_dwb_adamw${LR}_bs${BATCH_SIZE}_500ep"

nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/finetune_d013.json \
  --init-weights "$INIT_CKPT" \
  --longi \
  --out "$OUT" \
  --epochs "$FT_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --optimizer adamw \
  --lr "$LR" \
  --wd 3e-5 \
  --grad-clip 1.0 \
  --lr-schedule poly \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --loss dc_ce \
  --dl-bucket l \
  --dl-persistent-workers \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "$WANDB_NAME" || { rm -rf "$LOCAL_PREP/${DS_FOLDER}"; exit 1; }

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
