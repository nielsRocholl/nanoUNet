#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-finetune-d013-registered
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d013_registered.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d013_registered.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

FOLD=0
DATASET_ID=113
DS_FOLDER=Dataset113_Longitudinal_CT_registered
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
BASE_DS_FOLDER=Dataset999_Merged          # warm-start (stage-2 single-stream) checkpoint lives here
FT_EPOCHS=500
ITERS_PER_EPOCH=1000
VAL_ITERS=50
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
  echo "FATAL: nanounet_train not found or broken."; exit 1
fi

LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"

DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID"

# Stage plans, splits, and ALL preprocessed d013-registered case files. The ${DATA_ID}/** glob pulls
# b2nd, seg, pkl, centroids, optional weights AND the required *_bl_clicks.json sidecars
# (nanounet_longi_clicks, run in the preprocess job). No --only-prefix: the whole dataset is d013.
if ! rclone copy "$REMOTE_PREP/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress --transfers 32 --multi-thread-streams 16 --no-update-modtime --retries 5 --copy-links \
  --include "${PLANS_NAME}.json" \
  --include "splits_final.json" \
  --include "${DATA_ID}/**"; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"
export nnUNet_preprocessed="$LOCAL_PREP"

# Fail loudly if the warped-clicks sidecars are missing (nanounet_longi_clicks not run).
if ! ls "$LOCAL_PREP/${DS_FOLDER}/${DATA_ID}/"*_bl_clicks.json >/dev/null 2>&1; then
  echo "FATAL: no *_bl_clicks.json staged; run nanounet_longi_clicks in the preprocess job first."
  exit 1
fi

SRC="${NANOUNET_RESULTS}/nanounet/${BASE_DS_FOLDER}_${PLANS_NAME}_f${FOLD}"
INIT_CKPT="${SRC}/checkpoints/last.ckpt"
OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_finetune_dwb"

if [ ! -f "$INIT_CKPT" ]; then
  echo "FATAL: init checkpoint not found: $INIT_CKPT"; exit 1
fi

rm -rf "$OUT"

echo "batch_size in plans: $(python3 -c "import json; print(json.load(open('${LOCAL_PREP}/${DS_FOLDER}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['batch_size'])")"

# Two-stream BL+FU encoder, DWB at skips, warm-started from the stage-2 single-stream net.
# Warped BL is the 2nd input channel (voxel-aligned). Add --longi-null for the S8.1 collapse ablation.
if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/finetune_d013.json \
  --init-weights "$INIT_CKPT" \
  --longi \
  --out "$OUT" \
  --epochs "$FT_EPOCHS" \
  --optimizer adamw \
  --lr 1e-5 \
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
  --wandb-name "Dataset013_registered_f0_finetune_dwb_adamw1e-5_500ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
