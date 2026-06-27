#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=248G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-train-mae-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_train_mae_999.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_train_mae_999.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv

MAE_EPOCHS=1000
SUP_EPOCHS=2000
ITERS_PER_EPOCH=250
VAL_ITERS=50

STORAGE=/nnunet_data

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export NANOUNET_RAW="${STORAGE}/NanoUNet_raw"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
mkdir -p "$NANOUNET_RESULTS"

if ! nanounet_train --help &>/dev/null; then
  echo "FATAL: nanounet_train not found or broken."
  exit 1
fi

LOCAL_PREP=/root/NanoUNet_preprocessed
mkdir -p "$LOCAL_PREP"
if ! rclone copy "${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress \
  --transfers 32 \
  --multi-thread-streams 16 \
  --no-update-modtime \
  --retries 5 \
  --copy-links; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}"
# Host: .../NanoUNet_results/.../epoch=452-step=113250.ckpt (not last.ckpt — overwritten by 10-ep tests)
MAE_CKP="${OUT}/mae_pretrain/checkpoints/epoch=452-step=113250.ckpt"
SUP_CKP="${OUT}/checkpoints/last.ckpt"
RESUME_ARGS=(--mae-resume "$MAE_CKP")
[[ -f "$MAE_CKP" ]] || { echo "FATAL: MAE checkpoint missing: $MAE_CKP"; exit 1; }
[[ -f "$SUP_CKP" ]] && RESUME_ARGS+=(--resume "$SUP_CKP")

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --mae-pretrain \
  --mae-epochs "$MAE_EPOCHS" \
  --mae-lr-schedule cosine_warm_restarts \
  --mae-cosine-t0 250 \
  --epochs "$SUP_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --lr 1e-3 \
  --lr-schedule stretched_tail_poly \
  --loss cc_dc_ce \
  --dl-bucket m \
  --dl-persistent-workers \
  --accelerator cuda \
  --precision 16-mixed \
  "${RESUME_ARGS[@]}"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi
rm -rf "$LOCAL_PREP/${DS_FOLDER}"
