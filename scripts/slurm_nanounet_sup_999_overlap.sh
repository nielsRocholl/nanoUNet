#!/bin/bash
# Supervised resume with IO/aug overlap pipeline (bucket l, thread pin).
#
# Validation gate (run once before full 2000-epoch commit):
#   1. Submit this script; stop after ~20 epochs or set SUP_EPOCHS=$((EPOCH_RESUME+20)).
#   2. Check epoch_wall_time_sec (W&B) <= 320 s over epochs 5-20.
#   3. If gate passes, re-submit with SUP_EPOCHS=2000 for the long run.

#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus,dlc-groudon,dlc-meowth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=248G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-sup-999-overlap
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_overlap.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_overlap.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
SUP_EPOCHS=2000
ITERS_PER_EPOCH=250
VAL_ITERS=50
STORAGE=/nnunet_data

export PIP_CACHE_DIR=/root/.pip-cache
export NANOUNET_RAW="${STORAGE}/NanoUNet_raw"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$PIP_CACHE_DIR" "$NANOUNET_RESULTS" "$NANOUNET_TMPDIR"

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
SUP_CKP="${OUT}/checkpoints/last.ckpt"
if [[ ! -f "$SUP_CKP" ]]; then
  echo "FATAL: no supervised checkpoint at ${SUP_CKP}"
  exit 1
fi

echo "Resuming supervised from: $SUP_CKP"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --epochs "$SUP_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --lr 1e-3 \
  --lr-schedule stretched_tail_poly \
  --loss cc_dc_ce \
  --dl-bucket l \
  --dl-persistent-workers \
  --accelerator cuda \
  --precision 16-mixed \
  --resume "$SUP_CKP"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi
rm -rf "$LOCAL_PREP/${DS_FOLDER}"
