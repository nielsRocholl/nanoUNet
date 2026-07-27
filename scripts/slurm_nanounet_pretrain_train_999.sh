#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=248G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-sup-999-warmstart
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_warmstart.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_warmstart.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# Reuses the existing self-supervised (MAE) checkpoint via --mae-ckpt instead of rerunning MAE
# pretraining -- the MAE stage is prompt-free and unaffected by the prompt-robustness changes.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
MAE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt"
SUP_EPOCHS=600
ITERS_PER_EPOCH=1000
VAL_ITERS=50
# Placeholder -- weight to be chosen by measurement (W9 gate / held-out eval), not guessed.
CONSISTENCY_WEIGHT=0.0
PROMPTS_PER_PATCH=1
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
export nnUNet_preprocessed="$LOCAL_PREP"

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}"

# fresh run: clear only the supervised checkpoints. $MAE_CKPT lives under mae_pretrain/ of this
# same run dir ($OUT) and is reused via --mae-ckpt -- do NOT rm -rf "$OUT", that would delete it.
rm -rf "$OUT/checkpoints"

echo "batch_size in plans: $(python3 -c "import json; print(json.load(open('${LOCAL_PREP}/${DS_FOLDER}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['batch_size'])")"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --mae-ckpt "$MAE_CKPT" \
  --epochs "$SUP_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --lr-schedule stretched_tail_poly \
  --stretched-k 188 \
  --stretched-ref 250 \
  --loss dc_ce \
  --dl-bucket l \
  --dl-persistent-workers \
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_sup_warmstart_bs4_xl_dcce_1000it_sup600ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
