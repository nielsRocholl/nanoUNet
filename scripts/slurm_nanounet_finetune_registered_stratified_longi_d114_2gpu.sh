#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-finetune-d114-registered-2gpu
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d114_registered_2gpu.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d114_registered_2gpu.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# Longi finetune of Dataset114 (unigradicon-registered union clicks), A100-40GB (dlc-arceus).
#
# THROUGHPUT (2026-07-27, supersedes the old dataloader-bound note): rendering prompt
# heatmaps AFTER augmentation cut SpatialTransform to 2 CT channels instead of 6. Measured
# on one A100-40GB with bucket `l` and --prompts-per-patch 2: steady-state GPU utilisation
# 99.2% (was 84%), 0.96 s/iter (was 1.05). Do NOT drop below bucket `l` for longi.

# ---------------------------------------------------------------------------------------------
# 2-GPU VARIANT. NOT RUNNABLE YET -- prepared ahead of the code change.
#
# `devices=1` is currently hardcoded in nanounet/train/fit.py (two places) and there is no
# --devices CLI flag, so `--devices 2` below will be rejected until that lands.
#
# Before running this, the multi-GPU work in
# /nnunet_data/prompt_sensitivity/HANDOFF_RETRAIN.md section 4 must be done, in particular the
# SILENT bug: PatchIterable.__iter__ shards by worker id only, not by global rank, so under DDP
# both GPUs draw the IDENTICAL patch sequence. It does not crash -- it trains happily doing half
# the data at twice the cost, and no loss curve reveals it. Verify per-rank case ids differ before
# trusting any throughput number from this script.
#
# Effective batch stays 6 (3 per GPU) so LR needs no rescaling and VRAM headroom improves; the
# single-GPU run peaked at 39161/40960 MiB.
# ---------------------------------------------------------------------------------------------

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
# Placeholder -- weight to be chosen by measurement (W9 gate / held-out eval), not guessed.
CONSISTENCY_WEIGHT=0.0
PROMPTS_PER_PATCH=1
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
echo "lr=$LR (batch size taken from plans file)"

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

WANDB_NAME="Dataset114_registered_f0_finetune_dwb_adamw${LR}_500ep_2gpu"

nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/finetune_d013.json \
  --init-weights "$INIT_CKPT" \
  --longi \
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
  --dl-bucket l \
  --dl-persistent-workers \
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --accelerator cuda \
  --devices 2 \
  --precision 16-mixed \
  --wandb-name "$WANDB_NAME" || { rm -rf "$LOCAL_PREP/${DS_FOLDER}"; exit 1; }

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
