#!/bin/bash
#SBATCH --qos=high
#SBATCH --exclude=dlc-groudon,dlc-arceus,dlc-slowpoke,dlc-meowth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=nanounet-plan-pp-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_plan_preprocess_999_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_plan_preprocess_999_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"
#
# Merge raw datasets 010–027 (skip 026) into Dataset999_Merged, fingerprint + ResEnc plan + preprocess.
# Raw reads stay on the shared nnU-Net tree; preprocessed + results go to NanoUNet_* siblings (see env below).
#
# Replace --container-image with an image that has nanoUNet installed, or install before the command, e.g.:
#   pip install -e /path/to/nanoUNet
#
# nanoUNet: one -np for fingerprint pool and per-case preprocess (no separate nnU-Net -npfp / -np split).
# With 48 CPUs, 16 is a safe default; raise toward 32–40 if I/O and RAM stay flat.

set -euo pipefail

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

# Shared raw (same as classic nnU-Net).
export nnUNet_raw=/nnunet_data/nnUNet_raw

# nanoUNet outputs only (do not clobber nnUNet_preprocessed / nnUNet_results).
export nnUNet_preprocessed=/nnunet_data/NanoUNet_preprocessed
export nnUNet_results=/nnunet_data/NanoUNet_results

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

# Optional aliases (sync_nnunet_env maps NANOUNET_* → nnUNet_* if set).
export NANOUNET_RAW="$nnUNet_raw"
export NANOUNET_PREPROCESSED="$nnUNet_preprocessed"
export NANOUNET_RESULTS="$nnUNet_results"

# Source ids: 10–25 and 27 (exclude 26 RUMC_Pancreas). Merge target 999 → Dataset999_Merged.
IDS=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27)

# ResEnc L + 80 GB VRAM target (H200), custom plans basename like the old -overwrite_plans_name.
# nanoUNet has no nnUNetPlannerResEncL_torchres; use nnUNetPlannerResEncL.
PLANNER=nnUNetPlannerResEncL
GPU_MEM_GB=100
PLANS_NAME=nnUNetResEncUNetLPlans_h200
NP=16

nanounet_preprocess \
  -d "${IDS[@]}" \
  --merged-id 999 \
  --merged-name Merged \
  --planner "$PLANNER" \
  --gpu-memory-gb "$GPU_MEM_GB" \
  --plans-name "$PLANS_NAME" \
  -np "$NP"
