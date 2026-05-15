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
# Continue preprocess for Dataset999_Merged: fingerprint + plans already present; only missing .b2nd cases run.
# Raw: nnUNet_raw; out: NanoUNet_preprocessed / NanoUNet_results.
#
set -euo pipefail

NP=8

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

export nnUNet_raw=/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/nnunet_data/NanoUNet_preprocessed
export nnUNet_results=/nnunet_data/NanoUNet_results

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

export NANOUNET_RAW="$nnUNet_raw"
export NANOUNET_PREPROCESSED="$nnUNet_preprocessed"
export NANOUNET_RESULTS="$nnUNet_results"

IDS=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27)

PLANNER=nnUNetPlannerResEncL
GPU_MEM_GB=100
PLANS_NAME=nnUNetResEncUNetLPlans_h200

nanounet_preprocess \
  -d "${IDS[@]}" \
  --merged-id 999 \
  --merged-name Merged \
  --planner "$PLANNER" \
  --gpu-memory-gb "$GPU_MEM_GB" \
  --plans-name "$PLANS_NAME" \
  -np "$NP" \
  --skip-fingerprint \
  --skip-plan \
  --resume
