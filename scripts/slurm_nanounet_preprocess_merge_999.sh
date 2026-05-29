#!/bin/bash
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=nanounet-plan-pp-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_plan_preprocess_999.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_plan_preprocess_999.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"
#
# Dataset999_Merged: run planner (--patch-vol small → patch_edge 128) then preprocess.cases (--resume).
# Fingerprint reused (--skip-fingerprint). Point nanounet_train --plans at PLANS_NAME below.
#
set -euo pipefail

NP=25

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

STORAGE=/nnunet_data
export NANOUNET_RAW="${STORAGE}/NanoUNet_raw"
export NANOUNET_PREPROCESSED="${STORAGE}/NanoUNet_preprocessed"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
mkdir -p "$NANOUNET_PREPROCESSED" "$NANOUNET_RESULTS"

IDS=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27)

PLANNER=nnUNetPlannerResEncL
GPU_MEM_GB=100
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv

nanounet_preprocess \
  -d "${IDS[@]}" \
  --merged-id 999 \
  --merged-name Merged \
  --planner "$PLANNER" \
  --gpu-memory-gb "$GPU_MEM_GB" \
  --plans-name "$PLANS_NAME" \
  -np "$NP" \
  --skip-fingerprint \
  --patch-vol small \
  --resume
