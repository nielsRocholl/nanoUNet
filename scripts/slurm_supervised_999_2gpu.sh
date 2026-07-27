#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-sup-999-2gpu
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_2gpu.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_2gpu.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nanounet-sol-docker:latest"

# STAGE 2 -- SUPERVISED, SINGLE-TIMEPOINT, on the full merged pool (Dataset999_Merged).
# No --longi: this trains the ordinary one-stream prompted model, NOT the two-stream DWB
# longitudinal model. Stage 3 (finetune on d013 only) is slurm_finetune_d013_stratified_2gpu.sh.
#
# The self-supervised (MAE) stage is REUSED, not rerun: it is prompt-free and unaffected by any of
# the prompt-robustness work, so --mae-ckpt points at the existing checkpoint and --mae-pretrain is
# deliberately absent. That saves 250 epochs.
#
# 2 GPUs (DDP). Effective batch 12 = 6 rows per rank; each GPU does what the verified single-GPU run
# did. NOTE this is DOUBLE the single-GPU effective batch of 6 -- a deliberate choice for the
# wall-clock win, made with the LR left unscaled.
#
# Measured on this code, 2x A100-40GB, bucket l, --prompts-per-patch 2:
#   per-row throughput 1.5x single-GPU; GPU util medians 97% / 100%; peak VRAM 40317/40960 MiB.
# VRAM headroom is thin (~600 MiB). If it OOMs, drop to --prompts-per-patch 2 with --batch-size 4
# (effective 8) rather than reducing prompts, which would disable the consistency term.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
MAE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt"
SUP_EPOCHS=600
ITERS_PER_EPOCH=1000
VAL_ITERS=50
PROMPTS_PER_PATCH=2       # two independent click draws per patch, pairing rows for the consistency term
CONSISTENCY_WEIGHT=0.02   # measured, not guessed: train_loss_seg averages ~0.047 while the raw
                          # consistency term sits at 0.79-0.82, so 0.02 puts consistency at ~20-25%
                          # of total loss magnitude. Revisit from the logged
                          # train_loss_seg / train_loss_consistency ratio after a few epochs;
                          # val_prompt_gap collapsing toward 0 means it is too high.
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
if [ ! -f "$MAE_CKPT" ]; then
  echo "FATAL: MAE checkpoint not found: $MAE_CKPT"
  exit 1
fi

# /nnunet_data is a slow CIFS mount -- stage the preprocessed pack to local disk first.
LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"
DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID  (staging ~543 GB, this is inside the job's wall time)"

if ! rclone copy "$REMOTE_PREP/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress --transfers 32 --multi-thread-streams 16 --no-update-modtime --retries 5 --copy-links \
  --include "${PLANS_NAME}.json" \
  --include "splits_final.json" \
  --include "${DATA_ID}/**"; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"
export nnUNet_preprocessed="$LOCAL_PREP"

# The empirical click model needs volume_vox in the centroid sidecars; fail loudly, not 400 steps in.
if ! python3 -c "
import glob, json, sys
f = sorted(glob.glob('$LOCAL_PREP/$DS_FOLDER/$DATA_ID/*_centroids.json'))[:20]
sys.exit(0 if f and all('volume_vox' in json.load(open(x)) for x in f) else 1)"; then
  echo "FATAL: centroid sidecars lack volume_vox. Fix: bash scripts/run_preprocess_sidecars.sh"
  exit 1
fi

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}"
rm -rf "$OUT/checkpoints"   # supervised checkpoints only -- $OUT also holds mae_pretrain/, which we reuse

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
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --dl-bucket l \
  --dl-persistent-workers \
  --devices 2 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_sup_2gpu_promptfix_600ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
