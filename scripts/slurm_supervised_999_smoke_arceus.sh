#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --job-name=nanounet-sup-999-smoke
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_smoke_arceus.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_smoke_arceus.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nanounet-sol-docker:latest"

# SMOKE TEST -- NOT A REAL TRAINING RUN.
# Same code path as slurm_supervised_999_h200.sh (stage 2, single-timepoint, no --longi), shrunk so
# it surfaces errors fast on dlc-arceus, which stages data quickly.
#
# WHAT IT IS FOR
#   Exercising the whole pipeline end to end: staging, the volume_vox sidecar check, the empirical
#   click model, two prompts per patch, the consistency term, the new validation metrics, and
#   checkpoint writing. If this run is clean, the H200 script differs only in batch size, bucket and
#   node -- not in code path.
#
# WHAT IT IS NOT
#   Not a throughput measurement (batch 4 on an A100 is not the H200 config) and not a training
#   signal you should read anything into. 2 epochs of 200 iterations is far too short to mean
#   anything about Dice.
#
# BATCH 4 (rows) with --prompts-per-patch 2 -> 2 distinct patches per step. Deliberately small:
#   activations measured 21.7 GiB at 4 rows, comfortable on a 40 GB A100 with room to spare, so an
#   OOM here would indicate a real problem rather than a tight fit.
#
# IT WRITES TO ITS OWN OUTPUT DIRECTORY (_smoke suffix) and never touches the real stage-2 results.
#
# WHAT TO CHECK IN THE LOG
#   - "stratification" / volume_vox preflight checks pass
#   - train_loss_seg AND train_loss_consistency both logged and non-zero (a zero consistency term
#     means the ramp is broken again -- it was, once)
#   - val_prompt_agreement, val_dice_click_inside/outside, val_prompt_gap all present and finite
#   - GPU memory well under 40 GB

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
MAE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt"
SUP_EPOCHS=2
ITERS_PER_EPOCH=200
VAL_ITERS=10
BATCH_SIZE=4              # rows; 4/2 = 2 distinct patches per step. Small on purpose.
PROMPTS_PER_PATCH=2       # two independent click draws per patch, sharing one crop + one augmentation
CONSISTENCY_WEIGHT=0.02   # measured, not guessed: train_loss_seg averages ~0.047 while the raw
                          # consistency term sits at 0.79-0.82, so 0.02 puts consistency at ~20-25%
                          # of total loss. Revisit from the logged train_loss_seg /
                          # train_loss_consistency ratio; val_prompt_gap -> 0 means it is too high.
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

LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"
DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID   staging ~543 GB (arceus is fast)"

if ! rclone copy "$REMOTE_PREP/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress --transfers 32 --multi-thread-streams 16 --no-update-modtime --retries 5 --copy-links \
  --include "${PLANS_NAME}.json" \
  --include "splits_final.json" \
  --include "${DATA_ID}/**"; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"
export nnUNet_preprocessed="$LOCAL_PREP"

# The empirical click model needs volume_vox in the centroid sidecars; fail now, not 400 steps in.
if ! python3 -c "
import glob, json, sys
f = sorted(glob.glob('$LOCAL_PREP/$DS_FOLDER/$DATA_ID/*_centroids.json'))[:20]
sys.exit(0 if f and all('volume_vox' in json.load(open(x)) for x in f) else 1)"; then
  echo "FATAL: centroid sidecars lack volume_vox. Fix: bash scripts/run_preprocess_sidecars.sh"
  exit 1
fi

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_smoke"  # never the real run dir
rm -rf "$OUT"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --mae-ckpt "$MAE_CKPT" \
  --out "$OUT" \
  --batch-size "$BATCH_SIZE" \
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
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_SMOKE_arceus_bs4_promptfix_2ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
