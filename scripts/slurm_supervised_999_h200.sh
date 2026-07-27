#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-sup-999-h200
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_h200.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_h200.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# STAGE 2 -- SUPERVISED, SINGLE-TIMEPOINT, on the merged pool. Single H200 (dlc-slowpoke, 141 GB).
# No --longi: ordinary one-stream prompted model, NOT the two-stream DWB longitudinal model.
#
# WHY ONE H200 INSTEAD OF TWO A100s
#   Single GPU means no DDP: no rank sharding, no cross-rank metric reduction, no unused-parameter
#   handling, no effective-batch doubling. Four classes of subtle failure simply do not exist.
#   It also occupies one card instead of two.
#
# BATCH SIZE 12 (rows), with --prompts-per-patch 2 -> 6 DISTINCT patches per step.
#   batch_size counts ROWS, not patches. Two prompts share one patch and one augmentation pass, so
#   rows/2 is the number of distinct patches. For reference: the previously validated runs were
#   6 rows / 6 patches (1 prompt each); today's verified config was 6 rows / 3 patches. 12 rows
#   restores that 6-patch diversity exactly, while adding the consistency pairing.
#   Activations measured 31.1 GiB at 6 rows and scale ~linearly, so expect ~62 GiB of 141 -- ample.
#
# BUCKET xl, NOT l -- THIS IS THE LOAD-BEARING CHOICE ON THIS NODE.
#   On the A100 a step took 0.96 s and needed ~3.1 patches/s, which bucket l (8 workers) just met at
#   99% util. An H200 step should land near 0.6-0.7 s and need ~6 patches/s, so l would starve it.
#   xl is 16 train / 8 val workers, ~32-40 cores at peak, hence cpus-per-task=64.
#   If GPU util still sags, the bottleneck is the data path, not the GPU -- do not raise the batch.
#
# STORAGE: dlc-slowpoke has a slow link to /nnunet_data. Staging ~543 GB is a ONE-TIME cost inside
# this job's wall time; it does not affect per-step throughput once training starts.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
MAE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt"
SUP_EPOCHS=600
ITERS_PER_EPOCH=1000
VAL_ITERS=50
BATCH_SIZE=12             # rows; must divide by PROMPTS_PER_PATCH. 12 rows / 2 prompts = 6 DISTINCT
                          # patches per step -- exactly the patch diversity of the validated runs.
                          # Paired rows are highly correlated (same crop, same augmentation, same
                          # target; only the click differs), so the effective batch increase is far
                          # less than 12/6 suggests. Activations ~62 GiB of 141 -- ample.
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
echo "data_identifier: $DATA_ID   staging ~543 GB over a slow link -- expect this to take a while"

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

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_h200"
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
  --dl-bucket xl \
  --dl-persistent-workers \
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_sup_h200_bs12_promptfix_600ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
