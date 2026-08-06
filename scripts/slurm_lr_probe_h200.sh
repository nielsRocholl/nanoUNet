#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=nanounet-lr-probe
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_lr_probe_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_lr_probe_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# STEP 5a -- LR PROBE. Submit this THREE times, once per learning rate:
#
#   PROBE_LR=0.005 sbatch scripts/slurm_lr_probe_h200.sh
#   PROBE_LR=0.01  sbatch scripts/slurm_lr_probe_h200.sh
#   PROBE_LR=0.03  sbatch scripts/slurm_lr_probe_h200.sh
#
# Optionally a fourth on the winner with PROBE_MOMENTUM_NOTE below.
#
# WHY THIS EXISTS. lr=0.01 / momentum=0.99 are inherited from nnU-Net's BATCH-2 regime. This runs 6
# distinct patches per step, so gradient variance is much lower than that momentum was chosen to
# smooth -- 0.99 averages over ~100 steps, about 600 patches, which is over-damped. Do not settle
# this by argument; the handoff is explicit that it is measured.
#
# COMPARE AT EPOCH 60 ON THE FIXED MANIFEST. This comparison is only readable because the val set
# is fixed: 3 runs x 60 epochs against the old per-epoch random draw would have been noise.
# Report a TABLE of val_dice (and val/all_clicked/val_dice) at epoch 60. Not a recommendation.
#
# NOTE ON MOMENTUM. --optimizer sgd hardcodes momentum 0.99 in lightning_module.py:configure_optimizers.
# Testing 0.97 needs a one-line change there or a new flag; it is a follow-up, not part of this probe.
#
# CONFIG. Uses configs/longrun.json so the probe measures the LR under the objective and mixture the
# long run will actually use. If the Step 6 probe has NOT yet passed, run this against
# configs/default.json instead and say so when reporting -- an LR tuned under one objective does not
# automatically transfer to another.

set -euo pipefail

PROBE_LR="${PROBE_LR:?set PROBE_LR, e.g. PROBE_LR=0.005 sbatch $0}"
FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
MAE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt"
PROBE_EPOCHS=60
ITERS_PER_EPOCH=1000
BATCH_SIZE=12
PROMPTS_PER_PATCH=2
CONSISTENCY_WEIGHT=0.02
CONFIG="${PROBE_CONFIG:-configs/longrun.json}"
STORAGE=/nnunet_data

export PIP_CACHE_DIR=/root/.pip-cache
export NANOUNET_RAW="${STORAGE}/NanoUNet_raw"
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
echo "lr=$PROBE_LR  config=$CONFIG  data_identifier=$DATA_ID   staging ~543 GB"

if ! rclone copy "$REMOTE_PREP/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress --transfers 32 --multi-thread-streams 16 --no-update-modtime --retries 5 --copy-links \
  --include "${PLANS_NAME}.json" \
  --include "splits_final.json" \
  --include "valset_1500*" \
  --include "${DATA_ID}/**"; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"
export nnUNet_preprocessed="$LOCAL_PREP"

VAL_MANIFEST="${LOCAL_PREP}/${DS_FOLDER}/valset_1500.json"
if [ ! -f "$VAL_MANIFEST" ]; then
  echo "FATAL: val manifest not staged: $VAL_MANIFEST"
  exit 1
fi

TAG=$(echo "$PROBE_LR" | tr -d '.')
OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_lrprobe_${TAG}"
rm -rf "$OUT"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config "$CONFIG" \
  --mae-ckpt "$MAE_CKPT" \
  --val-manifest "$VAL_MANIFEST" \
  --val-every-n-epochs 2 \
  --out "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$PROBE_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --lr "$PROBE_LR" \
  --warmup-epochs 5 \
  --monitor val_dice \
  --lr-schedule poly \
  --loss dc_ce \
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --dl-bucket xl \
  --dl-persistent-workers \
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_lrprobe_lr${PROBE_LR}_60ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
