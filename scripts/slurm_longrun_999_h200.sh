#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-longrun-999
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_longrun_999.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_longrun_999.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# STEP 5b -- THE SINGLE STRATIFIED SUPERVISED STAGE. No finetune stage; that is removed on purpose.
#
# DO NOT LAUNCH THIS UNTIL THE STEP 6 PROBE HAS PASSED.
#   scripts/slurm_step6_probe_h200.sh must first show val/subset_clicked/val_selectivity_margin
#   moving from -0.2709 to positive while val/all_clicked/val_dice holds near 0.839. This run is
#   ~7 days of H200; do not spend it on an objective that has not been shown to work.
#   The LR probe (Step 5a) should also have run -- see LR below.
#
# WHY 1200 EPOCHS. The 600-epoch val_dice curve is a smooth concave rise with a clearly positive
# slope at 600 and no knee; a log fit over 200->600 predicts +4 to +7 Dice from doubling. 1200 is
# the conservative end of the handoff's 1200-1800 range: 1800 is a 3x budget resting on the same
# single log fit, and the WSD option below is the better way to buy more if 1200 is not enough.
#
# STRETCHED-TAIL RETUNED. --stretched-k/--stretched-ref are ABSOLUTE epoch counts, not fractions.
# Reusing 188/250 at 1200 epochs would put epochs 188-1200 (84% of the run) in the linear tail.
# Scaled 2x with the horizon: 376/500 keeps the original shape -- poly decay against a 500-epoch
# reference for the first 376 epochs, then linear decay over the remaining 824.
#
# WILL NOT FIT IN ONE 7-DAY JOB. At ~531 s/epoch (454 train + 77 val amortised at every-2-epochs)
# 1200 epochs is ~177 h = 7.4 days. Expect ONE resume: re-submit with RESUME set to the last.ckpt
# below. The script does not delete $OUT when RESUME is set.
#
# NO EARLY STOPPING. The curve never turns over, so it would never fire; the constraint is budget,
# not overfitting.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
MAE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt"

SUP_EPOCHS=1200
ITERS_PER_EPOCH=1000
BATCH_SIZE=12
PROMPTS_PER_PATCH=2
CONSISTENCY_WEIGHT=0.02
LR=0.01                   # REPLACE with the Step 5a probe winner before launching. 0.01 is the
                          # inherited nnU-Net batch-2 value and is very likely wrong at 6 distinct
                          # patches per step; the probe exists because this should not be argued.
WARMUP_EPOCHS=10          # none existed before; step 1 ran at full LR with momentum 0.99 on an
                          # MAE-initialised net.
EMA_DECAY=0.0             # off: the EMA callback currently runs a SECOND full pass over the 1500
                          # patch manifest to log val_dice_ema, and that cost was never measured.
                          # Turn on only after measuring it.
STRETCHED_K=376
STRETCHED_REF=500
RESUME="${RESUME:-}"      # set to .../checkpoints/last.ckpt to continue a timed-out run
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
if [ -z "$RESUME" ] && [ ! -f "$MAE_CKPT" ]; then
  echo "FATAL: MAE checkpoint not found: $MAE_CKPT"
  exit 1
fi

LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"
DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID   staging ~543 GB over a slow link -- expect this to take a while"

# valset_1500* is REQUIRED: without it --val-manifest fails at startup, after the 543 GB stage.
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
  echo "Fix: confirm --include \"valset_1500*\" is in the rclone call above."
  exit 1
fi

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_longrun"
if [ -z "$RESUME" ]; then
  rm -rf "$OUT"
  INIT_ARGS=(--mae-ckpt "$MAE_CKPT")
else
  echo "resuming from $RESUME"
  INIT_ARGS=(--resume "$RESUME")
fi

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/longrun.json \
  --val-manifest "$VAL_MANIFEST" \
  --val-every-n-epochs 2 \
  "${INIT_ARGS[@]}" \
  --out "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$SUP_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --lr "$LR" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --ema-decay "$EMA_DECAY" \
  --monitor val_dice \
  --lr-schedule stretched_tail_poly \
  --stretched-k "$STRETCHED_K" \
  --stretched-ref "$STRETCHED_REF" \
  --loss dc_ce \
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --dl-bucket xl \
  --dl-persistent-workers \
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_longrun_instance_targets_1200ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
