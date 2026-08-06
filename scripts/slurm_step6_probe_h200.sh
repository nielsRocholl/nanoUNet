#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=nanounet-step6-probe
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_step6_probe.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_step6_probe.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# STEP 6 PROBE -- does making the target click-conditional actually teach selectivity?
#
# THE QUESTION, AND THE ONLY THING THIS RUN ANSWERS
#   On the fixed val manifest the 600-epoch checkpoint scores
#     val/subset_clicked/val_selectivity_margin = -0.2709
#   i.e. clicking one of three lesions produces a mask that matches "segment all three" 27 Dice
#   points better than "segment the one you clicked". Instance-conditional targets + click dropout
#   are supposed to fix that. This run is 80 epochs to see the number MOVE, not to produce a model.
#
# WARM START, NOT FROM SCRATCH -- deliberate.
#   --init-weights from the measured baseline isolates the objective change: 80 epochs from MAE
#   init would still be climbing basic Dice and would say nothing about selectivity. The cost is
#   that this is not a clean training curve; it is a diagnostic.
#
# NO COHORT WEIGHTS HERE -- also deliberate.
#   sampling.cohorts is left empty so the probe changes exactly one thing. Reweighting and the new
#   objective at once would make a null result uninterpretable.
#
# LOWER LR THAN THE MAIN RUN. Warm-starting at 0.01 with momentum 0.99 would kick the net hard
# enough to confound the read. 0.003 with 5 warmup epochs moves it without destroying it.
#
# EXPECT TRAIN LOSS TO RISE relative to the old objective. The same lesion is foreground in one
# patch and background in another; the task stays well-posed (the click channel differs) but the
# shortcut is gone. Judge ONLY on the val manifest strata below.
#
# SUCCESS (all four, on the fixed manifest):
#   val/subset_clicked/val_selectivity_margin   -0.2709 -> positive
#   val/none_clicked/val_pred_fg                 0.0196 -> toward 0
#   val_prompt_gap                               0.0819 -> higher
#   val/all_clicked/val_dice                     0.8390 -> holds (watch for over-suppression here)
#
# ALSO CLOSE THE GPU GATE HERE: sample `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1`
# for >=3 epochs and compare epoch_wall_time_sec against the baseline run. The instance path was
# measured at ~1.9% of the per-patch budget in isolation; the >95% rule has never been checked on
# a real training loop.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
BASE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200/checkpoints/best-epoch=570-val_dice=0.8030.ckpt"
PROBE_EPOCHS=80
ITERS_PER_EPOCH=1000
BATCH_SIZE=12
PROMPTS_PER_PATCH=2
CONSISTENCY_WEIGHT=0.02   # re-scoped, not dropped: the kept lesion set is drawn ONCE per patch so
                          # both variants share a target and the term measures click PLACEMENT only.
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
if [ ! -f "$BASE_CKPT" ]; then
  echo "FATAL: baseline checkpoint not found: $BASE_CKPT"
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
  echo "Fix: confirm --include \"valset_1500*\" is in the rclone call above, and that"
  echo "     ${REMOTE_PREP}/valset_1500.json exists (build it with nanounet_build_valset)."
  exit 1
fi

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_step6probe"
rm -rf "$OUT"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/instance_conditional.json \
  --init-weights "$BASE_CKPT" \
  --val-manifest "$VAL_MANIFEST" \
  --val-every-n-epochs 2 \
  --out "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$PROBE_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --lr 0.003 \
  --warmup-epochs 5 \
  --lr-schedule poly \
  --monitor val_dice \
  --loss dc_ce \
  --prompts-per-patch "$PROMPTS_PER_PATCH" \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --dl-bucket xl \
  --dl-persistent-workers \
  --devices 1 \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_step6probe_instance_targets_80ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
