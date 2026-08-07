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
# ================== WHAT CHANGED SINCE THE 600-EPOCH RUN ==================
#  1. --config configs/longrun.json   NEW OBJECTIVE. Targets are click-conditional: only lesions
#     that received a click are foreground. 20% of lesions in a patch deliberately go unclicked
#     (click_modes.pos 0.80) so "visible lesion, no click, leave it alone" is actually trained.
#     Reason: on the fixed val set the 600-epoch model scored
#     val/subset_clicked/val_selectivity_margin = -0.2709, i.e. clicking ONE of three lesions
#     produced a mask matching "segment all three" 27 Dice points better than "segment the one you
#     clicked". The click was not selecting anything.
#  2. Same config carries COHORT WEIGHTS. Liver was 40% of training (spread over 7 datasets) and
#     lung 20% -- 61% of the data was two sites. Now balanced by lesion site, with d013 at 25%
#     (deployment target, was 9%) and the starved sites lifted: bone 2.6->6%, colon 2.1->4%,
#     pancreas 4.8->6%, GIST 4.2->5%. Nothing is oversampled more than 2.7x.
#  3. --val-manifest + --val-every-n-epochs 2   FIXED validation set: 1500 pinned patches over four
#     prompt scenarios, identical every epoch. The old per-epoch random draw made every curve noisy
#     and could not see selectivity at all. Because the noise is gone, validating every 2nd epoch
#     costs ~3% of run time instead of ~18% and still gives 600 points.
#  4. 1200 epochs, stretched tail RETUNED to 376/500. --stretched-k/--stretched-ref are ABSOLUTE
#     epoch counts, not fractions: reusing 188/250 here would put 84% of the run in the linear tail.
#     376/500 preserves the original shape -- poly decay for 376 epochs, then 824 epochs of slow
#     linear decay (69% of the run at low LR).
#  5. --warmup-epochs 10. There was no warmup: step 1 ran at full LR with momentum 0.99 on an
#     MAE-initialised net.
#  6. --monitor val_dice, explicit. The old code silently switched to val_dice_macro whenever
#     --init-weights was set; macro averages over foreground-bearing rows only and is structurally
#     blind to false positives, which is why the d013 finetune's best-*.ckpt pointed at its worst
#     epoch. Not used here (no --init-weights) but the flag is now explicit rather than implied.
#  7. LR left at 0.01. The probe was skipped deliberately -- U-Nets are LR-robust and 0.01 trained
#     the current 0.80-Dice model. What matters is time at low LR, which item 4 handles.
#  8. EMA left off. The callback runs a SECOND full pass over the 1500-patch manifest to log
#     val_dice_ema, and that cost was never measured. Do not enable it on a 7-day run untested.
#
# EXPECT val_dice TO SIT BELOW THE OLD RUN and train loss to be higher. The target no longer
# contains unclicked lesions, so the model is scored against a harder objective. Judge on the
# per-scenario metrics, not the aggregate:
#     val/subset_clicked/val_selectivity_margin   -0.2709 -> should go POSITIVE
#     val/subset_clicked/val_dice_vs_clicked_subset  0.4673 -> should go UP (the un-confounded one)
#     val/none_clicked/val_pred_fg                 0.0196 -> should go toward 0
#     val/all_clicked/val_dice                     0.8390 -> should hold
#
# KNOWN, MEASURED ON A PROBE: for roughly the first ~50 epochs from MAE the model ignores the click
# entirely (val_prompt_gap ~ 0). This happens with AND without the consistency term, so it is a
# training stage, not a defect -- a net has to learn what a lesion looks like before a click can
# help it. Do not kill the run over an early flat prompt gap.
#
# WHY ONE H200 INSTEAD OF TWO A100s
#   Single GPU means no DDP: no rank sharding, no cross-rank metric reduction, no unused-parameter
#   handling, no effective-batch doubling. Four classes of subtle failure simply do not exist.
#
# BATCH SIZE 12 (rows), with --prompts-per-patch 2 -> 6 DISTINCT patches per step.
#   batch_size counts ROWS, not patches. Two prompts share one patch and one augmentation pass, so
#   rows/2 is the number of distinct patches. Activations ~62 GiB of 141 -- ample.
#
# BUCKET xl, NOT l -- THIS IS THE LOAD-BEARING CHOICE ON THIS NODE.
#   On the A100 a step took 0.96 s and needed ~3.1 patches/s, which bucket l (8 workers) just met at
#   99% util. An H200 step should land near 0.6-0.7 s and need ~6 patches/s, so l would starve it.
#   xl is 16 train / 8 val workers, ~32-40 cores at peak, hence cpus-per-task=64.
#   If GPU util still sags, the bottleneck is the data path, not the GPU -- do not raise the batch.
#   The instance-target path adds cc3d on the crop: 5.7 ms mean against a 477 ms per-patch budget.
#
# WILL NOT FIT IN ONE 7-DAY JOB. ~1200 epochs at the measured rate is ~7.4 days. EXPECT ONE RESUME:
#   RESUME=<out>/checkpoints/last.ckpt sbatch slurm_supervised_999_h200.sh
# $OUT is NOT deleted when RESUME is set.
#
# STORAGE: dlc-slowpoke has a slow link to /nnunet_data. Staging ~543 GB is a ONE-TIME cost inside
# this job's wall time; it does not affect per-step throughput once training starts.

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
MAE_CKPT="/nnunet_data/NanoUNet_results/nanounet/Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0/mae_pretrain/checkpoints/last.ckpt"
ROI_CONFIG=configs/longrun.json   # relative paths resolve against the nanoUNet repo root
SUP_EPOCHS=1200
ITERS_PER_EPOCH=1000
BATCH_SIZE=12             # rows; must divide by PROMPTS_PER_PATCH. 12 rows / 2 prompts = 6 DISTINCT
                          # patches per step. Paired rows are highly correlated (same crop, same
                          # augmentation, same target; only the click differs).
PROMPTS_PER_PATCH=2       # two independent click draws per patch, sharing one crop + one augmentation
CONSISTENCY_WEIGHT=0.02   # measured, not guessed: train_loss_seg averages ~0.047 while the raw
                          # consistency term sits at 0.79-0.82, so 0.02 puts consistency at ~20-25%
                          # of total loss. A probe with this term REMOVED was strictly worse
                          # (val_dice 0.175 vs 0.307 at 50 epochs), so keep it.
WARMUP_EPOCHS=10
STRETCHED_K=376
STRETCHED_REF=500
VAL_EVERY_N=2
RESUME="${RESUME:-}"      # set to <out>/checkpoints/last.ckpt to continue a timed-out run
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

# valset_1500* is REQUIRED. Without it --val-manifest dies at startup, AFTER the 543 GB stage.
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
if [ ! -f "$VAL_MANIFEST" ] || [ ! -f "${LOCAL_PREP}/${DS_FOLDER}/valset_1500.targets.npz" ]; then
  echo "FATAL: val manifest not staged: $VAL_MANIFEST (+ .targets.npz)"
  echo "Fix: confirm --include \"valset_1500*\" above, and that both files exist under $REMOTE_PREP."
  echo "     Rebuild with: nanounet_build_valset -d 999 --plans $PLANS_NAME --config $ROI_CONFIG \\"
  echo "                     --out ${REMOTE_PREP}/valset_1500.json"
  exit 1
fi

# splits_final.json must be the SINGLE balanced 15% split, not the old dataset-blind 5-fold.
if ! python3 -c "
import json, sys
s = json.load(open('$LOCAL_PREP/$DS_FOLDER/splits_final.json'))
sys.exit(0 if len(s) == 1 else 1)"; then
  echo "FATAL: splits_final.json is not the single balanced split this run expects."
  echo "Fix: nanounet_build_splits -d 999 --plans $PLANS_NAME --val-frac 0.15 --force"
  exit 1
fi

# The empirical click model needs volume_vox; instance-conditional targets need bboxes_zyx to map
# each cc3d component in a crop back to its parent lesion. Fail now, not 400 steps in.
if ! python3 -c "
import glob, json, sys
f = sorted(glob.glob('$LOCAL_PREP/$DS_FOLDER/$DATA_ID/*_centroids.json'))[:20]
need = ('volume_vox', 'bboxes_zyx')
sys.exit(0 if f and all(all(k in json.load(open(x)) for k in need) for x in f) else 1)"; then
  echo "FATAL: centroid sidecars lack volume_vox and/or bboxes_zyx."
  echo "Fix: bash scripts/run_preprocess_sidecars.sh"
  exit 1
fi

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_h200_instance_1200ep"
if [ -z "$RESUME" ]; then
  rm -rf "$OUT"
  INIT_ARGS=(--mae-ckpt "$MAE_CKPT")
else
  echo "resuming from $RESUME"
  INIT_ARGS=(--resume "$RESUME")
fi

# A CSVLogger is attached automatically -> $OUT/metrics/version_*/metrics.csv carries all 99
# per-scenario metrics as a curve, independent of wandb.
if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config "$ROI_CONFIG" \
  --val-manifest "$VAL_MANIFEST" \
  --val-every-n-epochs "$VAL_EVERY_N" \
  "${INIT_ARGS[@]}" \
  --out "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$SUP_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --lr 0.01 \
  --warmup-epochs "$WARMUP_EPOCHS" \
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
  --wandb-name "Dataset999_f0_instance_targets_cohorts_1200ep"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
