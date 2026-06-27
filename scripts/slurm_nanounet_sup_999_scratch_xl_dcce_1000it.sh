#!/bin/bash
# Fresh supervised Dataset999 run from MAE checkpoint.
# Current fast recipe: DC+CE, bucket xl, 500 epochs x 1000 iters.

#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=248G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-sup-999-xl-dcce
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_xl_dcce_1000it.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_sup_999_xl_dcce_1000it.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
SUP_EPOCHS=500
ITERS_PER_EPOCH=1000
VAL_ITERS=50
STORAGE=/nnunet_data

export PIP_CACHE_DIR=/root/.pip-cache
export NANOUNET_RAW="${STORAGE}/NanoUNet_raw"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$PIP_CACHE_DIR" "$NANOUNET_RESULTS" "$NANOUNET_TMPDIR"

if ! nanounet_train --help &>/dev/null; then
  echo "FATAL: nanounet_train not found or broken."
  exit 1
fi

LOCAL_PREP=/root/NanoUNet_preprocessed
mkdir -p "$LOCAL_PREP"
if ! rclone copy "${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress \
  --transfers 32 \
  --multi-thread-streams 16 \
  --no-update-modtime \
  --retries 5 \
  --copy-links; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"

OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}"
MAE_CKP="${OUT}/mae_pretrain/checkpoints/last.ckpt"

if [[ ! -f "$MAE_CKP" ]]; then
  echo "FATAL: MAE checkpoint missing: $MAE_CKP"
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

# Fresh supervised run, keeping mae_pretrain/.
rm -rf "${OUT}/checkpoints"
rm -f "${OUT}/mem_diag.jsonl"

echo "MAE ckpt: $MAE_CKP"
echo "batch_size in plans: $(python3 -c "import json; print(json.load(open('${LOCAL_PREP}/${DS_FOLDER}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['batch_size'])")"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --mae-ckpt "$MAE_CKP" \
  --epochs "$SUP_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --lr 1e-3 \
  --lr-schedule stretched_tail_poly \
  --stretched-k 188 \
  --stretched-ref 250 \
  --loss dc_ce \
  --dl-bucket xl \
  --dl-persistent-workers \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_sup_scratch_bs4_xl_dcce_1000it"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
