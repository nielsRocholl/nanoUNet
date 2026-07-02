#!/bin/bash
# d013 two-stream longi finetune: stage d013 preprocessed cases + baseline sidecars, AdamW 1e-5.

#SBATCH --qos=vram
#SBATCH --nodelist=dlc-arceus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=248G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-longi-d013
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d013_longi.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_finetune_d013_longi.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

FOLD=0
DATASET_ID=999
DS_FOLDER=Dataset999_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
META_DIR=/nnunet_data/unprocessed-universal-lesion-segmentation/meta_lesion_types
FINETUNE_EPOCHS=500
ITERS_PER_EPOCH=1000
VAL_ITERS=50
STORAGE=/nnunet_data
ONLY_PREFIX=d013_

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

# Step 0: expect 2-channel longi dataset preprocessed on shared storage with _bl_clicks.json sidecars.
# Build: register_longi → longi_build → (patch plans ch0→ch1) → nanounet_preprocess → longi_clicks

LOCAL_PREP=/root/NanoUNet_preprocessed
CASE_ID=$(python3 -c "import json; print(json.load(open('${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
REMOTE_CASE="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}/${CASE_ID}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}/${CASE_ID}"

shopt -s nullglob
for f in "${REMOTE_CASE}/${ONLY_PREFIX}"*; do
  base=$(basename "$f")
  case "$base" in
    *.b2nd|*.pkl|*_centroids.json|*_weights.json|*_bl_clicks.json)
      rclone copyto "$f" "$LOCAL_PREP/${DS_FOLDER}/${CASE_ID}/$base" \
        --retries 5 --copy-links
      ;;
  esac
done
shopt -u nullglob

shopt -s nullglob
missing=0
for b2 in "$LOCAL_PREP/${DS_FOLDER}/${CASE_ID}/${ONLY_PREFIX}"*.b2nd; do
  [[ "$b2" == *_seg.b2nd ]] && continue
  id=$(basename "$b2" .b2nd)
  if [[ ! -f "$LOCAL_PREP/${DS_FOLDER}/${CASE_ID}/${id}_bl_clicks.json" ]]; then
    echo "FATAL: missing ${id}_bl_clicks.json (run longi_clicks after preprocess)"
    missing=1
  fi
  if [[ ! -f "$LOCAL_PREP/${DS_FOLDER}/${CASE_ID}/${id}_weights.json" ]]; then
    echo "FATAL: missing ${id}_weights.json"
    missing=1
  fi
done
shopt -u nullglob
[[ $missing -eq 0 ]] || exit 1

export NANOUNET_PREPROCESSED="$LOCAL_PREP"

INIT_CKPT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}/checkpoints/last.ckpt"
OUT="${NANOUNET_RESULTS}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_finetune_d013_longi"
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "FATAL: stage-2 supervised checkpoint missing: $INIT_CKPT"
  exit 1
fi

echo "Init weights: $INIT_CKPT"
echo "Out:          $OUT"

if ! nanounet_train \
  -d "$DATASET_ID" \
  -f "$FOLD" \
  --plans "$PLANS_NAME" \
  --config configs/finetune_d013.json \
  --init-weights "$INIT_CKPT" \
  --only-prefix "$ONLY_PREFIX" \
  --longi \
  --optimizer adamw \
  --lr 1e-5 \
  --wd 3e-5 \
  --grad-clip 1.0 \
  --lr-schedule poly \
  --epochs "$FINETUNE_EPOCHS" \
  --iters-per-epoch "$ITERS_PER_EPOCH" \
  --val-iters "$VAL_ITERS" \
  --loss dc_ce \
  --dl-bucket l \
  --dl-persistent-workers \
  --out "$OUT" \
  --accelerator cuda \
  --precision 16-mixed \
  --wandb-name "Dataset999_f0_longi_d013_adamw"; then
  rm -rf "$LOCAL_PREP/${DS_FOLDER}"
  exit 1
fi

rm -rf "$LOCAL_PREP/${DS_FOLDER}"
