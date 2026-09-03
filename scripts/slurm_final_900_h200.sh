#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-900-final
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_900_final.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_900_final.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-name=nanounet-900-final
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# Dataset900: MAE 250ep → supervised 1200ep (instance targets, site-balanced) → mixed d013 FT 80ep.
# One H200. No --longi. No Dataset999 MAE. No --only-prefix. Loss dc_ce.
#
# WALL: 578 s/ep × 1200 = 8.03 d supervised alone (wandb ekkxcgi6 runtime 8.0 d). qos=vram is 7 d,
# so EXPECT AT LEAST ONE RESUME. Copy ~1 d + MAE ~1 d + FT ~0.5 d ⇒ 10–11 d total.
# Staging is /root/NanoUNet_preprocessed. --container-name keeps that overlay on the node after
# the job ends, so a resubmit on dlc-slowpoke reuses it (rclone copy then only fills gaps).
# Do not bind-mount /scratch: it is not a host path, and the image has no /scratch dest.
#
# Resume is a state machine on NFS, not RESUME=last.ckpt. FRESH=1 wipes $OUT only.
# Never deletes $OUT_FT; refuses to overwrite it.

set -euo pipefail

FOLD=0
DATASET_ID=900
DS_FOLDER=Dataset900_Merged
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
ROI_CONFIG=configs/longrun900.json
FT_CONFIG=configs/finetune900_d013.json
MAE_EPOCHS=250
SUP_EPOCHS=1200
FT_EPOCHS=80
ITERS_PER_EPOCH=1000
BATCH_SIZE=12
PROMPTS_PER_PATCH=2
CONSISTENCY_WEIGHT=0.02
WARMUP_EPOCHS=10
EMA_DECAY=0.999
STRETCHED_K=376
STRETCHED_REF=500
VAL_EVERY_N=2
STORAGE=/nnunet_data
FRESH="${FRESH:-0}"

OUT="${STORAGE}/NanoUNet_results/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_h200_final"
OUT_FT="${OUT}_ft"
SUP_LAST="$OUT/checkpoints/last.ckpt"
MAE_LAST="$OUT/mae_pretrain/checkpoints/last.ckpt"
FT_LAST="$OUT_FT/finetune/last.ckpt"

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

if [ "$FRESH" = 1 ]; then
  if [ -e "$OUT_FT" ]; then
    echo "FATAL: FRESH=1 but \$OUT_FT exists; move it aside first: $OUT_FT"
    exit 1
  fi
  echo "FRESH=1: wiping $OUT"
  rm -rf "$OUT"
fi

LOCAL_PREP=/root/NanoUNet_preprocessed
REMOTE_PREP="${STORAGE}/NanoUNet_preprocessed/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"

for f in "${REMOTE_PREP}/${PLANS_NAME}.json" "${REMOTE_PREP}/splits_final.json" \
         "${REMOTE_PREP}/cohorts.json" "${REMOTE_PREP}/valset_2000.json" \
         "${REMOTE_PREP}/valset_2000.targets.npz"; do
  [ -f "$f" ] || { echo "FATAL: required remote file missing (fail before rclone): $f"; exit 1; }
done

DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID   staging Dataset900 (~332 GB) over a slow link"

shopt -s nullglob
weights=( "${REMOTE_PREP}/${DATA_ID}"/d013_*_weights.json )
shopt -u nullglob
if [ ${#weights[@]} -eq 0 ]; then
  echo "FATAL: no d013_*_weights.json under ${REMOTE_PREP}/${DATA_ID}"
  echo "Fix: nanounet_lesion_weights -d 900 --plans $PLANS_NAME --meta-dir ${STORAGE}/Longitudinal-CT/meta"
  exit 1
fi

if ! rclone copy "$REMOTE_PREP/" "$LOCAL_PREP/${DS_FOLDER}" \
  --progress --transfers 32 --multi-thread-streams 16 --no-update-modtime --retries 5 --copy-links \
  --include "${PLANS_NAME}.json" \
  --include "splits_final.json" \
  --include "cohorts.json" \
  --include "valset_2000*" \
  --include "${DATA_ID}/**"; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"

VAL_MANIFEST="${LOCAL_PREP}/${DS_FOLDER}/valset_2000.json"
if [ ! -f "$VAL_MANIFEST" ] || [ ! -f "${LOCAL_PREP}/${DS_FOLDER}/valset_2000.targets.npz" ]; then
  echo "FATAL: val manifest not staged: $VAL_MANIFEST (+ .targets.npz)"
  exit 1
fi

if ! python3 -c "
import json, sys
s = json.load(open('$LOCAL_PREP/$DS_FOLDER/splits_final.json'))
sys.exit(0 if len(s) == 1 else 1)"; then
  echo "FATAL: splits_final.json is not the single balanced split this run expects."
  exit 1
fi

if ! python3 -c "
import glob, json, sys
f = sorted(glob.glob('$LOCAL_PREP/$DS_FOLDER/$DATA_ID/*_centroids.json'))[:20]
need = ('volume_vox', 'bboxes_zyx')
sys.exit(0 if f and all(all(k in json.load(open(x)) for k in need) for x in f) else 1)"; then
  echo "FATAL: centroid sidecars lack volume_vox and/or bboxes_zyx."
  echo "Fix: nanounet_preprocess -d 900 --sidecars-only"
  exit 1
fi

shopt -s nullglob
local_w=( "$LOCAL_PREP/$DS_FOLDER/$DATA_ID"/d013_*_weights.json )
shopt -u nullglob
if [ ${#local_w[@]} -eq 0 ]; then
  echo "FATAL: d013 weights not staged under $LOCAL_PREP/$DS_FOLDER/$DATA_ID"
  exit 1
fi

# Re-derived on every retry attempt below, not just once -- a crash mid-run leaves a fresher
# last.ckpt than we started with, and the next attempt must resume from it, not repeat from
# scratch.
compute_main_args() {
  SKIP_MAIN=0
  MAIN_ARGS=()
  MAE_FLAGS=()
  if [ -f "$FT_LAST" ]; then
    echo "FT checkpoint present: skip SSL+supervised, resume FT from $FT_LAST"
    SKIP_MAIN=1
  elif [ -f "$SUP_LAST" ]; then
    echo "supervised resume from $SUP_LAST (no --mae-pretrain)"
    MAIN_ARGS=(--resume "$SUP_LAST")
  elif [ -f "$MAE_LAST" ]; then
    echo "MAE resume from $MAE_LAST"
    MAIN_ARGS=(--mae-pretrain --mae-resume "$MAE_LAST")
    MAE_FLAGS=(--mae-epochs "$MAE_EPOCHS" --mae-lr 1e-2 --mae-lr-schedule cosine_warm_restarts --mae-cosine-t0 250 --mae-mask-ratio 0.75)
  else
    echo "fresh MAE+supervised into $OUT"
    MAIN_ARGS=(--mae-pretrain)
    MAE_FLAGS=(--mae-epochs "$MAE_EPOCHS" --mae-lr 1e-2 --mae-lr-schedule cosine_warm_restarts --mae-cosine-t0 250 --mae-mask-ratio 0.75)
  fi

  if [ "${MAIN_ARGS[0]:-}" = "--resume" ] && [ ! -f "$SUP_LAST" ]; then
    echo "FATAL: claimed supervised resume missing: $SUP_LAST"
    exit 1
  fi
  if [ "${MAIN_ARGS[1]:-}" = "--mae-resume" ] && [ ! -f "$MAE_LAST" ]; then
    echo "FATAL: claimed MAE resume missing: $MAE_LAST"
    exit 1
  fi
}

compute_main_args

if [ "$SKIP_MAIN" = 0 ]; then
  mkdir -p "$OUT"
  if [ -f "$OUT/wandb_run_id.txt" ]; then
    export WANDB_RUN_ID
    WANDB_RUN_ID=$(tr -d '[:space:]' < "$OUT/wandb_run_id.txt")
  else
    WANDB_RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
    export WANDB_RUN_ID
    echo "$WANDB_RUN_ID" > "$OUT/wandb_run_id.txt"
  fi
  export WANDB_RESUME=allow
  echo "wandb run $WANDB_RUN_ID"

  # A crash here must NOT kill the job: retry from the latest last.ckpt. Named container
  # overlay also survives resubmission on dlc-slowpoke.
  MAIN_MAX_RETRIES="${MAIN_MAX_RETRIES:-8}"
  attempt=1
  while :; do
    compute_main_args
    if [ "$SKIP_MAIN" = 1 ]; then
      break
    fi
    echo "=== nanounet_train (SSL+supervised) attempt $attempt/$MAIN_MAX_RETRIES ==="
    if nanounet_train \
      -d "$DATASET_ID" \
      -f "$FOLD" \
      --plans "$PLANS_NAME" \
      --config "$ROI_CONFIG" \
      --val-manifest "$VAL_MANIFEST" \
      --val-every-n-epochs "$VAL_EVERY_N" \
      "${MAIN_ARGS[@]}" \
      "${MAE_FLAGS[@]}" \
      --out "$OUT" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$SUP_EPOCHS" \
      --iters-per-epoch "$ITERS_PER_EPOCH" \
      --lr 0.01 \
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
      --devices 1 \
      --accelerator cuda \
      --precision 16-mixed \
      --wandb-name "Dataset900_f0_ssl_sup_instance_1200ep"; then
      break
    fi
    if [ "$attempt" -ge "$MAIN_MAX_RETRIES" ]; then
      echo "FATAL: nanounet_train (SSL+supervised) failed $MAIN_MAX_RETRIES times in this allocation; giving up"
      exit 1
    fi
    echo "nanounet_train (SSL+supervised) attempt $attempt failed; retrying in 30s from the latest checkpoint"
    attempt=$((attempt + 1))
    sleep 30
  done
fi

sup_done() {
  python3 -c "
from nanounet.lightning_ckpt import pl_ckpt_epoch_and_target
ep, tgt = pl_ckpt_epoch_and_target('$SUP_LAST')
# PL 2.x last.ckpt after N epochs stores current_epoch = N-1. Treat both as done.
raise SystemExit(0 if ep >= tgt - 1 else 1)
"
}

pick_init_ckpt() {
  python3 -c "
import csv, glob, os, re, sys
out = sys.argv[1]
ck = os.path.join(out, 'checkpoints')
def _metric(path, key):
    m = re.search(re.escape(key) + r'=([0-9.]+)', os.path.basename(path))
    return float(m.group(1)) if m else -1.0
bestsel = sorted(glob.glob(os.path.join(ck, 'bestsel-*.ckpt')), key=lambda p: _metric(p, 'val_prompt_score'))
best = sorted(
    (p for p in glob.glob(os.path.join(ck, 'best-*.ckpt'))
     if not os.path.basename(p).startswith('bestsel-')),
    key=lambda p: _metric(p, 'val_dice'),
)
if not best:
    sys.exit('FATAL: no best-*.ckpt under ' + ck)
if not bestsel:
    print(best[-1]); raise SystemExit(0)
sel = bestsel[-1]
m = re.search(r'epoch=(\d+)', os.path.basename(sel))
ep = int(m.group(1)) if m else None
val_dice = None
if ep is not None:
    for csvp in glob.glob(os.path.join(out, 'metrics', 'version_*', 'metrics.csv')):
        with open(csvp, newline='') as f:
            for row in csv.DictReader(f):
                if not row.get('epoch') or not row.get('val_dice'):
                    continue
                if int(float(row['epoch'])) == ep:
                    val_dice = float(row['val_dice'])
if val_dice is not None and val_dice < 0.60:
    print(best[-1], file=sys.stderr)
    print('bestsel val_dice=%.4f < 0.60; falling back to best-*.ckpt' % val_dice, file=sys.stderr)
    print(best[-1])
else:
    print(sel)
" "$OUT"
}

run_ft() {
  local ft_args=("$@")
  unset WANDB_RUN_ID WANDB_RUN_PATH
  export WANDB_RESUME=never
  nanounet_train \
    -d "$DATASET_ID" \
    -f "$FOLD" \
    --plans "$PLANS_NAME" \
    --config "$FT_CONFIG" \
    --val-manifest "$VAL_MANIFEST" \
    --val-every-n-epochs "$VAL_EVERY_N" \
    "${ft_args[@]}" \
    --out "$OUT_FT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$FT_EPOCHS" \
    --iters-per-epoch "$ITERS_PER_EPOCH" \
    --optimizer adamw --lr 1e-5 --wd 3e-5 --grad-clip 1.0 \
    --warmup-epochs 2 \
    --lr-schedule poly \
    --loss dc_ce \
    --prompts-per-patch "$PROMPTS_PER_PATCH" \
    --consistency-weight "$CONSISTENCY_WEIGHT" \
    --consistency-warmup-epochs 0 \
    --ema-decay "$EMA_DECAY" \
    --monitor val_dice \
    --dl-bucket xl \
    --devices 1 \
    --accelerator cuda \
    --precision 16-mixed \
    --wandb-name "Dataset900_f0_mixed_d013_ft_80ep"
}

# Same in-job retry rationale as the main call above: FT re-derives --resume vs --init-weights
# from $FT_LAST on every attempt, so a crash after FT has already checkpointed resumes instead of
# restarting FT from the supervised init.
run_ft_with_retry() {
  local init_ckpt="$1"
  local ft_max_retries="${FT_MAX_RETRIES:-8}"
  local attempt=1
  while :; do
    echo "=== FT attempt $attempt/$ft_max_retries ==="
    if [ -f "$FT_LAST" ]; then
      echo "FT resume from $FT_LAST"
      run_ft --resume "$FT_LAST" && return 0
    else
      echo "FT init from $init_ckpt"
      run_ft --init-weights "$init_ckpt" && return 0
    fi
    if [ "$attempt" -ge "$ft_max_retries" ]; then
      echo "FATAL: FT failed $ft_max_retries times in this allocation; giving up"
      return 1
    fi
    echo "FT attempt $attempt failed; retrying in 30s from the latest checkpoint"
    attempt=$((attempt + 1))
    sleep 30
  done
}

if [ -f "$FT_LAST" ]; then
  echo "resuming FT from $FT_LAST"
  run_ft_with_retry "" || exit 1
elif [ -f "$SUP_LAST" ] && sup_done; then
  INIT_CKPT=$(pick_init_ckpt)
  echo "supervised done; FT init $INIT_CKPT"
  mkdir "$OUT_FT" || {
    echo "FATAL: output already exists; refusing to overwrite: $OUT_FT"
    exit 1
  }
  run_ft_with_retry "$INIT_CKPT" || exit 1
else
  echo "supervised not finished (no last.ckpt or epoch < 1200-1); skip FT this allocation"
fi
