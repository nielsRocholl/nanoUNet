#!/bin/bash
#SBATCH --qos=vram
#SBATCH --nodelist=dlc-slowpoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-900-foundation
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_900_foundation_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_900_foundation_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-name=nanounet-900-foundation
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

# Dataset900, nnFoundationCNN encoder + z-only 1 mm plans (nnFoundationCNN_z1p0): supervised SUP_EPOCHS
# (instance targets, site-balanced) -> mixed d013 FT 80ep. One H200. No MAE stage. Loss dc_ce.
#
# Recipe (from the plans' pretrain_info, nanounet_train defaults): foundation encoder, SGD, lr 1e-3,
# 2 warmup epochs, poly, deep supervision off. Not re-passed below; the config table prints them.
#
# WALL BUDGET (D10, ~4 d supervised). Measured on one H200, batch 8, 192^3 patch, prompts-per-patch 2, 24 CPUs:
#   1.15 s/step => ~19 min per 1000-iter epoch; a 2000-patch validation pass takes ~6 min and runs every 2nd epoch
#   => ~22.3 min/epoch on average. GPU util median 93 %, peak 94 GB of 143 GB (batch 12 peaked at 133 GB).
#   Supervised 250 ep ~ 93 h (3.9 d, 2.0 M patches); d013 FT 80 ep ~ 30 h; rclone staging ~0.7 h => ~5.2 d of the 7 d qos.
#   Both epoch counts are even on purpose: checkpoints are written at validation, and validation runs every 2nd epoch.
#   To shorten: lower SUP_EPOCHS / FT_EPOCHS (keep them even).
# Loader: --dl-bucket l (8 train / 4 val workers). xl (16/8) was killed by an 80 GB cgroup; it was not tested at 200 GB.
# CODE: the container image must carry feat/nnfoundation-zonly (>= CODE_SHA). The job checks this below and refuses to
# start on the old image. Use a NEW --container-name per image rebuild: a named container keeps its overlay on the node.
# Resume is a state machine on NFS, not RESUME=last.ckpt. FRESH=1 wipes $OUT only.
# Never deletes $OUT_FT; refuses to overwrite it. SKIP_SUP=1 stops supervised where last.ckpt is and goes to FT.

set -euo pipefail

FOLD=0
DATASET_ID=900
DS_FOLDER=Dataset900_Merged
PLANS_NAME=nnFoundationCNN_z1p0
ROI_CONFIG=nanounet/configs/longrun900.json
FT_CONFIG=nanounet/configs/finetune900_d013.json
SUP_EPOCHS=250
FT_EPOCHS=80
ITERS_PER_EPOCH=1000
BATCH_SIZE=8
PROMPTS_PER_PATCH=2
CONSISTENCY_WEIGHT=0.02
EMA_DECAY=0.999
VAL_EVERY_N=2
STORAGE=/nnunet_data
RESULTS_ROOT="${STORAGE}/NanoUNet_results"
REMOTE_ROOT="${STORAGE}/NanoUNet_preprocessed"
LOCAL_ROOT=/root/NanoUNet_preprocessed
# Rehearsal harness (tests only, never set in a real job): runs this whole script on a small copy of the data.
if [ -n "${REHEARSAL_DIR:-}" ]; then
  RESULTS_ROOT="$REHEARSAL_DIR/results"; REMOTE_ROOT="$REHEARSAL_DIR/remote"; LOCAL_ROOT="$REHEARSAL_DIR/local"
  SUP_EPOCHS="${REH_SUP_EPOCHS:-2}"; FT_EPOCHS="${REH_FT_EPOCHS:-1}"; ITERS_PER_EPOCH="${REH_ITERS:-30}"
fi
CODE_SHA="${CODE_SHA:-}"   # optional: commit the image was built from; checked against $NANOUNET_GIT_SHA when both are set
FRESH="${FRESH:-0}"
SKIP_SUP="${SKIP_SUP:-0}"

OUT="${RESULTS_ROOT}/nanounet/${DS_FOLDER}_${PLANS_NAME}_f${FOLD}_foundation"
OUT_FT="${OUT}_ft"
SUP_LAST="$OUT/checkpoints/last.ckpt"
FT_LAST="$OUT_FT/finetune/last.ckpt"

export PIP_CACHE_DIR=/root/.pip-cache
export NANOUNET_RAW="${STORAGE}/NanoUNet_raw"
export NANOUNET_RESULTS="$RESULTS_ROOT"
export NANOUNET_PRETRAINED=/root/NanoUNet_pretrained
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$PIP_CACHE_DIR" "$NANOUNET_RESULTS" "$NANOUNET_TMPDIR" "$NANOUNET_PRETRAINED"

if ! python3 -c "from nanounet.model.foundation import verify_foundation; from nanounet.cli.train_foundation import resolve_foundation" &>/dev/null; then
  echo "FATAL: the image's nanounet is the old code (no nnFoundationCNN support)."
  echo "Fix: rebuild the image from feat/nnfoundation-zonly and use a new --container-name"
  exit 1
fi
if [ -n "$CODE_SHA" ] && [ -n "${NANOUNET_GIT_SHA:-}" ] && [ "$CODE_SHA" != "$NANOUNET_GIT_SHA" ]; then
  echo "FATAL: image code is $NANOUNET_GIT_SHA, this script expects $CODE_SHA"
  exit 1
fi
if ! nanounet_train --help &>/dev/null; then
  echo "FATAL: nanounet_train not found or broken."
  exit 1
fi
echo "image code: ${NANOUNET_GIT_SHA:-unknown}"

if [ "$SKIP_SUP" = 1 ] && [ "$FRESH" = 1 ]; then
  echo "FATAL: SKIP_SUP=1 and FRESH=1 are contradictory (FRESH wipes the supervised checkpoints FT needs)"
  exit 1
fi
if [ "$SKIP_SUP" = 1 ] && [ ! -f "$FT_LAST" ] && [ ! -f "$SUP_LAST" ]; then
  echo "FATAL: SKIP_SUP=1 but no supervised checkpoint to finetune from: $SUP_LAST"
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

LOCAL_PREP="$LOCAL_ROOT"
VALSET=valset_2000_${PLANS_NAME}
REMOTE_PREP="${REMOTE_ROOT}/${DS_FOLDER}"
mkdir -p "$LOCAL_PREP/${DS_FOLDER}"

for f in "${REMOTE_PREP}/${PLANS_NAME}.json" "${REMOTE_PREP}/splits_final.json" \
         "${REMOTE_PREP}/cohorts.json" "${REMOTE_PREP}/${VALSET}.json" \
         "${REMOTE_PREP}/${VALSET}.targets.npz"; do
  [ -f "$f" ] || { echo "FATAL: required remote file missing (fail before rclone): $f"; exit 1; }
done

DATA_ID=$(python3 -c "import json; print(json.load(open('${REMOTE_PREP}/${PLANS_NAME}.json'))['configurations']['3d_fullres']['data_identifier'])")
echo "data_identifier: $DATA_ID   staging Dataset900 (~740 GB) over a slow link"

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
  --include "${VALSET}*" \
  --include "dataset_fingerprint.json" \
  --include "${DATA_ID}/**"; then
  exit 1
fi

export NANOUNET_PREPROCESSED="$LOCAL_PREP"

VAL_MANIFEST="${LOCAL_PREP}/${DS_FOLDER}/${VALSET}.json"
if [ ! -f "$VAL_MANIFEST" ] || [ ! -f "${LOCAL_PREP}/${DS_FOLDER}/${VALSET}.targets.npz" ]; then
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


# nnFoundationCNN checkpoint: plans' pretrain_info records the path and sha256 from the preprocess machine;
# stage it to local disk, check the sha256, and rewrite the path in the LOCAL plans copy only.
python3 - <<PY
import hashlib, json, shutil, sys
p = "$LOCAL_PREP/$DS_FOLDER/$PLANS_NAME.json"
plans = json.load(open(p))
info = plans["pretrain_info"]
dst = "$NANOUNET_PRETRAINED/checkpoint_final.pth"
shutil.copyfile(info["checkpoint_path"], dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
if h != info["sha256"]:
    sys.exit("FATAL: nnFoundationCNN sha256 %s != plans %s" % (h, info["sha256"]))
info["checkpoint_path"] = dst
json.dump(plans, open(p, "w"), indent=2)
print("nnFoundationCNN checkpoint staged, sha256 ok:", h[:12])
PY

# Re-derived on every retry attempt, not just once -- a crash mid-run leaves a fresher last.ckpt.
compute_main_args() {
  SKIP_MAIN=0
  MAIN_ARGS=()
  if [ -f "$FT_LAST" ]; then
    echo "FT checkpoint present: skip supervised, resume FT from $FT_LAST"
    SKIP_MAIN=1
  elif [ "$SKIP_SUP" = 1 ]; then
    echo "SKIP_SUP=1: stop supervised at $SUP_LAST, go to FT"
    SKIP_MAIN=1
  elif [ -f "$SUP_LAST" ]; then
    echo "supervised resume from $SUP_LAST"
    MAIN_ARGS=(--resume "$SUP_LAST")
  else
    echo "fresh supervised (nnFoundationCNN encoder) into $OUT"
  fi
}

# A hang (stuck dataloader/checkpoint write, no exception) never exits, so SLURM and the retry
# loop above never see it -- the job just idles out the wall clock. Runs the wrapped command in
# its own process group (setsid) so we can kill the whole tree, not just the parent; watchdog
# kills the group if nothing under watch_dir has been modified in stale_min minutes.
watchdog() {
  local watch_dir="$1" pgid="$2" stale_min="${3:-45}"
  while kill -0 "$pgid" 2>/dev/null; do
    sleep 300
    latest=$(find "$watch_dir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    [ -z "$latest" ] && continue
    age_min=$(( ($(date +%s) - ${latest%.*}) / 60 ))
    if [ "$age_min" -ge "$stale_min" ]; then
      echo "WATCHDOG: nothing under $watch_dir modified in ${age_min}m (>= ${stale_min}m); killing pgid $pgid"
      kill -TERM -- "-$pgid" 2>/dev/null || true
      sleep 15
      kill -KILL -- "-$pgid" 2>/dev/null || true
      return
    fi
  done
}
WATCHDOG_STALE_MIN="${WATCHDOG_STALE_MIN:-45}"

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
    echo "=== nanounet_train (supervised) attempt $attempt/$MAIN_MAX_RETRIES ==="
    setsid nanounet_train \
      -d "$DATASET_ID" \
      -f "$FOLD" \
      --plans "$PLANS_NAME" \
      --config "$ROI_CONFIG" \
      --val-manifest "$VAL_MANIFEST" \
      --val-every-n-epochs "$VAL_EVERY_N" \
      "${MAIN_ARGS[@]}" \
      --out "$OUT" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$SUP_EPOCHS" \
      --iters-per-epoch "$ITERS_PER_EPOCH" \
      --ema-decay "$EMA_DECAY" \
      --monitor val_dice \
      --loss dc_ce \
      --prompts-per-patch "$PROMPTS_PER_PATCH" \
      --consistency-weight "$CONSISTENCY_WEIGHT" \
      --dl-bucket l \
      --accelerator cuda \
      --precision 16-mixed \
      --wandb-name "Dataset900_f0_foundation_z1p0_sup" &
    train_pid=$!
    watchdog "$OUT" "$train_pid" "$WATCHDOG_STALE_MIN" &
    watchdog_pid=$!
    train_rc=0
    wait "$train_pid" || train_rc=$?
    kill "$watchdog_pid" 2>/dev/null || true; wait "$watchdog_pid" 2>/dev/null || true
    if [ "$train_rc" -eq 0 ]; then
      break
    fi
    if [ "$attempt" -ge "$MAIN_MAX_RETRIES" ]; then
      echo "FATAL: nanounet_train (supervised) failed $MAIN_MAX_RETRIES times in this allocation; giving up"
      exit 1
    fi
    echo "nanounet_train (supervised) attempt $attempt failed; retrying in 30s from the latest checkpoint"
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
    m = re.search(re.escape(key) + r'=([0-9]+(?:\.[0-9]+)?)', os.path.basename(path))
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
  setsid nanounet_train \
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
    --dl-bucket l \
    --accelerator cuda \
    --precision 16-mixed \
    --wandb-name "Dataset900_f0_foundation_d013_ft_80ep" &
  local train_pid=$!
  watchdog "$OUT_FT" "$train_pid" "$WATCHDOG_STALE_MIN" &
  local watchdog_pid=$!
  local train_rc=0
  wait "$train_pid" || train_rc=$?
  kill "$watchdog_pid" 2>/dev/null || true; wait "$watchdog_pid" 2>/dev/null || true
  return "$train_rc"
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
elif [ -f "$SUP_LAST" ] && { [ "$SKIP_SUP" = 1 ] || sup_done; }; then
  INIT_CKPT=$(pick_init_ckpt)
  echo "supervised done (SKIP_SUP=$SKIP_SUP); FT init $INIT_CKPT"
  mkdir "$OUT_FT" || {
    echo "FATAL: output already exists; refusing to overwrite: $OUT_FT"
    exit 1
  }
  run_ft_with_retry "$INIT_CKPT" || exit 1
else
  echo "supervised not finished (no last.ckpt or epoch < SUP_EPOCHS-1); skip FT this allocation"
fi
