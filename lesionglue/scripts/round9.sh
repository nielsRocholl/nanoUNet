#!/usr/bin/env bash
# Round 9 full sweep: all CV ablations -> rank -> (RUN_FINAL=1) retrain best + tau sweep + test gate.
# One command: scripts/round9.sh   (resumable: folds/configs with a cv_summary.json are skipped)
# Test split is touched ONLY in the final stage, gated behind RUN_FINAL=1.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=python3
RUNS=${RUNS:-runs/round9}
PROJ=${WANDB_PROJECT:-lesion-tracking}
mkdir -p "$RUNS"

# config name -> extra train flags. CV runs inherit rolled-back R9 defaults (hard_pair_w=0, desc_norm=off,
# new dustbin+bilinear matcher) unless a flag overrides. r8_baseline doubles as the Phase-1 bundle-ON arm.
declare -A CONFIGS=(
  [r8_baseline]="--hard-pair-w 0.2 --desc-norm"
  [r9_base]=""
  [r9_nce_0.1]="--nce-w 0.1"
  [r9_nce_0.2]="--nce-w 0.2"
  [r9_nce_0.5]="--nce-w 0.5"
)
ORDER=(r8_baseline r9_base r9_nce_0.1 r9_nce_0.2 r9_nce_0.5)

cv () {
  local name=$1 extra=${CONFIGS[$1]} out="$RUNS/$1"
  if [ -f "$out/cv_summary.json" ]; then echo "skip $name (cv_summary.json exists)"; return; fi
  echo "=== CV $name : ${extra:-<defaults>} ==="
  if [ -n "$extra" ]; then
    $PY tracking/cli/cv.py --out "$out" --n-folds 5 --cv-seed 0 \
      --wandb --wandb-project "$PROJ" --wandb-run-name "$name" --extra-train-args "$extra"
  else
    $PY tracking/cli/cv.py --out "$out" --n-folds 5 --cv-seed 0 \
      --wandb --wandb-project "$PROJ" --wandb-run-name "$name"
  fi
}

for n in "${ORDER[@]}"; do cv "$n"; done

# rank configs by mean val_match_score_ema; write winner name to best_config.txt
$PY - "$RUNS" <<'EOF'
import sys, json, glob, os
root = sys.argv[1]
rows = []
for p in sorted(glob.glob(f"{root}/*/cv_summary.json")):
    s = json.load(open(p))
    m = s.get("val_match_score_ema", {})
    sub = lambda k: s.get(k, {}).get("mean", float("nan"))
    rows.append((m.get("mean", 0.0), m.get("std", 0.0),
                 sub("val_acc_unchanged_split"), sub("val_acc_disappeared"), sub("val_acc_newly_appearing"),
                 os.path.basename(os.path.dirname(p))))
rows.sort(reverse=True)
print("\n# Round 9 CV ranking  (ema mean +/- std | unchanged | disap | newly)")
for mu, sd, uc, di, ne, name in rows:
    print(f"{mu:.4f} +/- {sd:.4f} | {uc:.3f} | {di:.3f} | {ne:.3f}  {name}")
if rows:
    best = rows[0][-1]
    open(f"{root}/best_config.txt", "w").write(best)
    print(f"\nbest -> {best}  (written to {root}/best_config.txt)")
EOF

if [ "${RUN_FINAL:-0}" != "1" ]; then
  echo; echo "CV done. Inspect ranking above, then run with RUN_FINAL=1 to retrain best + tau sweep + TEST gate."
  exit 0
fi

# ---- FINAL: retrain winner on full train+val, sweep dust_tau on val, eval test once ----
BEST=$(cat "$RUNS/best_config.txt")
FLAGS=${CONFIGS[$BEST]}
FOUT="$RUNS/final_$BEST"
echo "=== FINAL retrain $BEST : ${FLAGS:-<defaults>} ==="
if [ ! -f "$FOUT/best.ckpt" ]; then
  if [ -n "$FLAGS" ]; then
    $PY tracking/cli/train.py --out "$FOUT" $FLAGS --wandb --wandb-project "$PROJ" --wandb-run-name "final_$BEST"
  else
    $PY tracking/cli/train.py --out "$FOUT" --wandb --wandb-project "$PROJ" --wandb-run-name "final_$BEST"
  fi
fi
CKPT="$FOUT/best.ckpt"

echo "=== dust_tau sweep on val (post-hoc, no retrain) ==="
best_tau=0.20; best_score=-1
for tau in 0.10 0.15 0.18 0.20 0.22 0.25 0.30 0.35; do
  score=$($PY tracking/cli/eval.py --ckpt "$CKPT" --split val --dust-tau "$tau" --num-workers 0 \
          | awk -F': ' '/^val_match_score:/{print $2}')
  [ -z "$score" ] && { echo "tau=$tau -> no score (skip)"; continue; }
  echo "tau=$tau val_match_score=$score"
  awk "BEGIN{exit !($score>$best_score)}" && { best_score=$score; best_tau=$tau; }
done
echo "best dust_tau=$best_tau (val_match_score=$best_score)"

echo "=== TEST GATE (touched once) dust_tau=$best_tau ==="
$PY tracking/cli/eval.py --ckpt "$CKPT" --split test --dust-tau "$best_tau" --num-workers 0
