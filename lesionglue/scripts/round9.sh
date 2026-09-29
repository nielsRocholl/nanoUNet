#!/usr/bin/env bash
# Train r9_base from configs/base.json -> dust_tau sweep -> (RUN_FINAL=1) test gate once.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=python3
CFG=${CONFIG:-configs/base.json}
RUNS=${RUNS:-runs/round9}
PROJ=${WANDB_PROJECT:-lesion-tracking}
mkdir -p "$RUNS"

echo "=== CV 5-fold ($CFG) ==="
if [ ! -f "$RUNS/cv_summary.json" ]; then
  $PY tracking/cli/cv.py --config "$CFG" --out "$RUNS" --wandb --wandb-project "$PROJ" --wandb-run-name r9_base
fi

$PY - "$RUNS" <<'EOF'
import sys, json, os
root = sys.argv[1]
p = f"{root}/cv_summary.json"
if not os.path.isfile(p):
    raise SystemExit(f"missing {p}")
s = json.load(open(p))
m = s.get("val_match_score_ema", {})
sub = lambda k: s.get(k, {}).get("mean", float("nan"))
print(f"\n# CV summary  ema {m.get('mean', 0):.4f} +/- {m.get('std', 0):.4f} | uc {sub('val_acc_unchanged_split'):.3f} | dis {sub('val_acc_disappeared'):.3f} | new {sub('val_acc_newly_appearing'):.3f}")
EOF

if [ "${RUN_FINAL:-0}" != "1" ]; then
  echo; echo "CV done. RUN_FINAL=1 to retrain + tau sweep + TEST gate."
  exit 0
fi

FOUT="$RUNS/final"
echo "=== FINAL retrain ($CFG) ==="
if [ ! -f "$FOUT/best.ckpt" ]; then
  $PY tracking/cli/train.py --config "$CFG" --out "$FOUT" --wandb --wandb-project "$PROJ" --wandb-run-name final_r9_base
fi
CKPT="$FOUT/best.ckpt"

echo "=== dust_tau sweep on val ==="
best_tau=0.20; best_score=-1
for tau in 0.10 0.15 0.18 0.20 0.22 0.25 0.30 0.35; do
  score=$($PY tracking/cli/eval.py --ckpt "$CKPT" --split val --dust-tau "$tau" --num-workers 0 \
          | awk -F': ' '/^val_match_score:/{print $2}')
  [ -z "$score" ] && { echo "tau=$tau -> no score (skip)"; continue; }
  echo "tau=$tau val_match_score=$score"
  awk "BEGIN{exit !($score>$best_score)}" && { best_score=$score; best_tau=$tau; }
done
echo "best dust_tau=$best_tau (val_match_score=$best_score)"

echo "=== TEST GATE (once) dust_tau=$best_tau ==="
$PY tracking/cli/eval.py --ckpt "$CKPT" --split test --dust-tau "$best_tau" --num-workers 0
