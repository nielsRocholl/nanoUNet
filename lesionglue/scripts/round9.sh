#!/usr/bin/env bash
# Train r9_base from lesionglue/configs/base.json -> dust_tau sweep -> (RUN_FINAL=1) test gate once.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
PY=python3
CFG=${CONFIG:-configs/base.json}
RUNS=${RUNS:-runs/round9}
PROJ=${WANDB_PROJECT:-lesion-tracking}
mkdir -p "$RUNS"

echo "=== CV 5-fold ($CFG) ==="
if [ ! -f "$RUNS/cv_summary.json" ]; then
  $PY lesionglue/cli/cv.py --config "$CFG" --out "$RUNS" --wandb --wandb-project "$PROJ" --wandb-run-name r9_base
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
  $PY lesionglue/cli/train.py --config "$CFG" --out "$FOUT" --wandb --wandb-project "$PROJ" --wandb-run-name final_r9_base
fi
CKPT="$FOUT/best.ckpt"

echo "=== dust_tau sweep on val ==="
$PY lesionglue/cli/eval.py --ckpt "$CKPT" --split val --num-workers 0 --out "$RUNS/tau_sweep_val.json" \
  --dust-tau 0.10 --dust-tau 0.15 --dust-tau 0.18 --dust-tau 0.20 --dust-tau 0.22 --dust-tau 0.25 --dust-tau 0.30 --dust-tau 0.35
best_tau=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['selected']['dust_tau'])" "$RUNS/tau_sweep_val.json")
echo "best dust_tau=$best_tau"

echo "=== TEST GATE (once) dust_tau=$best_tau ==="
$PY lesionglue/cli/eval.py --ckpt "$CKPT" --split test --dust-tau "$best_tau" --num-workers 0
