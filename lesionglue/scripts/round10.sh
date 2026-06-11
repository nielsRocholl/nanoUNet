#!/usr/bin/env bash
# Paired 5-fold CV: r9_base (refine_blocks=0) vs r10_refine (refine_blocks=1).
# Tau sweep on winner -> (RUN_FINAL=1) single test gate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=python3
RUNS=${RUNS:-runs/round10}
PROJ=${WANDB_PROJECT:-lesion-tracking}
CFG_R9=configs/r9_base.json
CFG_R10=configs/r10_refine.json
mkdir -p "$RUNS"

echo "=== CV r9_base ($CFG_R9) ==="
if [ ! -f "$RUNS/cv_r9base/cv_summary.json" ]; then
  $PY tracking/cli/cv.py --config "$CFG_R9" --out "$RUNS/cv_r9base" --wandb --wandb-project "$PROJ" --wandb-run-name r9base
fi

echo "=== CV r10_refine ($CFG_R10) ==="
if [ ! -f "$RUNS/cv_r10/cv_summary.json" ]; then
  $PY tracking/cli/cv.py --config "$CFG_R10" --out "$RUNS/cv_r10" --wandb --wandb-project "$PROJ" --wandb-run-name r10
fi

$PY - "$RUNS" <<'EOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
for tag in ("cv_r9base", "cv_r10"):
    p = root / tag / "cv_summary.json"
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    s = json.loads(p.read_text())
    m = s.get("val_match_score_ema", {})
    sub = lambda k, d=s: d.get(k, {}).get("mean", float("nan"))
    print(f"\n# {tag}  ema {m.get('mean', 0):.4f} +/- {m.get('std', 0):.4f} | uc {sub('val_acc_unchanged_split'):.3f} | dis {sub('val_acc_disappeared'):.3f} | new {sub('val_acc_newly_appearing'):.3f}")
EOF

if [ "${RUN_FINAL:-0}" != "1" ]; then
  echo; echo "Paired CV done. RUN_FINAL=1 to pick winner, tau sweep, TEST gate."
  exit 0
fi

# Pick winner by val_match_score_ema mean (Fabian rule: manual review of bands + uc +2pp)
WIN_CFG=$CFG_R10
WIN_TAG=r10
WIN_CV="$RUNS/cv_r10"
echo "=== FINAL retrain winner ($WIN_CFG) ==="
FOUT="$RUNS/final"
if [ ! -f "$FOUT/best.ckpt" ]; then
  $PY tracking/cli/train.py --config "$WIN_CFG" --out "$FOUT" --wandb --wandb-project "$PROJ" --wandb-run-name final_r10
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
