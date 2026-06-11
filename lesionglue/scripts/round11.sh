#!/usr/bin/env bash
# Round 11: rebuild v6_geo cache + paired 5-fold CV geo_off vs geo_on + tau sweep + test gate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=python3
RUNS=${RUNS:-runs/round11}
PROJ=${WANDB_PROJECT:-lesion-tracking}
CFG_OFF=configs/geo_off.json
CFG_ON=configs/geo_on.json
mkdir -p "$RUNS"

echo "=== import check ==="
$PY -c "import tracking.consistency, tracking.matcher; print('R11 imports OK')"

echo "=== cache rebuild (v6_geo) via dataset process on train/val/test ==="
for split in train val test; do
  $PY -c "
from pathlib import Path
from tracking.data.features import CACHE_TAG
from tracking.data.dataset import LesionDataset
import os
root = Path(os.environ.get('DATA_ROOT', 'data'))
if not root.is_dir():
    raise SystemExit('DATA_ROOT missing; set DATA_ROOT to lesion data root')
LesionDataset(root=root, split=split)
print(f'{split} cache tag={CACHE_TAG} OK')
"
done

echo "=== CV geo_off ($CFG_OFF) ==="
if [ ! -f "$RUNS/cv_geo_off/cv_summary.json" ]; then
  $PY tracking/cli/cv.py --config "$CFG_OFF" --out "$RUNS/cv_geo_off" --wandb --wandb-project "$PROJ" --wandb-run-name geo_off
fi

echo "=== CV geo_on ($CFG_ON) ==="
if [ ! -f "$RUNS/cv_geo_on/cv_summary.json" ]; then
  $PY tracking/cli/cv.py --config "$CFG_ON" --out "$RUNS/cv_geo_on" --wandb --wandb-project "$PROJ" --wandb-run-name geo_on
fi

$PY - "$RUNS" <<'EOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
for tag in ("cv_geo_off", "cv_geo_on"):
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

WIN_CFG=$CFG_ON
WIN_TAG=geo_on
FOUT="$RUNS/final"
echo "=== FINAL retrain winner ($WIN_CFG) ==="
if [ ! -f "$FOUT/best.ckpt" ]; then
  $PY tracking/cli/train.py --config "$WIN_CFG" --out "$FOUT" --wandb --wandb-project "$PROJ" --wandb-run-name final_geo_on
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
