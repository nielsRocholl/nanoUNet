#!/bin/bash
# Warp every BL/FU pair into the FU frame (itk-elastix), resumable.
#
# A case counts as done only when all 6 output files exist (inputsTrBL/targetsTrBL
# .nii.gz+.json, inputsTrFU/targetsTrFU .nii.gz+.json, meta csv) — NOT just the
# warped BL, since output.py used to crash mid-write on CIFS (shutil.copy2 -> utime
# -> PermissionError) leaving inputsTrBL populated but FU/meta missing.

#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=nanounet-register-longi
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_register_longi.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_register_longi.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NANOUNET_REG_THREADS=1

DATA_ROOT=/nnunet_data/unprocessed-universal-lesion-segmentation
OUT=/nnunet_data/unprocessed-universal-lesion-segmentation-registered
J=6

mkdir -p "$OUT/logs"

python3 <<'PY' > "$OUT/cases.list"
import glob, os
root = os.environ.get("DATA_ROOT", "/nnunet_data/unprocessed-universal-lesion-segmentation")
for f in sorted(glob.glob(os.path.join(root, "inputsTrFU", "*.nii.gz"))):
    stem = os.path.basename(f)[:-len(".nii.gz")]
    if os.path.isfile(os.path.join(root, "inputsTrBL", stem + ".nii.gz")):
        pid, idx = stem.rsplit("_", 1)
        print(f"{pid}\t{idx}")
PY

complete() {
  local s="$1"
  [[ -f "$OUT/inputsTrBL/${s}.nii.gz" && -f "$OUT/inputsTrBL/${s}.json" \
     && -f "$OUT/targetsTrBL/${s}.nii.gz" && -f "$OUT/inputsTrFU/${s}.nii.gz" \
     && -f "$OUT/inputsTrFU/${s}.json" && -f "$OUT/targetsTrFU/${s}.nii.gz" \
     && -f "$OUT/meta/${s%_*}.csv" ]]
}

n_total=0
n_skip=0
n_run=0
while IFS=$'\t' read -r pid idx; do
  n_total=$((n_total + 1))
  stem="${pid}_${idx}"
  if complete "$stem"; then
    n_skip=$((n_skip + 1))
    continue
  fi
  n_run=$((n_run + 1))
  while (( $(jobs -rp | wc -l) >= J )); do sleep 5; done
  python3 -m nanounet.cli.register_longi \
    --data-root "$DATA_ROOT" --out "$OUT" \
    --pid "$pid" --idx "$idx" --threads 1 \
    > "$OUT/logs/${stem}.log" 2>&1 &
done < "$OUT/cases.list"
wait

echo "total pairs: $n_total  already complete: $n_skip  (re)run: $n_run"
echo "now complete: $(find "$OUT/targetsTrFU" -name '*.nii.gz' | wc -l) / $n_total"
