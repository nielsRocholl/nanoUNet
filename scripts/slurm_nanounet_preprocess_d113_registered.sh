#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=nanounet-preprocess-d113-registered
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_preprocess_d113_registered.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_preprocess_d113_registered.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/gi
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

# Was id 13 / Dataset013_Longitudinal_CT_registered, but Dataset013_Longitudinal_CT (the
# unregistered uclp-pro single-channel dataset) already occupies id 13, and
# convert_id_to_dataset_name() raises "ambiguous dataset id" if >1 Dataset013* folder exists
# under raw/preprocessed/results. Moved to the first free id, 113.
DATASET_ID=113
DS_FOLDER=Dataset113_Longitudinal_CT_registered
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
SRC_PLANS=/nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/${PLANS_NAME}.json
STORAGE=/nnunet_data

# register_longi output (fixed from the old d013_register_out placeholder, which never existed).
REGISTER_OUT=/nnunet_data/unprocessed-universal-lesion-segmentation-registered
TEMPLATE_DJ=/nnunet_data/nnUNet_raw/Dataset013_Longitudinal_CT/dataset.json

export NANOUNET_RAW="${STORAGE}/nnUNet_raw"
export NANOUNET_PREPROCESSED="${STORAGE}/NanoUNet_preprocessed"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
export nnUNet_raw="$NANOUNET_RAW"
export nnUNet_preprocessed="$NANOUNET_PREPROCESSED"
export nnUNet_results="$NANOUNET_RESULTS"
# nanounet_preprocess crashes ("preprocess worker died", no core dump since ulimit -c is 0) with
# more than ~2-3 worker processes hitting the CIFS mount concurrently -- reproduced at -np 4/8/24,
# stable at -np 1/2. Compensate by giving each of the 2 workers more BLAS/OMP threads instead of
# adding more worker processes.
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
NUM_PROC=2

RAW_DS="${NANOUNET_RAW}/${DS_FOLDER}"
PRE_DS="${NANOUNET_PREPROCESSED}/${DS_FOLDER}"

# convert_id_to_dataset_name(113) raises if >1 Dataset113* exists under raw/preprocessed/results.
CONFLICTS=$(ls -d "${NANOUNET_RAW}"/Dataset113* "${NANOUNET_PREPROCESSED}"/Dataset113* "${NANOUNET_RESULTS}"/Dataset113* 2>/dev/null | grep -v "/${DS_FOLDER}$" || true)
if [ -n "$CONFLICTS" ]; then
  echo "FATAL: other Dataset113* folders make id 113 ambiguous:"; echo "$CONFLICTS"
  echo "Use a different free id and rerun."; exit 1
fi

# 1) Build the 2-channel raw dataset from the warped register output (skip if already built).
if [ ! -f "${RAW_DS}/dataset.json" ]; then
  python3 -m nanounet.cli.longi_build \
    --register-out "$REGISTER_OUT" \
    --template-dj "$TEMPLATE_DJ" \
    --out "$RAW_DS"
fi

# 2) Stage dataset.json + plans into the preprocessed dir; duplicate channel-0 CT norm to channel-1
#    (warped BL is CT; identical stats keep preprocessing byte-identical to the single-stream run).
mkdir -p "$PRE_DS"
cp "$RAW_DS/dataset.json" "$PRE_DS/dataset.json"
python3 - "$SRC_PLANS" "$PRE_DS/${PLANS_NAME}.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
p = json.load(open(src))
cfg = p["configurations"]["3d_fullres"]
cfg["normalization_schemes"] = ["CTNormalization", "CTNormalization"]
cfg["use_mask_for_norm"] = [0, 0]
fi = p["foreground_intensity_properties_per_channel"]
fi["1"] = dict(fi["0"])
json.dump(p, open(dst, "w"), indent=2)
print(f"patched plans -> {dst}")
PY

# 3) Preprocess reusing the existing plans (no fingerprint, no planning). --resume makes this
#    safe to rerun after an interruption or the worker-death crash above: without it, a rerun
#    wipes the whole output dir and starts over.
nanounet_preprocess -d "$DATASET_ID" --skip-fingerprint --skip-plan --plans-name "$PLANS_NAME" -np "$NUM_PROC" --resume

# 4) Map warped BL clicks -> preprocessed sidecars <case>_bl_clicks.json.
python3 -m nanounet.cli.longi_clicks -d "$DATASET_ID" --plans "$PLANS_NAME" --clicks-dir "$RAW_DS/clicksTr"

echo "DONE -> $PRE_DS"
