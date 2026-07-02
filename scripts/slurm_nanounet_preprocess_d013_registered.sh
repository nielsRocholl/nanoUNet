#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=nanounet-preprocess-d013-registered
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_preprocess_d013_registered.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/nanounet_preprocess_d013_registered.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/gi
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

DATASET_ID=13
DS_FOLDER=Dataset013_Longitudinal_CT_registered
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv
SRC_PLANS=/nnunet_data/NanoUNet_preprocessed/Dataset999_Merged/${PLANS_NAME}.json
NUM_PROC=24
STORAGE=/nnunet_data

# EDIT: warped register_longi output dir, and a dataset.json carrying the d013 labels/file_ending.
REGISTER_OUT=/nnunet_data/nnUNet_raw/d013_register_out
TEMPLATE_DJ=/nnunet_data/nnUNet_raw/Dataset013_Longitudinal_CT/dataset.json

export NANOUNET_RAW="${STORAGE}/nnUNet_raw"
export NANOUNET_PREPROCESSED="${STORAGE}/NanoUNet_preprocessed"
export NANOUNET_RESULTS="${STORAGE}/NanoUNet_results"
export nnUNet_raw="$NANOUNET_RAW"
export nnUNet_preprocessed="$NANOUNET_PREPROCESSED"
export nnUNet_results="$NANOUNET_RESULTS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RAW_DS="${NANOUNET_RAW}/${DS_FOLDER}"
PRE_DS="${NANOUNET_PREPROCESSED}/${DS_FOLDER}"

# convert_id_to_dataset_name(13) raises if >1 Dataset013* exists under raw/preprocessed/results.
CONFLICTS=$(ls -d "${NANOUNET_RAW}"/Dataset013* "${NANOUNET_PREPROCESSED}"/Dataset013* "${NANOUNET_RESULTS}"/Dataset013* 2>/dev/null | grep -v "/${DS_FOLDER}$" || true)
if [ -n "$CONFLICTS" ]; then
  echo "FATAL: other Dataset013* folders make id 13 ambiguous:"; echo "$CONFLICTS"
  echo "Use a free id (e.g. Dataset113_...) and rerun."; exit 1
fi

# 1) Build the 2-channel raw dataset from the warped register output (skip if already built).
if [ ! -f "${RAW_DS}/dataset.json" ]; then
  python -m nanounet.cli.longi_build \
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

# 3) Preprocess reusing the existing plans (no fingerprint, no planning).
nanounet_preprocess -d "$DATASET_ID" --skip-fingerprint --skip-plan --plans-name "$PLANS_NAME" -np "$NUM_PROC"

# 4) Map warped BL clicks -> preprocessed sidecars <case>_bl_clicks.json.
python -m nanounet.cli.longi_clicks -d "$DATASET_ID" --plans "$PLANS_NAME" --clicks-dir "$RAW_DS/clicksTr"

echo "DONE -> $PRE_DS"
