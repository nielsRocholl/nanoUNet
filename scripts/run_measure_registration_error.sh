#!/usr/bin/env bash
# Produces /nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json. One-time step:
# rerun only if Longitudinal-CT derivatives change; do not rerun to "refresh" a working table.
set -euo pipefail

cd /root/nanounet
python3 scripts/measure_registration_error.py \
  --longi-root /nnunet_data/Longitudinal-CT \
  --out /nnunet_data/Longitudinal-CT/derivatives/registration_error_table.json \
  --spacing 1.25 0.781 0.789 \
  --max-lesion-offset-vox 100.0 \
  --max-case-median-vox 20.0 \
  --min-per-bin 30
