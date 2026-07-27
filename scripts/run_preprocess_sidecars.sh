#!/usr/bin/env bash
# Regenerates *_centroids.json sidecars (seed_zyx, volume_vox) for Dataset999_Merged and
# Dataset114_longi, after backing up the existing sidecars. Overwrites files the current working
# checkpoints depend on -- backup is mandatory, refuses to run if a backup already exists.
set -euo pipefail

BACKUP_DIR=/nnunet_data/prompt_sensitivity/sidecar_backup
BACKUP_FILE="$BACKUP_DIR/centroids_before.tgz"
PLANS_NAME=nnUNetResEncUNetLPlans_h200_smallpv

if [ -f "$BACKUP_FILE" ]; then
  echo "Refusing to overwrite existing backup: $BACKUP_FILE" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
cd /nnunet_data/NanoUNet_preprocessed
tar czf "$BACKUP_FILE" \
  Dataset999_Merged/nnUNetPlans_3d_fullres/*_centroids.json \
  Dataset114_longi/nnUNetPlans_3d_fullres/*_centroids.json

cd /root/nanounet
nanounet_preprocess -d 999 --plans-name "$PLANS_NAME" --sidecars-only
nanounet_preprocess -d 114 --plans-name "$PLANS_NAME" --sidecars-only
