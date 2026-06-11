#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=24G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=lesion-round11-cv
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/lesion_round11_%j.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/lesion_round11_%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data,/home/nielsrocholl/projects/git_projects/lesion-tracking:/home/nielsrocholl/projects/git_projects/lesion-tracking
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

REPO=/home/nielsrocholl/projects/git_projects/lesion-tracking
cd "$REPO"

export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DATA_ROOT=/nnunet_data/lesion_tracking
export RUNS=/nnunet_data/lesion_tracking/runs/round11
export WANDB_PROJECT=lesion-tracking
export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$RUNS" "$PIP_CACHE_DIR"

pip3 install -r requirements.txt
python3 -c "import torch_geometric; print(f'torch_geometric {torch_geometric.__version__} OK')"
python3 -c "import tracking.consistency, tracking.matcher; print('R11 imports OK')"

bash scripts/round11.sh
