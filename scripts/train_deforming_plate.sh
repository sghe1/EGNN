#!/usr/bin/env bash

set -e

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Activate the meshgraphnets conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate meshgraphnets

# Set paths
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/deforming_plate"
DATA_DIR="${PROJECT_ROOT}/data/deforming_plate"
NUM_STEPS=${1:-10000}  # Allow passing num_steps as first argument, default 10000

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"

# Change to the meshgraphnets directory
cd "${PROJECT_ROOT}/Project2/deepmind-research"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/Project2/deepmind-research:${PYTHONPATH}"

# Run training with 50% of the dataset
python -m meshgraphnets.run_model \
  --mode=train \
  --model=deforming_plate \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --dataset_dir="${DATA_DIR}" \
  --num_training_steps="${NUM_STEPS}" \
  --dataset_fraction=0.5

