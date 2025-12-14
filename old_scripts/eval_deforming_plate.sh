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
ROLLOUT_PATH="${PROJECT_ROOT}/rollouts/deforming_plate_rollout.pkl"
ROLLOUT_SPLIT=${1:-valid}  # Allow passing split as first argument, default 'valid'
NUM_ROLLOUTS=${2:-10}       # Allow passing num_rollouts as second argument, default 10

# Create rollout directory
mkdir -p "$(dirname "${ROLLOUT_PATH}")"

# Change to the meshgraphnets directory
cd "${PROJECT_ROOT}/Project2/deepmind-research"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/Project2/deepmind-research:${PYTHONPATH}"

# Run evaluation
python -m meshgraphnets.run_model \
  --mode=eval \
  --model=deforming_plate \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --dataset_dir="${DATA_DIR}" \
  --rollout_path="${ROLLOUT_PATH}" \
  --rollout_split="${ROLLOUT_SPLIT}" \
  --num_rollouts="${NUM_ROLLOUTS}"

echo "Rollouts saved to: ${ROLLOUT_PATH}"

