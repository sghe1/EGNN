#!/usr/bin/env bash

set -e

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Set paths
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/egnn"
DATA_DIR="${PROJECT_ROOT}/data/deforming_plate"
DATASET_FRACTION=${1:-0.1}  # Default to 10% of dataset
NUM_EPOCHS=${2:-10}          # Default to 10 epochs

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"

# Change to the egnn-pytorch directory (where train_egnn.py is located)
cd "${PROJECT_ROOT}/egnn-pytorch"

# Run training
python train_egnn.py \
  --data_dir="${DATA_DIR}" \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --dataset_fraction="${DATASET_FRACTION}" \
  --num_epochs="${NUM_EPOCHS}" \
  --batch_size=1 \
  --learning_rate=1e-4 \
  --hidden_dim=128 \
  --depth=4 \
  --device=cuda

echo "Training completed. Checkpoints saved to: ${CHECKPOINT_DIR}"

