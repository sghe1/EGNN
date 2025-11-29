#!/usr/bin/env bash

set -e

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Set paths
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/egnn"
DATA_DIR="${PROJECT_ROOT}/data/deforming_plate"
OUTPUT_DIR="${PROJECT_ROOT}/results/egnn"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/best_model.pt"

# Default values
NUM_TRAJECTORIES=${1:-10}  # Default to 10 trajectories for evaluation

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_FILE}"
    echo "Please train the model first using: bash scripts/train_egnn.sh"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Change to the egnn-pytorch directory (where evaluate_egnn.py is located)
cd "${PROJECT_ROOT}/egnn-pytorch"

# Run evaluation
python evaluate_egnn.py \
  --checkpoint="${CHECKPOINT_FILE}" \
  --data_dir="${DATA_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_trajectories="${NUM_TRAJECTORIES}" \
  --batch_size=1 \
  --device=cpu

echo ""
echo "Evaluation completed!"
echo "Predictions saved to: ${OUTPUT_DIR}"
echo ""
echo "Files created:"
echo "  - predictions_velocity.npy: Predicted velocities (list of arrays, shape: (T-1, N, 3) per trajectory)"
echo "  - predictions_stress.npy: Predicted stress (list of arrays, shape: (T-1, N, 1) per trajectory)"
echo "  - true_velocity.npy: True velocities (list of arrays, shape: (T-1, N, 3) per trajectory)"
echo "  - true_stress.npy: True stress (list of arrays, shape: (T-1, N, 1) per trajectory)"
echo "  - metrics.json: Evaluation metrics (MSE, MAE for velocity and stress)"

