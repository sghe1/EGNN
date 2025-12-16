#!/bin/bash
# Simple script to run MeshGraphNet training on trajectory 0 for 500 epochs
# This script handles environment activation and runs training

set -e

cd "$(dirname "$0")"

# Check which environment to use
if conda env list | grep -q "^meshgraphnet310 "; then
    ENV_NAME="meshgraphnet310"
elif conda env list | grep -q "^meshgraphnet "; then
    ENV_NAME="meshgraphnet"
else
    echo "Error: No meshgraphnet environment found!"
    echo "Please run: bash quick_setup_python310.sh"
    exit 1
fi

echo "Activating conda environment: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Set environment variable
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo ""
echo "=========================================="
echo "Running MeshGraphNet Training"
echo "=========================================="
echo "Configuration:"
echo "  - Trajectory: 0"
echo "  - Epochs: 500"
echo "  - Validation frequency: Every 10 epochs"
echo ""

# Run training
bash run_meshgraphnets.sh

