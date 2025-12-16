#!/bin/bash
# Training script for MeshGraphNets Deforming Plate
# Run from Project2/ directory

cd "$(dirname "$0")"

# Set protobuf compatibility mode for dm-sonnet
# This is needed because dm-sonnet 1.36 was compiled with older protobuf
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Note: Dependencies should be installed before running
# If you get import errors, install with:
#   pip install -r deepmind-research/meshgraphnets/requirements.txt
#   or run: ./fix_tensorflow_versions.sh

# Build command - using 1 trajectory (trajectory_id=0), 500 epochs, Adam optimizer with lr=1e-4
CMD="python deepmind-research/meshgraphnets/train_deforming_plate.py \
  --data_dir=../raw_data \
  --output_dir=checkpoints/meshgraphnet \
  --plots_dir=plots/meshgraphnet \
  --num_epochs=500 \
  --steps_per_epoch=4 \
  --learning_rate=1e-4 \
  --trajectory_id=0"

# Execute command
eval $CMD
