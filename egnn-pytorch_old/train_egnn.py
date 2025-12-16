#!/usr/bin/env python3
"""
Training script for EGNN on deforming plate dataset.
Uses data_loader_egnn.py to load data and MeshEGNN model.

This is a thin wrapper entrypoint that calls egnn_train.main().
All training logic has been moved to egnn_train.py for better modularity.

Dataset Structure:
- Input features: [position(3), actuation(3), node_type_one_hot(2)] = 8 dims
- Targets: velocity (T, N, 3), stress (T, N, 1)
- Coordinates: world_pos (T, N, 3) - same as position in features

Normalization Pipeline (MeshGraphNet-style):
- ALL inputs are normalized BEFORE going into the model:
  * Positions: (pos - pos_mean) / pos_std
  * Actuation: (act - act_mean) / act_std
  * Node type: unchanged (one-hot)
- ALL targets are normalized BEFORE computing loss:
  * Velocity: vel / vel_std
  * Stress: (stress - stress_mean) / stress_std
- Model operates entirely in normalized space (O(1) magnitudes)
- Denormalization happens ONLY when saving predictions for visualization

Training Process:
- For each timestep t, model predicts normalized velocity and stress at time t
- Loss is computed on normalized values
- Denormalization only for saving predictions
"""

import sys
import os

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
egnn_pytorch_dir = os.path.join(script_dir, 'egnn-pytorch')
sys.path.insert(0, script_dir)
sys.path.insert(0, egnn_pytorch_dir)

from egnn_train import main

if __name__ == '__main__':
    main()
