# EGNN Training and Evaluation Guide

This document explains how the EGNN (E(n) Equivariant Graph Neural Network) training and evaluation pipeline works for the deforming plate dataset.

## Overview

The codebase implements an EGNN model to predict velocity and stress fields on a deforming plate mesh. The model operates on graph-structured data where nodes represent mesh points and edges represent mesh connectivity.

## Dataset Structure

### Input Features (8 dimensions per node):
- **Position** (3D): Current 3D world coordinates `(x, y, z)` at time `t`
- **Actuation** (3D): Boundary displacement / input signal for the node
- **Node Type** (2D one-hot): Encoding of node type:
  - `[0, 0]` → NORMAL (plate nodes, where we compute loss)
  - `[1, 0]` → OBSTACLE (boundary nodes, fixed, no loss)
  - `[0, 1]` → HANDLE (actuator nodes, fixed, no loss)

### Targets:
- **Velocity** `(T, N, 3)`: Lagrangian velocity computed as `pos[t] - pos[t-1]`
- **Stress** `(T, N, 1)`: Scalar von-Mises stress per node per timestep

### Mesh Connectivity:
- **Cells** `(C, 4)`: Tetrahedral cell connectivity (4 node indices per cell)
- Used to derive graph edges (element adjacency)

## Normalization Pipeline

The training uses a **MeshGraphNet-style normalization** approach:

### Input Normalization (before model):
- **Positions**: `(pos - pos_mean) / pos_std` → normalized to O(1) range
- **Actuation**: `(act - act_mean) / act_std` → normalized to O(1) range
- **Node type**: Unchanged (one-hot encoding)

### Target Normalization (before loss computation):
- **Velocity**: `vel / vel_std` → normalized by scalar standard deviation
- **Stress**: `(stress - stress_mean) / stress_std` → normalized by mean and std

### Key Points:
- Model operates **entirely in normalized space** (O(1) magnitudes)
- Denormalization happens **ONLY when saving predictions** for visualization
- Normalization statistics are computed from a sample of trajectories and saved to `norm_stats.json`

## Training Process

### Script: `train_egnn.py`

#### What it does:

1. **Data Loading**:
   - Loads trajectories from TFRecord files
   - Computes or loads normalization statistics
   - Converts trajectories to EGNN input format with normalized features

2. **Model Architecture**:
   - **Input MLP**: Projects 8D features → hidden dimension (128)
   - **EGNN Layers**: Stack of 4 EGNN layers for message passing
   - **Velocity Head**: Predicts 3D velocity from node embeddings
   - **Stress Head**: Predicts scalar stress from node embeddings

3. **Training Loop**:
   - For each trajectory and timestep `t` (starting from `t=1`):
     - Uses **teacher forcing**: Ground truth positions at `t-1` to predict `t`
     - Model receives normalized inputs and outputs normalized predictions
     - Loss is computed on normalized values

4. **Loss Function**:
   ```
   Total Loss = Velocity Loss + Stress Loss + Penalties
   ```
   
   - **Velocity Loss**: MSE on normalized velocity (only NORMAL nodes)
   - **Stress Loss**: MSE on normalized stress (only NORMAL nodes)
   - **Penalties**:
     - Zero prediction penalty: Penalizes predicting zero when true stress is non-zero
     - Underestimation penalty: Penalizes severe underestimation (pred < 10% of target)

5. **Node Masking**:
   - Only **NORMAL nodes** (node_type == 0) contribute to loss
   - **OBSTACLE** and **HANDLE** nodes are masked out (they're fixed boundaries)
   - Model automatically sets predictions to zero for non-NORMAL nodes

6. **Training Configuration**:
   - **Warmup**: Skips first 2.5% of timesteps (early timesteps have near-zero stress)
   - **Stress Threshold**: Only trains on timesteps with mean stress > 5000.0
   - **Gradient Clipping**: Max norm = 1.0
   - **Optimizer**: Adam with learning rate 1e-4

7. **Checkpointing**:
   - Saves best model (lowest loss) to `best_model.pt`
   - Saves periodic checkpoints every 5 epochs
   - Saves normalization statistics to `norm_stats.json`

8. **Prediction Saving** (if `--save_predictions` is enabled):
   - Runs inference on evaluation trajectories
   - Uses **autoregressive prediction**: Uses predicted positions from previous step
   - Denormalizes predictions before saving
   - Saves to `checkpoints/egnn/predictions/epoch_{N}/`:
     - `predictions_velocity.npy`: Predicted velocities
     - `predictions_stress.npy`: Predicted stress
     - `predictions_positions.npy`: Predicted positions
     - `true_velocity.npy`: Ground truth velocities
     - `true_stress.npy`: Ground truth stress
     - `true_positions.npy`: Ground truth positions

#### Command Line Arguments:

```bash
python train_egnn.py \
  --data_dir data/deforming_plate \
  --checkpoint_dir checkpoints/egnn \
  --dataset_fraction 0.1 \
  --batch_size 1 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --hidden_dim 128 \
  --depth 4 \
  --save_predictions \
  --save_predictions_every 1 \
  --num_trajectories_for_predictions 5 \
  --max_timesteps 400 \
  --min_stress_threshold 5000.0 \
  --warmup_fraction 0.025
```

## Evaluation Process

### Script: `visualize_predictions.py`

#### What it does:

1. **Loads Predictions**:
   - Loads saved predictions from `checkpoints/egnn/predictions/epoch_{N}/`
   - Loads ground truth values
   - Loads mesh connectivity (cells) from dataset

2. **Visualization**:
   - Creates 3D mesh visualizations comparing true vs predicted stress/velocity
   - Uses color mapping to show stress/velocity magnitude
   - Side-by-side comparison of true and predicted values

3. **Features**:
   - Can visualize specific timesteps or multiple timesteps
   - Supports different color modes: `stress`, `velocity`, `position_norm`
   - Can generate scatter plots of predicted vs true stress

#### Command Line Arguments:

```bash
python visualize_predictions.py \
  --checkpoint_dir checkpoints/egnn/predictions \
  --epoch 100 \
  --traj_idx 0 \
  --t 20 \
  --data_dir ../data/deforming_plate \
  --color stress \
  --output visualization.png
```

## Key Concepts

### Teacher Forcing vs Autoregressive:

- **Teacher Forcing** (training): Uses ground truth positions at `t-1` to predict `t`
  - Prevents error accumulation during training
  - Allows model to learn single-step predictions correctly

- **Autoregressive** (evaluation): Uses predicted positions from previous step
  - More realistic evaluation scenario
  - Errors can accumulate over time
  - Used when saving predictions for visualization

### Node Types:

- **NORMAL (0)**: Plate nodes that deform and have stress
  - These are the nodes where loss is computed
  - Model should predict non-zero velocity and stress

- **OBSTACLE (1)**: Fixed boundary nodes
  - No loss computed
  - Model predictions are masked to zero (correct behavior)

- **HANDLE (3)**: Actuator nodes
  - No loss computed
  - Model predictions are masked to zero (correct behavior)

### Normalization:

- All inputs and targets are normalized to O(1) range
- This ensures stable training and balanced loss terms
- Denormalization only happens for visualization/saving

## File Structure

```
egnn-pytorch/
├── train_egnn.py          # Training script
├── visualize_predictions.py # Evaluation/visualization script
├── EGNN.py                # Model architecture
├── data_loader_egnn.py    # Data loading utilities
├── README.md              # This file
└── checkpoints/egnn/
    ├── best_model.pt      # Best model checkpoint
    ├── norm_stats.json    # Normalization statistics
    └── predictions/       # Saved predictions
        └── epoch_{N}/
            ├── predictions_velocity.npy
            ├── predictions_stress.npy
            ├── predictions_positions.npy
            ├── true_velocity.npy
            ├── true_stress.npy
            └── true_positions.npy
```

## Troubleshooting

### Model predicts zero stress at boundaries:

- **Check node types**: Boundary nodes should be NORMAL (type 0), not OBSTACLE (type 1)
- **Check masking**: Only OBSTACLE and HANDLE nodes should be masked to zero
- **Check training**: Model may need more training or penalty terms to learn boundary stress

### Loss not decreasing:

- Check normalization statistics are reasonable
- Check that stress threshold isn't too high (skipping all timesteps)
- Check gradient flow (debug output shows gradient norms)

### Predictions look wrong:

- Verify normalization statistics match between training and evaluation
- Check that autoregressive mode is used correctly
- Verify mesh connectivity (cells) are loaded correctly

## References

- Paper: "E(n) Equivariant Graph Neural Networks" (arXiv:2102.09844v3)
- Dataset: Deforming Plate from MeshGraphNets
- Normalization: MeshGraphNet-style normalization pipeline
