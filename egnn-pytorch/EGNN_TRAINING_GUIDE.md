# EGNN Training Guide - Deforming Plate Dataset

This document provides a complete guide to running the EGNN (E(n) Equivariant Graph Neural Network) model on the deforming plate dataset.

## Table of Contents
1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Evaluation and Results](#evaluation-and-results)
7. [Key Modifications Made](#key-modifications-made)
8. [Running the Model](#running-the-model)

---

## Overview

The EGNN model is trained to predict **velocity** (3D) and **stress** (1D) for each node in a deforming plate mesh over time. The model takes as input:
- Node features (8-dimensional: [node_type(1), vel(3), acc(3), stress(1)])
- Node coordinates (3D positions)
- Adjacency matrix (graph structure)

And outputs:
- **Predicted velocity (3D)**: Computed using the formula `v_i^{l+1} = phi_v(h_i^l) * v_i^{init} + C * sum_{j != i} (x_i^l - x_j^l) * phi_x(m_ij)`
- **Predicted stress (1D)**: Directly predicted by a dedicated head from node embeddings
- **Updated positions (3D)**: Computed as `x_i^{l+1} = x_i^l + v_i^{l+1}` from predicted velocity

**Note**: The velocity prediction follows the EGNN paper formula (arXiv:2102.09844v3), combining initial velocity modulation with neighbor interactions. Coordinates are updated from the predicted velocity, not from EGNN's internal coordinate updates.

---

## Setup and Installation

### Step 1: Install Required Dependencies

```bash
# Navigate to the project directory
cd /Users/tommasobasile/Desktop/SCRIVANIA/MA3/ML/ML_project/MLproject2

# Install required packages
pip install tfrecord einops egnn-pytorch torch numpy tqdm
```

**Packages installed:**
- `tfrecord`: For reading TFRecord dataset files
- `einops`: Tensor operations (required by egnn-pytorch)
- `egnn-pytorch`: EGNN implementation
- `torch`: PyTorch deep learning framework
- `numpy`: Numerical computations
- `tqdm`: Progress bars

### Step 2: Verify Dataset Location

The dataset should be located at:
```
data/deforming_plate/
├── train.tfrecord
├── valid.tfrecord
├── test.tfrecord
└── meta.json
```

---

## Dataset Preparation

### Dataset Structure

- **Format**: TFRecord files containing trajectory data
- **Trajectories**: Each trajectory contains 400 timesteps
- **Nodes**: Variable number of nodes per mesh (typically 800-1000)
- **Features**: 8-dimensional feature vector per node
- **Targets**: 
  - Velocity: 3D vector (computed as position difference)
  - Stress: 1D scalar value

### Dataset Fraction

The training script supports using a fraction of the dataset:
- `--dataset_fraction 0.1`: Use 10% of the dataset (100 trajectories)
- `--dataset_fraction 1.0`: Use full dataset (1000 trajectories)

---

## Model Architecture

### MeshEGNN Model

The model follows the EGNN paper (arXiv:2102.09844v3) formulas for velocity and coordinate prediction. The architecture consists of:

1. **Input MLP**: Linear layer mapping 8D features → hidden dimension (128)
2. **EGNN Network**: 4 layers of E(n) Equivariant Graph Neural Network (updates node embeddings `h`)
3. **Message computation MLP (φ_e)**: Computes messages `m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)`
4. **Velocity modulation MLP (φ_v)**: Outputs scalar `gamma = phi_v(h_i)` to modulate initial velocity
5. **Neighbor interaction MLP (φ_x)**: Outputs scalar weight `phi_x(m_ij)` for neighbor contributions
6. **Stress head**: Linear layer mapping embeddings → 1D stress at each node

**Model Parameters**: ~933,208 parameters

### Forward Pass Details

The model's forward pass follows the EGNN paper formulas:

1. **Input**: Node features `feats` (B, N, 8), coordinates `coors` (B, N, 3), adjacency `adj_mat` (N, N)
2. **Extract initial velocity**: `v_i^{init}` from features (indices 1:4)
3. **Input MLP**: Projects features to hidden dimension → `h` (B, N, 128)
4. **EGNN layers**: Updates node embeddings:
   - `h, _ = egnn(h, coors, adj_mat)`
   - `h` contains learned node embeddings (we ignore EGNN's internal coordinate updates)
5. **Message computation**:
   - `m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)` for all pairs
6. **Velocity prediction** (following paper formula):
   - `gamma = phi_v(h_i)` → scalar gate (B, N, 1)
   - `neighbor_term = sum_{j != i} (x_i - x_j) * phi_x(m_ij)` → (B, N, 3)
   - `pred_vel = gamma * v_i^{init} + C * neighbor_term` → (B, N, 3)
7. **Stress prediction**:
   - `pred_stress = stress_head(h)` → (B, N, 1)
8. **Coordinate prediction** (following paper formula):
   - `pred_coors = coors + pred_vel` → (B, N, 3)
9. **Return**: `(pred_vel, pred_stress, pred_coors)`

**Key formulas** (from EGNN paper):
- **Velocity**: `v_i^{l+1} = phi_v(h_i^l) * v_i^{init} + C * sum_{j != i} (x_i^l - x_j^l) * phi_x(m_ij)`
- **Coordinates**: `x_i^{l+1} = x_i^l + v_i^{l+1}`
- **Messages**: `m_ij = phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2)`

### Key Features

- **E(n) Equivariance**: Model respects rotation and translation symmetries
- **Graph Structure**: Uses adjacency matrix to process mesh connectivity
- **Message Passing**: Computes edge messages `m_ij` using node embeddings and distances
- **Velocity Formula**: Combines initial velocity modulation with neighbor interactions
- **Coordinate Integration**: Updates coordinates from predicted velocity (kinematic integration)
- **Multi-task Learning**: Predicts both velocity and stress simultaneously from shared embeddings
- **Paper-compliant**: Architecture follows EGNN paper formulas (arXiv:2102.09844v3)

---

## Training Process

### Loss Function

The model uses a **weighted combined loss**:

```python
loss = velocity_loss_weight * MSE(velocity_pred, velocity_true) + MSE(stress_pred, stress_true)
```

**Why scaling + weighting?**
- Velocity values are very small (~0.0002 magnitude), so we scale them by `VELOCITY_SCALE = 1e4` inside the loss.
- After scaling, velocities and stress have comparable magnitudes, so a modest `velocity_loss_weight` (default `1.0`) balances the objectives.

### Training Configuration

**Default hyperparameters:**
- Learning rate: `1e-4`
- Batch size: `1` (one trajectory at a time)
- Hidden dimension: `128`
- EGNN depth: `4` layers
- Optimizer: Adam
- Gradient clipping: `max_norm=1.0`

### Training Monitoring

The training script tracks:
- **Total loss**: Combined weighted loss
- **Velocity loss**: MSE for velocity predictions
- **Stress loss**: MSE for stress predictions

These are displayed in the progress bar and printed after each epoch.

### Velocity Normalization

Velocities in the dataset are extremely small (~1e‑4). To keep gradients well‑scaled, training applies a fixed normalization factor:

- Constant: `VELOCITY_SCALE = 1e4`
- Usage: both predicted and target velocities are multiplied by this scale **inside the loss only**
- Outputs: saved predictions remain in the original physical units (no scaling applied when writing `.npy` files)

**Theoretical correctness**: Scaling both predicted and target values by the same constant before computing MSE is mathematically equivalent to multiplying the loss by the square of that constant. This is a standard technique for numerical stability and does not change the optimization problem—it only reweights the gradient contributions. The model still learns to predict velocities in the original physical units.

If your dataset uses a different time step, adjust `VELOCITY_SCALE` in `train_egnn.py` so that scaled velocities are `O(1)`.

**Note on stress**: Stress values (~10³–10⁵) are already in a reasonable range, so no scaling is currently applied. If needed, you could add `STRESS_SCALE` similar to velocity scaling.

---

## Evaluation and Results

### Saving Predictions

To save predictions and ground truth values during training:

```bash
--save_predictions                    # Enable saving
--save_predictions_every 1           # Save every epoch
--num_trajectories_for_predictions 10 # Save for first 10 trajectories
```

**Output structure:**
```
checkpoints/egnn/predictions/
└── epoch_N/
    ├── predictions_velocity.npy    # Shape: (num_traj, T-1, N, 3)
    ├── predictions_stress.npy      # Shape: (num_traj, T-1, N, 1)
    ├── predictions_positions.npy   # Shape: (num_traj, T-1, N, 3)
    ├── true_velocity.npy           # Shape: (num_traj, T-1, N, 3)
    ├── true_stress.npy             # Shape: (num_traj, T-1, N, 1)
    └── true_positions.npy          # Shape: (num_traj, T-1, N, 3)
```

### Evaluation Script

Use `evaluate_egnn.py` to evaluate a trained model:

```bash
python evaluate_egnn.py \
  --checkpoint checkpoints/egnn/best_model.pt \
  --data_dir data/deforming_plate \
  --output_dir results/egnn_eval \
  --dataset_fraction 0.1
```

### Comparison Script

Use `compare_predictions.py` to analyze results:

```bash
python compare_predictions.py \
  --data_dir test_results/egnn_quick_test \
  --plot
```

This generates:
- Detailed statistics (MSE, MAE per trajectory)
- Comparison plots (predictions vs ground truth)
- Error analysis over time

---

## Key Modifications Made

### 1. Velocity Prediction Following EGNN Paper Formulas

**Design**: The model implements the velocity prediction formula from the EGNN paper (arXiv:2102.09844v3):
- **φ_v MLP**: `nn.Sequential` outputting scalar `gamma = phi_v(h_i)` to modulate initial velocity
- **φ_e MLP**: `nn.Sequential` computing messages `m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)`
- **φ_x MLP**: `nn.Sequential` outputting scalar weight `phi_x(m_ij)` for neighbor interactions
- **Velocity formula**: `v_i^{l+1} = gamma * v_i^{init} + C * sum_{j != i} (x_i - x_j) * phi_x(m_ij)`
- **Coordinate formula**: `x_i^{l+1} = x_i^l + v_i^{l+1}`

**Model output**: Returns `(pred_vel, pred_stress, pred_coors)` where `pred_coors = coors + pred_vel`.

**Location**: `train_egnn.py`, lines 47-157, `myEGNN/EGNN.py`

### 2. Stress Head

**Design**: Simple linear head for stress prediction:
- **Stress head**: `nn.Linear(hidden_dim, 1)` - predicts 1D stress directly from node embeddings

### 3. Velocity Scaling and Weighted Loss

**Problem**: Velocity values (~0.0002) are much smaller than stress values (~18,000), causing stress loss to dominate.

**Solution**: 
- Added `VELOCITY_SCALE = 1e4` constant to scale velocities inside the loss only
- Added `--velocity_loss_weight` parameter (default: 1.0) to balance losses
- Scaling is applied only during loss computation; saved predictions remain in physical units

**Location**: `train_egnn.py`, lines 19-20, 448-455

### 4. Separate Loss Tracking

**Added**: Separate tracking of velocity and stress losses for better monitoring.

**Location**: `train_egnn.py`, lines 398-400, 421-423, 458-460

### 5. Prediction Saving Functionality

**Added**: `save_predictions()` function to save predictions and ground truth during training.

**Features**:
- Saves predictions for each epoch
- Saves both velocity and stress predictions
- Saves ground truth values for comparison
- Configurable number of trajectories

**Location**: `train_egnn.py`, lines 185-275

### 6. Quick Test Script

**Created**: `test_egnn_quick.py` for rapid testing on small dataset.

**Purpose**: Verify model works correctly before full training.

**Usage**:
```bash
python test_egnn_quick.py \
  --data_dir data/deforming_plate \
  --num_trajectories 3 \
  --num_epochs 5
```

### 7. Comparison and Visualization Script

**Created**: `compare_predictions.py` for analyzing results.

**Features**:
- Detailed statistics per trajectory
- Component-wise error analysis
- Visualization plots (velocity and stress over time)
- Error plots

---

## Running the Model

### Quick Test (1 trajectory, 5 epochs)

```bash
cd egnn-pytorch
python train_egnn.py \
  --data_dir ../data/deforming_plate \
  --dataset_fraction 0.001 \
  --num_epochs 5 \
  --save_predictions \
  --num_trajectories_for_predictions 1
```

**Purpose**: Verify model works correctly before full training.

**Expected results** (from test run):
- Epoch 1: Loss ~156M (Vel: ~82M, Stress: ~74M)
- Epoch 2: Loss ~29M (Vel: ~29M, Stress: ~698K)
- Epoch 3: Loss ~7M (Vel: ~6M, Stress: ~1M)
- Epoch 4: Loss ~3.5M (Vel: ~2.3M, Stress: ~1.2M)
- Epoch 5: Loss ~1.4M (Vel: ~832K, Stress: ~563K)

Losses decrease significantly, showing the model is learning.

### Training on 10% of Dataset (Recommended for Testing)

```bash
cd egnn-pytorch
python train_egnn.py \
  --data_dir ../data/deforming_plate \
  --checkpoint_dir ../checkpoints/egnn \
  --dataset_fraction 0.1 \
  --num_epochs 10 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --hidden_dim 128 \
  --depth 4 \
  --velocity_loss_weight 1.0 \
  --save_predictions \
  --save_predictions_every 2 \
  --num_trajectories_for_predictions 10 \
  --device cpu
```

**Expected time**: ~8-10 hours on CPU (100 trajectories)

**Note**: `--save_predictions_every 2` saves predictions every 2 epochs, allowing you to monitor progress without waiting for completion.

### Training on Full Dataset

```bash
cd egnn-pytorch
python train_egnn.py \
  --data_dir ../data/deforming_plate \
  --checkpoint_dir ../checkpoints/egnn \
  --dataset_fraction 1.0 \
  --num_epochs 50 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --hidden_dim 128 \
  --depth 4 \
  --velocity_loss_weight 1.0 \
  --save_predictions \
  --save_predictions_every 5 \
  --num_trajectories_for_predictions 10 \
  --device cpu
```

**Expected time**: ~400-500 hours on CPU (1000 trajectories, 50 epochs)

**For Colab (CPU or GPU)**:
```python
# In Colab notebook cell
!cd "/content/drive/MyDrive/ML_PROJECT2/egnn-pytorch" && python train_egnn.py \
  --data_dir "/content/drive/MyDrive/ML_PROJECT2/data/deforming_plate" \
  --checkpoint_dir "/content/drive/MyDrive/ML_PROJECT2/checkpoints/egnn" \
  --dataset_fraction 0.01 \
  --num_epochs 10 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --hidden_dim 128 \
  --depth 4 \
  --velocity_loss_weight 1.0 \
  --save_predictions \
  --save_predictions_every 2 \
  --num_trajectories_for_predictions 10 \
  --device cuda  # or cpu if no GPU
```

**Note**: `--dataset_fraction 0.01` gives ~10 trajectories from a 1000-trajectory dataset.

### Using Shell Script

Alternatively, use the provided shell script:

```bash
./scripts/train_egnn.sh 0.1 10
```

This runs training with:
- Dataset fraction: 0.1 (10%)
- Number of epochs: 10

---

## File Structure

```
egnn-pytorch/
├── train_egnn.py              # Main training script
├── evaluate_egnn.py            # Evaluation script
├── test_egnn_quick.py         # Quick test script
├── compare_predictions.py      # Results comparison script
├── data_loader_egnn.py        # Data loading utilities
├── myEGNN/
│   └── EGNN.py                 # EGNN model implementation
└── EGNN_TRAINING_GUIDE.md     # This file
```

---

## Output Files

### Checkpoints

Saved in `checkpoints/egnn/`:
- `best_model.pt`: Best model based on validation loss
- `checkpoint_epoch_N.pt`: Periodic checkpoints (every 5 epochs)

### Predictions

Saved in `checkpoints/egnn/predictions/epoch_N/`:
- `predictions_velocity.npy`: Model velocity predictions
- `predictions_stress.npy`: Model stress predictions
- `predictions_positions.npy`: Model coordinate predictions
- `true_velocity.npy`: Ground truth velocities
- `true_stress.npy`: Ground truth stress values
- `true_positions.npy`: Ground truth positions

### Test Results

Saved in `test_results/egnn_quick_test/`:
- `summary.json`: Overall metrics
- `trajectory_N/`: Per-trajectory results and plots

---

## Troubleshooting

### Issue: Velocity predictions are poor

**Solution**: Adjust `--velocity_loss_weight`:
- Increase slightly if velocity loss is too small (try 2.0 or 5.0)
- Decrease if stress loss stalls (try 0.5)

### Issue: Out of memory

**Solution**: 
- Reduce batch size (already 1, cannot reduce further)
- Reduce number of trajectories: `--num_trajectories_for_predictions 5`
- Use smaller hidden dimension: `--hidden_dim 64`

### Issue: Training is very slow

**Solution**:
- Use GPU if available: `--device cuda`
- Reduce dataset fraction: `--dataset_fraction 0.1`
- Reduce number of epochs for testing

---

## Summary of Changes

1. **Installed dependencies**: tfrecord, einops, egnn-pytorch
2. **Velocity prediction formula** (EGNN paper): Implemented `v_i^{l+1} = phi_v(h_i^l) * v_i^{init} + C * sum_{j != i} (x_i^l - x_j^l) * phi_x(m_ij)`
   - φ_v MLP: Outputs scalar `gamma` to modulate initial velocity
   - φ_e MLP: Computes messages `m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)`
   - φ_x MLP: Outputs scalar weight `phi_x(m_ij)` for neighbor interactions
   - Model extracts `v_i^{init}` from input features (indices 1:4)
3. **Coordinate prediction formula** (EGNN paper): `x_i^{l+1} = x_i^l + v_i^{l+1}`
   - Coordinates computed from predicted velocity, not from EGNN's internal updates
4. **Stress head**: `nn.Linear(hidden_dim, 1)` predicts stress from node embeddings
5. **Velocity scaling**: `VELOCITY_SCALE = 1e4` applied inside loss only for gradient stability
6. **Weighted loss**: `--velocity_loss_weight` (default: 1.0) balances velocity and stress objectives
7. **Added loss tracking**: Separate velocity and stress loss monitoring
8. **Added prediction saving**: Save predictions during training with configurable frequency
9. **Position predictions**: Also save predicted positions (`pred_coors = coors + pred_vel`) for analysis
10. **Created test script**: Quick verification on small dataset
11. **Created comparison script**: Analyze and visualize results with plots
12. **Documentation**: This comprehensive guide

---

## Monitoring Training Progress

### During Training

While training is running, you can monitor progress by:

1. **Check intermediate predictions** (if `--save_predictions` is enabled):
   ```bash
   python compare_predictions.py \
     --data_dir checkpoints/egnn/predictions/epoch_10 \
     --plot
   ```

2. **Evaluate a checkpoint mid-training**:
   ```bash
   python evaluate_egnn.py \
     --checkpoint checkpoints/egnn/checkpoint_epoch_20.pt \
     --data_dir data/deforming_plate \
     --output_dir eval_epoch_20 \
     --num_trajectories 10
   ```

3. **View loss progression**: Check the terminal output for velocity and stress losses decreasing over epochs.

### Expected Performance

With proper training (10+ trajectories, 10+ epochs):
- **Velocity MAE**: Should drop to < 5.0
- **Stress MAE**: Should drop to < 100 (ideally < 50)
- **Position MAE**: Should remain < 0.01 (already excellent)

If stress MAE remains high (> 500) after 10+ epochs, consider:
- Training for more epochs
- Using more trajectories
- Adjusting `--velocity_loss_weight` (try 0.5 or 2.0)

## Next Steps

1. Run quick test (3 trajectories, 5 epochs) to verify setup
2. Train on 10 trajectories × 10 epochs with intermediate saves
3. Monitor progress using saved predictions every 2-5 epochs
4. Analyze results using comparison script
5. Adjust hyperparameters if needed
6. Run full training on complete dataset (50+ epochs)

---

**Last Updated**: 2025-11-29
**Model**: EGNN (E(n) Equivariant Graph Neural Network) with velocity/stress prediction
**Architecture**: Follows EGNN paper formulas (arXiv:2102.09844v3)
- Velocity: `v_i^{l+1} = phi_v(h_i^l) * v_i^{init} + C * sum_{j != i} (x_i^l - x_j^l) * phi_x(m_ij)`
- Coordinates: `x_i^{l+1} = x_i^l + v_i^{l+1}`
- Messages: `m_ij = phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2)`
**Dataset**: Deforming Plate (MeshGraphNets)
