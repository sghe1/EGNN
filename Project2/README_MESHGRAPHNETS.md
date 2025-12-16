# MeshGraphNets Training for Deforming Plate

This directory contains the training script for MeshGraphNets baseline on the Deforming Plate dataset.

## Installation

First, install the required dependencies:

```bash
# From Project2/ directory
./install_dependencies.sh

# If you get TensorFlow/TensorFlow Probability compatibility errors:
./fix_tensorflow_versions.sh

# Or manually install compatible versions:
pip install "tensorflow>=2.8.0,<2.16.0"
pip install "tensorflow-probability>=0.16.0,<0.23.0"
pip install "dm-sonnet<2" matplotlib absl-py numpy
```

Required packages:
- `tensorflow>=2.8.0,<2.16.0` (compatible version range)
- `tensorflow-probability>=0.16.0,<0.23.0` (must match TensorFlow version)
- `dm-sonnet<2` (DeepMind Sonnet)
- `matplotlib`
- `absl-py`
- `numpy`

**Note**: If you encounter version compatibility errors (ValueError about arg specs), use `fix_tensorflow_versions.sh` to install compatible versions.

## Quick Start

From the `Project2/` directory, run:

```bash
./run_meshgraphnets.sh
```

Or directly:

```bash
python deepmind-research/meshgraphnets/train_deforming_plate.py \
  --data_dir=raw_data \
  --output_dir=checkpoints/meshgraphnet \
  --plots_dir=plots/meshgraphnet \
  --num_epochs=500 \
  --val_freq=10 \
  --steps_per_epoch=1000 \
  --learning_rate=1e-4 \
  --num_val_batches=50
```

## Loss Computation

The training script computes three separate losses:

1. **Acceleration Loss** (primary): MSE on predicted vs target acceleration
   - Computed in normalized space
   - Only on NORMAL nodes (same as original MeshGraphNets)
   - Location: `compute_losses()` function, lines 109-110

2. **Velocity Loss** (secondary): MSE on predicted vs target velocity
   - Velocity derived from positions: `velocity = next_pos - current_pos`
   - Target velocity: `target|world_pos - world_pos`
   - Predicted velocity: `predicted_next_pos - world_pos` (from acceleration)
   - Location: `compute_losses()` function, lines 91-102

3. **Stress Loss** (regularization): MSE on stress feature magnitude
   - Stress is an input feature (not predicted by model)
   - Computed on NORMAL and HANDLE nodes
   - Location: `compute_losses()` function, lines 104-106

**Total Loss**: `acceleration_loss + 0.1 * velocity_loss`

## Parity Plots

Parity plots are generated after training using validation data:

1. **Velocity Parity Plot** (`parity_velocity.png`):
   - X-axis: Ground truth velocity magnitude
   - Y-axis: Predicted velocity magnitude
   - Data: Only NORMAL nodes from validation batches
   - Denormalization: Handled automatically (velocities derived from denormalized positions)

2. **Stress Parity Plot** (`parity_stress.png`):
   - X-axis: Ground truth stress (input stress)
   - Y-axis: Predicted stress (same as input, since model doesn't predict stress)
   - Data: NORMAL and HANDLE nodes from validation batches
   - Note: Since MeshGraphNets doesn't predict stress, this shows input consistency

## Output Structure

```
Project2/
├── checkpoints/meshgraphnet/
│   ├── model.ckpt-* (checkpoints)
│   └── training_history.pkl
└── plots/meshgraphnet/
    ├── training_convergence.png
    ├── loss_velocity.png
    ├── loss_stress.png
    ├── parity_velocity.png
    └── parity_stress.png
```

## Notes

- The model predicts **acceleration**, not velocity or stress directly
- Velocity is derived from predicted positions using Verlet integration
- Stress is used as an input feature but not predicted
- All losses are computed in the same normalized space as the original MeshGraphNets
- Validation runs every `val_freq` epochs (default: 10)
