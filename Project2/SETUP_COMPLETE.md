# ✅ Easiest Way to Run MeshGraphNet - COMPLETE SETUP

## Quick Start (3 Steps)

### Step 1: Setup Environment (One-time)
```bash
cd Project2
bash quick_setup_python310.sh
bash fix_python310_env.sh
```

### Step 2: Run Training
```bash
cd Project2
bash run_training.sh
```

That's it! The training will run on trajectory 0 for 500 epochs and generate all plots.

## What Gets Generated

After training completes (~500 epochs), you'll find:

### Plots (`plots/meshgraphnet/`):
- `training_convergence.png` - Training loss curves (log scale)
- `loss_velocity.png` - Velocity loss over epochs
- `velocity_error.png` - Velocity error (MSE) over epochs  
- `stress_magnitude.png` - Stress magnitude over epochs
- `predictions_positions.png` - Position predictions (X, Y, Z) over time
- `predictions_velocities.png` - Velocity predictions (X, Y, Z) over time
- `stress_evolution.png` - Stress evolution over time
- `parity_velocity.png` - Velocity parity plot

### Checkpoints (`checkpoints/meshgraphnet/`):
- Model checkpoints saved every 50 epochs
- `training_history.pkl` - Complete training history

## Manual Run (Alternative)

If you prefer to run manually:

```bash
cd Project2
conda activate meshgraphnet310
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
bash run_meshgraphnets.sh
```

## Troubleshooting

### If imports fail:
```bash
bash fix_python310_env.sh
```

### If you get protobuf errors:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### Check environment:
```bash
conda activate meshgraphnet310
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "import sonnet as snt; print('Sonnet: OK')"
```

## Training Configuration

The training is configured for:
- **Trajectory**: 0 (single trajectory)
- **Epochs**: 500
- **Steps per epoch**: 1000
- **Learning rate**: 1e-4
- **Validation**: Every 10 epochs
- **Validation batches**: 50

All plots and errors are automatically generated and saved!
