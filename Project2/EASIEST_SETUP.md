# Easiest Way to Run MeshGraphNet for Trajectory 0 (500 epochs)

## The Problem
There are compatibility issues between TensorFlow 2.13.1, tensorflow-probability 0.8.0, dm-sonnet, and Python 3.11.

## Easiest Solution: Use Python 3.10

Python 3.10 has better compatibility with the older packages required by MeshGraphNet.

### Quick Setup (Recommended)

```bash
cd Project2

# Create new environment with Python 3.10
conda create -n meshgraphnet310 python=3.10 -y
conda activate meshgraphnet310

# Install dependencies in correct order
pip install --upgrade pip
pip install "numpy>=1.22,<=1.24.3"
pip install tensorflow-macos==2.13.1 tensorflow-metal==1.1.0
pip install "tensorflow-probability>=0.8.0,<0.9.0"
pip install "dm-sonnet<2"
pip install matplotlib absl-py "protobuf>=3.20.3,<4.0.0" h5py six

# Set environment variable
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Test imports
python -c "
import os
import sys
sys.path.insert(0, 'deepmind-research')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
from meshgraphnets import dataset
from meshgraphnets import deforming_plate_model
print('✓ All imports successful!')
"

# Run training
bash run_meshgraphnets.sh
```

### Alternative: Use Existing Environment with Workarounds

If you want to stick with Python 3.11, you'll need to apply patches:

```bash
cd Project2
conda activate meshgraphnet

# Apply all fixes
bash fix_numpy_version.sh
bash complete_fix.sh

# Note: You may still encounter issues. Python 3.10 is recommended.
```

## Running Training

Once the environment is set up:

```bash
cd Project2
conda activate meshgraphnet310  # or meshgraphnet if using 3.11
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
bash run_meshgraphnets.sh
```

This will:
- Train on trajectory 0
- Run for 500 epochs
- Generate plots for velocity/stress errors and predictions
- Save checkpoints to `checkpoints/meshgraphnet/`
- Save plots to `plots/meshgraphnet/`

## What Gets Generated

After training completes, you'll find:

1. **Training plots:**
   - `training_convergence.png` - Loss curves
   - `loss_velocity.png` - Velocity loss over epochs
   - `velocity_error.png` - Velocity error (MSE) over epochs
   - `stress_magnitude.png` - Stress magnitude over epochs

2. **Prediction plots:**
   - `predictions_positions.png` - Position predictions over time
   - `predictions_velocities.png` - Velocity predictions over time
   - `stress_evolution.png` - Stress evolution over time
   - `parity_velocity.png` - Velocity parity plot

3. **Checkpoints:**
   - Saved in `checkpoints/meshgraphnet/`
   - Training history in `training_history.pkl`

## Troubleshooting

If you get import errors:
1. Make sure you're in the correct conda environment
2. Check Python version: `python --version` (should be 3.10 ideally)
3. Verify TensorFlow: `python -c "import tensorflow as tf; print(tf.__version__)"`
4. Set the protobuf environment variable

If training fails:
- Check that `../raw_data` contains `meta.json`, `train.tfrecord`, `valid.tfrecord`
- Verify trajectory 0 exists in the dataset
- Check disk space for checkpoints and plots

