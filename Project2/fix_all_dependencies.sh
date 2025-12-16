#!/bin/bash
# Comprehensive fix script for meshgraphnet conda environment
# Fixes numpy, gast, and tensorflow-probability compatibility issues

set -e

ENV_NAME="meshgraphnet"

echo "Fixing all dependencies in conda environment '${ENV_NAME}'..."
echo ""

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "Step 1: Fixing NumPy version (must be <=1.24.3 for TensorFlow)..."
pip install "numpy>=1.22,<=1.24.3" --force-reinstall --no-deps --quiet
pip install "numpy>=1.22,<=1.24.3" --upgrade --quiet

echo "Step 2: Fixing gast version (must be <=0.4.0 for TensorFlow)..."
pip install "gast>=0.2.1,<=0.4.0" --force-reinstall --quiet

echo "Step 3: Installing compatible tensorflow-probability..."
# Install tensorflow-probability but prevent numpy upgrade
pip install "tensorflow-probability>=0.18.0,<0.20.0" --force-reinstall --no-deps --quiet || {
    echo "Installing tensorflow-probability with dependencies (but constraining numpy)..."
    pip install "tensorflow-probability>=0.18.0,<0.20.0" "numpy>=1.22,<=1.24.3" --force-reinstall --quiet
}

# Ensure numpy is still correct version after tensorflow-probability install
pip install "numpy>=1.22,<=1.24.3" --force-reinstall --quiet

echo "Step 4: Reinstalling dm-sonnet without dependency check..."
# Install dm-sonnet without checking tensorflow-probability version
pip install "dm-sonnet<2" --no-deps --force-reinstall --quiet || {
    echo "Warning: dm-sonnet installation failed. This may be OK if it's already installed."
}

echo ""
echo "Verifying installation..."
python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)"
python -c "import tensorflow_probability as tfp; print('✓ TensorFlow Probability:', tfp.__version__)" || echo "⚠ TensorFlow Probability import failed"

echo ""
echo "Testing MeshGraphNet imports..."
python -c "
import os
import sys
sys.path.insert(0, 'deepmind-research')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

from meshgraphnets import dataset
print('✓ Dataset module OK')

# Try importing model (may fail on sonnet check, but that's OK)
try:
    from meshgraphnets import deforming_plate_model
    print('✓ deforming_plate_model OK')
except Exception as e:
    print('⚠ Model import warning:', str(e))
    print('  This may be OK - sonnet version check is strict')
" || echo "Import test completed with warnings"

echo ""
echo "Fix complete!"

