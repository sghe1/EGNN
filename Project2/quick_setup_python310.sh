#!/bin/bash
# Quick setup script for Python 3.10 (EASIEST solution)
# This avoids all the compatibility issues

set -e

ENV_NAME="meshgraphnet310"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Quick Setup: Python 3.10 Environment"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment..."
    conda env remove -n ${ENV_NAME} -y
fi

echo "Creating conda environment with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo ""
echo "Installing dependencies..."
pip install --upgrade pip

# Install in correct order
echo "  Installing NumPy..."
pip install "numpy>=1.22,<=1.24.3"

echo "  Installing TensorFlow..."
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    pip install tensorflow-macos==2.13.1 tensorflow-metal==1.1.0 || {
        echo "tensorflow-macos not available, using standard tensorflow..."
        pip install "tensorflow>=2.13.0,<2.16.0"
    }
else
    pip install "tensorflow>=2.13.0,<2.16.0"
fi

echo "  Installing TensorFlow Probability..."
pip install "tensorflow-probability>=0.8.0,<0.9.0"

echo "  Installing dm-sonnet..."
pip install "dm-sonnet<2"

echo "  Installing other dependencies..."
pip install matplotlib absl-py "protobuf>=3.20.3,<4.0.0" h5py six

# Set up environment variable
echo ""
echo "Setting up environment variables..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
EOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

echo ""
echo "Testing installation..."
python -c "
import os
import sys
sys.path.insert(0, 'deepmind-research')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

from meshgraphnets import dataset
print('✓ Dataset OK')

from meshgraphnets import deforming_plate_model
print('✓ deforming_plate_model OK!')
print('')
print('🎉 SUCCESS! Environment ready!')
" && {
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To run training:"
    echo "  conda activate ${ENV_NAME}"
    echo "  cd Project2"
    echo "  bash run_meshgraphnets.sh"
    echo ""
} || {
    echo ""
    echo "⚠ Some imports failed, but you can try running training anyway."
    echo "The environment is mostly set up."
}

