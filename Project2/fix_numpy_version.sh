#!/bin/bash
# Quick fix script to fix numpy version in meshgraphnet conda environment
# Run this if numpy was upgraded to an incompatible version

set -e

ENV_NAME="meshgraphnet"

echo "Fixing numpy version in conda environment '${ENV_NAME}'..."
echo ""

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Check current numpy version
CURRENT_NUMPY=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
echo "Current NumPy version: $CURRENT_NUMPY"

# Check if fix is needed
if python -c "from packaging import version; import numpy; exit(0 if version.parse(numpy.__version__) <= version.parse('1.24.3') else 1)" 2>/dev/null; then
    echo "✓ NumPy version is already compatible!"
    exit 0
fi

echo "Downgrading NumPy to compatible version (<=1.24.3)..."
pip install "numpy>=1.22,<=1.24.3" --force-reinstall

echo ""
echo "Verifying fix..."
python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)" || echo "✗ TensorFlow import failed - may need to reinstall"

echo ""
echo "Fix complete! Try running your training script again."

