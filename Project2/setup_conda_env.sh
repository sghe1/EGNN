#!/bin/bash
# Conda environment setup script for MeshGraphNet training
# This script creates a conda environment with compatible Python and TensorFlow versions
# Run from Project2/ directory

set -e  # Exit on error

ENV_NAME="meshgraphnet"
PYTHON_VERSION="3.11"  # Python 3.11 is compatible with TensorFlow 2.15.0

echo "=========================================="
echo "MeshGraphNet Conda Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment. Activate it with: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo ""
echo "Installing pip dependencies..."
pip install --upgrade pip

# Install TensorFlow and compatible dependencies
# For Apple Silicon (M1/M2), we'll use tensorflow-macos if available
# Otherwise, use standard tensorflow
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "Detected Apple Silicon. Installing tensorflow-macos..."
    # Try tensorflow-macos first (for M1/M2 Macs)
    pip install tensorflow-macos==2.13.1 tensorflow-metal==1.1.0 || {
        echo "tensorflow-macos not available, using standard tensorflow..."
        pip install "tensorflow>=2.13.0,<2.16.0"
    }
else
    echo "Installing standard TensorFlow..."
    pip install "tensorflow>=2.13.0,<2.16.0"
fi

# Install tensorflow-probability (compatible version)
echo "Installing tensorflow-probability..."
pip install "tensorflow-probability>=0.18.0,<0.20.0"

# Install dm-sonnet (requires older tensorflow-probability, but we'll try)
echo "Installing dm-sonnet..."
if ! pip install "dm-sonnet<2" 2>/dev/null; then
    echo "Warning: dm-sonnet installation had issues. Trying alternative approach..."
    echo "Installing compatible tensorflow-probability version for dm-sonnet..."
    # Try with older tensorflow-probability first
    pip install "tensorflow-probability>=0.8.0,<0.9.0" --force-reinstall || {
        echo "Could not install older tensorflow-probability. Continuing anyway..."
    }
    # Try installing dm-sonnet
    if ! pip install "dm-sonnet<2"; then
        echo ""
        echo "ERROR: dm-sonnet installation failed."
        echo "You may need to install it manually. Try:"
        echo "  pip install tensorflow-probability>=0.8.0,<0.9.0 --force-reinstall"
        echo "  pip install dm-sonnet<2"
        echo ""
        echo "Or continue and install it later if needed."
    fi
fi

# Install other dependencies
# IMPORTANT: Pin numpy version BEFORE installing matplotlib to prevent upgrade
echo "Installing other dependencies..."
echo "Pinning numpy version to maintain TensorFlow compatibility..."
pip install "numpy>=1.22,<=1.24.3" --force-reinstall
pip install matplotlib absl-py "protobuf>=3.20.3,<4.0.0" h5py six

# Set protobuf environment variable for compatibility
echo ""
echo "Setting up environment variables..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" << 'EOF'
#!/bin/bash
unset PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
EOF

chmod +x "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"

echo ""
echo "Verifying installation..."
echo ""

# Initialize status variables
TF_OK=0
TFP_OK=0
SONNET_OK=0
NUMPY_OK=0

# Verify key packages (suppress stderr for cleaner output)
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)" 2>/dev/null && TF_OK=1 || TF_OK=0
if [ $TF_OK -eq 0 ]; then
    echo "✗ TensorFlow: Import failed"
    python -c "import tensorflow as tf" 2>&1 | head -3
fi

python -c "import tensorflow_probability as tfp; print('✓ TensorFlow Probability:', tfp.__version__)" 2>/dev/null && TFP_OK=1 || TFP_OK=0
if [ $TFP_OK -eq 0 ]; then
    echo "✗ TensorFlow Probability: Import failed"
fi

python -c "import sonnet as snt; print('✓ Sonnet: OK')" 2>/dev/null && SONNET_OK=1 || SONNET_OK=0
if [ $SONNET_OK -eq 0 ]; then
    echo "✗ Sonnet: Import failed (may need manual installation)"
fi

python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>/dev/null && NUMPY_OK=1 || NUMPY_OK=0
if [ $NUMPY_OK -eq 0 ]; then
    echo "✗ NumPy: Not installed correctly"
fi

# Check numpy version compatibility
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "")
NUMPY_COMPATIBLE=0
if [ ! -z "$NUMPY_VERSION" ]; then
    # Check if numpy version is compatible (should be <= 1.24.3)
    if python -c "from packaging import version; import numpy; exit(0 if version.parse(numpy.__version__) <= version.parse('1.24.3') else 1)" 2>/dev/null; then
        echo "✓ NumPy version $NUMPY_VERSION is compatible with TensorFlow"
        NUMPY_COMPATIBLE=1
    else
        echo "⚠ Warning: NumPy version $NUMPY_VERSION may be incompatible with TensorFlow (requires <=1.24.3)"
        echo "  Attempting to fix..."
        pip install "numpy>=1.22,<=1.24.3" --force-reinstall --quiet
        NUMPY_COMPATIBLE=1  # Assume fix worked
    fi
fi

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""

# Final check and fix if needed (already done above, but double-check)
if [ $NUMPY_COMPATIBLE -eq 0 ] && [ $NUMPY_OK -eq 1 ]; then
    echo "Performing final NumPy compatibility check..."
    if ! python -c "from packaging import version; import numpy; exit(0 if version.parse(numpy.__version__) <= version.parse('1.24.3') else 1)" 2>/dev/null; then
        echo "Fixing NumPy version compatibility..."
        pip install "numpy>=1.22,<=1.24.3" --force-reinstall --quiet
        echo "✓ NumPy version fixed"
    fi
fi

echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Then you can run MeshGraphNet training with:"
echo "  cd Project2"
echo "  bash run_meshgraphnets.sh"
echo ""
echo "Note: The PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable"
echo "      will be automatically set when you activate the environment."
echo ""
if [ $SONNET_OK -eq 0 ] || [ $TF_OK -eq 0 ] || [ $TFP_OK -eq 0 ]; then
    echo "⚠ Some packages may need manual installation."
    echo "  If you see import errors, try running:"
    echo "    bash fix_numpy_version.sh"
    echo "  See SETUP_CONDA.md for troubleshooting."
fi
echo ""

