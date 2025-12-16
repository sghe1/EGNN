#!/bin/bash
# Fix TensorFlow version compatibility issues for MeshGraphNets
# Run from Project2/ directory

cd "$(dirname "$0")"

echo "Fixing TensorFlow version compatibility..."
echo "Current versions may be incompatible. Installing compatible versions..."

# Uninstall current versions
pip uninstall -y tensorflow tensorflow-probability protobuf

# Install compatible versions
# Note: protobuf 3.20.x is required for dm-sonnet compatibility
pip install "protobuf>=3.20.3,<4.0.0"
pip install "tensorflow>=2.8.0,<2.16.0"
pip install "tensorflow-probability>=0.16.0,<0.23.0"

echo ""
echo "Done! Compatible versions installed."
echo "You can now run: ./run_meshgraphnets.sh"
