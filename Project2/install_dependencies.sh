#!/bin/bash
# Install MeshGraphNets dependencies
# Run from Project2/ directory

cd "$(dirname "$0")"

echo "Installing MeshGraphNets dependencies..."
pip install -r deepmind-research/meshgraphnets/requirements.txt

echo ""
echo "Dependencies installed! You can now run:"
echo "  ./run_meshgraphnets.sh"
echo ""
echo "Or with a specific trajectory:"
echo "  TRAJECTORY_ID=0 ./run_meshgraphnets.sh"
