#!/bin/bash
#SBATCH --job-name=egnn_train
#SBATCH --output=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/%x_%j.out
#SBATCH --error=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/%x_%j.err
#SBATCH --time=08:30:00
#SBATCH --partition=gpu-xl
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18

# Load required modules
module load gcc/11.3.0
module load python/3.10.4
module load cuda/12.1.1

# Activate virtual environment (UPDATE THIS PATH)
source /path/to/your/venv/bin/activate
# Example: source /home/durante/Graph-U-Nets/venvs/gnn/bin/activate

echo "Running on node: $(hostname)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Start: $(date)"

# Change to project directory (UPDATE THIS PATH)
cd /absolute/path/to/EMA_EXPERIMENT/pytorch_model

# Run training
python train.py --config config.yaml --cuda

echo "End: $(date)"
