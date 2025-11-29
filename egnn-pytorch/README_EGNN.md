# EGNN Training Guide

This guide explains how to train the E(n)-Equivariant Graph Neural Network (EGNN) on the deforming plate dataset.

## Prerequisites

### 1. Install PyTorch

First, install PyTorch. For CPU:
```bash
pip install torch torchvision torchaudio
```

For GPU (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Required Packages

```bash
cd egnn-pytorch
pip install einops numba numpy tqdm
pip install tfrecord  # For reading tfrecord files
```

### 3. Install egnn-pytorch Package

```bash
cd egnn-pytorch/egnn-pytorch
pip install -e .
```

Or if that doesn't work:
```bash
cd egnn-pytorch
pip install -e ./egnn-pytorch
```

## Dataset

The dataset should be in:
```
data/deforming_plate/
├── train.tfrecord
├── valid.tfrecord
├── test.tfrecord
└── meta.json
```

## Training

### Basic Training (10% of dataset, 10 epochs)

```bash
bash scripts/train_egnn.sh
```

### Custom Training

```bash
# Train with 50% of dataset for 20 epochs
bash scripts/train_egnn.sh 0.5 20

# Or run directly:
cd egnn-pytorch
python train_egnn.py \
  --data_dir=../data/deforming_plate \
  --checkpoint_dir=../checkpoints/egnn \
  --dataset_fraction=0.1 \
  --num_epochs=10 \
  --batch_size=1 \
  --learning_rate=1e-4 \
  --hidden_dim=128 \
  --depth=4
```

### Training Parameters

- `--dataset_fraction`: Fraction of dataset to use (0.0 to 1.0, default: 0.1)
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 1, usually 1 for trajectories)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--hidden_dim`: Hidden dimension for EGNN (default: 128)
- `--depth`: Depth of EGNN layers (default: 4)
- `--device`: Device to use ('cpu' or 'cuda', default: auto-detect)

## Model Architecture

The EGNN model:
- **Input features**: node_type (1) + velocity (3) + acceleration (3) + stress (1) = 8 dimensions
- **Output**: Acceleration prediction (3D)
- **Architecture**: Input MLP → EGNN layers → Output MLP

## Data Format

The data loader (`data_loader_egnn.py`) converts trajectories to:
- `coors`: (T, N, 3) - Position coordinates over time
- `feats`: (T, N, 8) - Node features over time
- `edge_index`: (2, E) - Edge connectivity (converted to adjacency matrix)

## Checkpoints

Checkpoints are saved to:
```
checkpoints/egnn/
├── best_model.pt          # Best model (lowest loss)
└── checkpoint_epoch_N.pt  # Periodic checkpoints
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Install PyTorch (see Prerequisites).

### "ModuleNotFoundError: No module named 'egnn_pytorch'"
Install the egnn-pytorch package:
```bash
cd egnn-pytorch/egnn-pytorch
pip install -e .
```

### "ModuleNotFoundError: No module named 'tfrecord'"
Install tfrecord:
```bash
pip install tfrecord
```

### Out of Memory
- Reduce batch size (already 1 by default)
- Reduce dataset fraction
- Reduce hidden_dim or depth

## Next Steps

After training, you can:
1. Evaluate the model on validation/test set
2. Generate rollouts/predictions
3. Compare with MeshGraphNet results

