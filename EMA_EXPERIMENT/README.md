# EGNN Models for Deforming Plate Simulation

This repository contains the implementation of E(n) Equivariant Graph Neural Networks (EGNN) for learning physics simulations of deforming plates. The codebase supports end-to-end training, evaluation, and visualization of graph-based neural networks on trajectory data from TFRecord format.

## Project Overview

This project implements EGNN models to predict velocity and stress fields for deforming plate simulations. The models operate on graph-structured data where nodes represent mesh points and edges represent spatial connectivity. The training pipeline includes data preprocessing from TFRecords, model training with configurable hyperparameters, and comprehensive evaluation with visualization of predictions, residuals, and training metrics.

## Repository Structure

```
EMA_EXPERIMENT/
├── README.md                          # This file
├── requirements_egnn.txt              # Python dependencies
├── run_egnn.sbatch                    # Example SLURM script for SCITAS
├── processed_data/                    # Preprocessed trajectory data
│   └── data_world=radius_mesh=True_norm=standard_At=False/
│       ├── preprocessed_train.pt     # Preprocessed trajectories
│       └── used_dataconfig.yaml      # Data config used for preprocessing
├── model_out/                         # Training outputs
│   └── <dataset_name>/
│       ├── model.pt                  # Trained model checkpoint
│       └── plots/                    # Evaluation plots
│           ├── loss_metric_vs_epoch.png
│           ├── grad_norm_vs_epoch.png
│           ├── stress_vel_losses.png
│           ├── pred_vs_true/         # Prediction vs true plots (normalized)
│           └── residuals/            # Residual plots (denormalized)
├── pytorch_model/
│   ├── train.py                      # Main training entrypoint
│   ├── config.yaml                   # Training configuration
│   ├── dataconfig.yaml               # Data preprocessing configuration
│   ├── main_data.py                  # Data preprocessing script
│   ├── data_builder.py               # TFRecord loading and processing
│   ├── data_helper/
│   │   ├── defplate_dataset.py       # PyTorch Dataset class
│   │   ├── decode_tfrecord_utils.py  # TFRecord decoding utilities
│   │   └── add_world_edges.py        # World edge construction
│   ├── model_egnn/
│   │   ├── egnn_deforming_plate.py   # EGNN model wrapper
│   │   └── egnn_pytorch.py           # Core EGNN implementation
│   ├── helpers/
│   │   ├── helpers.py                # Utility functions
│   │   ├── evaluation_helper.py      # Evaluation and plotting orchestration
│   │   └── plots.py                  # Plotting functions
│   └── logs/                         # SLURM job logs (.out, .err)
└── raw_data/                         # Raw TFRecord data (not in repo)
    ├── train.tfrecord
    └── meta.json
```

## Requirements

- **Python**: 3.10+ (tested with 3.10.4 on SCITAS)
- **GPU**: Optional but recommended (CUDA 12.1+ for SCITAS, or CUDA 11.8+ locally)
- **Main packages**:
  - PyTorch >= 2.0.0 (with CUDA support for GPU training)
  - NumPy >= 1.21.0
  - PyYAML >= 6.0
  - tqdm >= 4.65.0
  - matplotlib >= 3.5.0
  - einops >= 0.6.0
  - tfrecord >= 1.14.0 (for reading TFRecord files)
  - plotly >= 5.0.0 (optional, for visualization)

## Environment Setup

### Local venv Setup

1. **Create and activate virtual environment**:
   ```bash
   cd EMA_EXPERIMENT
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements_egnn.txt
   ```

   **Note**: If using GPU locally, install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
   ```

### SCITAS/SLURM Setup

1. **Load required modules**:
   ```bash
   module load gcc/11.3.0
   module load python/3.10.4
   module load cuda/12.1.1
   ```

2. **Activate virtual environment** (adjust path to your venv):
   ```bash
   source /path/to/your/venv/bin/activate
   # Example: source /home/durante/Graph-U-Nets/venvs/gnn/bin/activate
   ```

3. **Verify CUDA**:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Logs location**: SLURM logs are saved to `pytorch_model/logs/` by default (see `run_egnn.sbatch` for `#SBATCH --output` and `--error` paths).

## Data

### Raw Data Location

Raw data is expected in `raw_data/` directory:
- `raw_data/train.tfrecord`: TFRecord file containing trajectory data
- `raw_data/meta.json`: Metadata file with dataset information

**Note**: Raw data files are not included in the repository. Place your TFRecord files in `raw_data/` before preprocessing.

### Preprocessing Step

Preprocessing converts TFRecords into PyTorch tensors with normalization and adjacency matrix construction.

**Command**:
```bash
cd pytorch_model
python main_data.py
```

This script:
1. Reads `dataconfig.yaml` to determine preprocessing parameters
2. Loads trajectories from `raw_data/train.tfrecord`
3. Normalizes features (positions, velocities, stress)
4. Builds adjacency matrices (mesh connectivity + optional world edges)
5. Saves preprocessed data to `processed_data/<dataset_name>/preprocessed_train.pt`
6. Saves `used_dataconfig.yaml` in the output directory for reproducibility

**Configuration** (`pytorch_model/dataconfig.yaml`):
- `include_mesh_pos`: Whether to include mesh positions in features (True/False)
- `normalization_method`: `"standard"` (mean/std) or `"row"` normalization
- `max_trajs`: Maximum number of trajectories to process (None = all)
- `tfrecord_path`: Path to TFRecord file (relative to `pytorch_model/`)
- `output_dir`: Output directory name (auto-generated from parameters)
- `add_world_edges`: `"radius"`, `"neighbours"`, or `None` for additional edges
- `radius_world_edge`: Radius threshold for world edges (if `add_world_edges="radius"`)
- `k_neighb`: Number of nearest neighbors (if `add_world_edges="neighbours"`)
- `a_time_var`: Whether adjacency is time-varying (True/False)

**Example output directory**: `processed_data/data_world=radius_mesh=True_norm=standard_At=False/`

### Number of Trajectories

The `num_train_trajs` parameter in `config.yaml` controls how many trajectories are loaded for training:
- If `num_train_trajs=1`, only the first trajectory is used
- If `num_train_trajs=5`, the first 5 trajectories are used
- If `num_train_trajs=None` or exceeds available trajectories, all are used

**Note**: Trajectories are loaded in sequential order from the preprocessed file (traj_id 0, 1, 2, ...).

### Full Timesteps vs Overfit Mode

- **Full timesteps mode** (`mode: null` or `mode: "standard"` in `config.yaml`):
  - Uses all time steps from selected trajectories
  - Creates train/test split (80/20) across all (traj_id, time_idx) pairs
  - Suitable for general training

- **Overfit mode** (`mode: "overfit"` in `config.yaml`):
  - Trains on specific trajectory and time indices
  - Configured via `overfit_traj_id` and `overfit_time_idx` (list of time indices)
  - Example: `overfit_traj_id: 0`, `overfit_time_idx: [0,1,2,3]` trains only on trajectory 0, timesteps 0→1, 1→2, 2→3, 3→4
  - Useful for debugging and ensuring model can fit small subsets

## Training

### Configuration (config.yaml)

Key parameters in `pytorch_model/config.yaml`:

**Model hyperparameters** (`model:`):
- `activation_gnn`: Activation for GNN layers (`"ELU"`, `"ReLU"`, etc.)
- `activation_mlps_final`: Activation for final MLPs
- `hid_gnn_layer_dim`: Hidden dimension for GNN layers (e.g., 128)
- `hid_mlp_dim`: Hidden dimension for final MLPs (e.g., 256)
- `k_pool_ratios`: Pooling ratios for graph U-Net (list, e.g., `[0.95, 0.95, 0.95]`)
- `dropout_gnn`: Dropout rate for GNN layers (0.0 = no dropout)
- `dropout_mlps_final`: Dropout rate for final MLPs
- `adj_norm`: Adjacency normalization (`"row"` or `"sym"`)

**Training parameters** (`training:`):
- `lr`: Learning rate (e.g., 0.001)
- `epochs`: Number of training epochs
- `batch_size`: Batch size (e.g., 4; reduce if OOM)
- `shuffle`: Whether to shuffle training data (False for overfit mode)
- `adam_weight_decay`: L2 regularization weight (0.0 = no weight decay)
- `num_train_trajs`: Number of trajectories to use (see Data section)
- `mode`: `null` (standard) or `"overfit"` (see Data section)
- `gamma_lr_scheduler`: Exponential LR decay factor (e.g., 0.999)
- `random_seed`: Random seed for reproducibility
- `overfit_traj_id`: Trajectory ID for overfit mode
- `overfit_time_idx`: List of time indices for overfit mode
- `datapath`: Path to preprocessed data (relative to `EMA_EXPERIMENT/`)
- `model_path_out`: Output directory for models/plots (relative to `EMA_EXPERIMENT/`)
- `amp`: Enable Automatic Mixed Precision (True/False)
- `move_all_to_device`: Move all data to GPU at once (True/False; requires sufficient GPU memory)

### Local Training

**Basic command**:
```bash
cd pytorch_model
python train.py
```

**With custom config**:
```bash
python train.py --config /path/to/custom_config.yaml
```

**With CUDA**:
```bash
python train.py --cuda
```

**With custom number of workers**:
```bash
python train.py --num-workers 8 --pin-memory
```

**Full example**:
```bash
python train.py --config config.yaml --cuda --num-workers 4 --pin-memory
```

### SCITAS/SLURM Training

**Example SLURM script** (`run_egnn.sbatch`):
```bash
#!/bin/bash
#SBATCH --job-name=egnn_train
#SBATCH --output=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/%x_%j.out
#SBATCH --error=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/%x_%j.err
#SBATCH --time=08:30:00
#SBATCH --partition=gpu-xl
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18

module load gcc/11.3.0
module load python/3.10.4
module load cuda/12.1.1
source /path/to/your/venv/bin/activate

echo "Running on node: $(hostname)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

cd /absolute/path/to/EMA_EXPERIMENT/pytorch_model
python train.py --config config.yaml --cuda

echo "End: $(date)"
```

**Submit job**:
```bash
sbatch run_egnn.sbatch
```

**Monitor job**:
```bash
squeue -u $USER
```

**View logs** (replace JOBID with actual job ID):
```bash
tail -f pytorch_model/logs/egnn_train_JOBID.out
tail -f pytorch_model/logs/egnn_train_JOBID.err
```

### Monitoring Logs

- **`.out` files**: Standard output (training progress, tqdm bars, epoch summaries)
- **`.err` files**: Standard error (warnings, errors, CUDA errors)

**Note**: tqdm progress bars may appear in `.err` on some systems. Check both files if output seems incomplete.

**Example log output**:
```
[Train] [Epoch 000] Train Loss: 0.123456 | Test Loss: 0.234567 | Vel Loss: 0.111111 | Stress Loss: 0.012345 | LR: 0.001000
```

## Evaluation + Plots

### Final Evaluation

Evaluation runs automatically after training completes (called in `train.py` main block). It:
1. Runs model on test set (or overfit set)
2. Collects predictions and targets (normalized and denormalized)
3. Generates plots in `model_out/<dataset_name>/plots/`

**Manual evaluation** (if needed):
```python
from helpers.evaluation_helper import run_final_evaluation
# ... load model, test_loader, history, etc. ...
run_final_evaluation(model, test_loader, device, history, velocity_idxs, stress_idxs, plots_dir, config_path)
```

### Plot Outputs

Plots are saved to `model_out/<dataset_name>/plots/`:

- **`loss_metric_vs_epoch.png`**: Training and validation loss curves
- **`grad_norm_vs_epoch.png`**: Gradient norm over epochs (for debugging)
- **`stress_vel_losses.png`**: Separate velocity and stress loss components
- **`pred_vs_true/feat_*.png`**: Scatter plots of predictions vs true values (normalized) for each feature:
  - `feat_0.png`: Velocity X
  - `feat_1.png`: Velocity Y
  - `feat_2.png`: Velocity Z
  - `feat_3.png`: Stress
- **`residuals/feat_*.png`**: Residual plots (denormalized) showing prediction errors
- **`residuals_norm/feat_*.png`**: Residual plots (normalized)

### Copying Plots from SCITAS

**Using scp** (from local machine):
```bash
scp username@scitas.epfl.ch:/absolute/path/to/EMA_EXPERIMENT/model_out/<dataset_name>/plots/*.png ./local_plots/
```

**Using rsync** (recommended for directories):
```bash
rsync -avz username@scitas.epfl.ch:/absolute/path/to/EMA_EXPERIMENT/model_out/<dataset_name>/plots/ ./local_plots/
```

**Example**:
```bash
rsync -avz durante@scitas.epfl.ch:/home/durante/EMA_EXPERIMENT/model_out/data_world=radius_mesh=True_norm=standard_At=False/plots/ ./plots_from_scitas/
```

## Common Issues / Troubleshooting

### CUDA Not Available

**Error**: `RuntimeError: CUDA not available`

**Solutions**:
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- On SCITAS: Ensure `module load cuda/12.1.1` and `CUDA_VISIBLE_DEVICES` is set

### Torch Not Installed (venv not activated)

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Activate virtual environment:
```bash
source venv/bin/activate  # or your venv path
```

### YAML Type Errors

**Error**: `TypeError: unsupported operand type(s) for *: 'float' and 'str'` (e.g., `weight_decay` parsed as string)

**Solution**: Ensure numeric values in `config.yaml` are not quoted:
```yaml
# WRONG:
adam_weight_decay: "0.0001"

# CORRECT:
adam_weight_decay: 0.0001
```

### AMP dtype Mismatch Errors in index_add_

**Error**: `RuntimeError: Expected all tensors to be on the same device, and of the same dtype, but found at least two devices/cuda:0!` or dtype mismatch in `index_add_`

**Solution**: This occurs when AMP mixes float16/float32. Fix pattern: cast source tensor to destination dtype before `index_add_`:
```python
# In model code (e.g., egnn_pytorch.py):
# BEFORE:
out = out.index_add_(0, indices, values)

# AFTER:
out = out.index_add_(0, indices, values.to(out.dtype))
```

If the error persists, disable AMP temporarily: set `amp: False` in `config.yaml`.

### GPU OOM (Out of Memory)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. **Reduce batch size**: Change `batch_size: 4` to `batch_size: 2` or `batch_size: 1` in `config.yaml`
2. **Enable AMP**: Set `amp: True` in `config.yaml` (reduces memory by ~50%)
3. **Disable move_all_to_device**: Set `move_all_to_device: False` (loads data on-demand)
4. **Reduce number of trajectories**: Set `num_train_trajs: 1` to use fewer trajectories
5. **Reduce model size**: Decrease `hid_gnn_layer_dim` and `hid_mlp_dim` in `config.yaml`

**Recommended starting point for OOM**: `batch_size: 2`, `amp: True`, `move_all_to_device: False`

### Logs Saved in Wrong Directory

**Issue**: Logs appear in unexpected location

**Solution**: Use absolute paths in `#SBATCH --output` and `--error` directives:
```bash
#SBATCH --output=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/%x_%j.out
#SBATCH --error=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/%x_%j.err
```

Relative paths are resolved from the directory where `sbatch` is called, which may differ from the script's working directory.

## Reproducing Experiments

### Reproducing a Specific Run

1. **Use the same config**: Save your `config.yaml` with a descriptive name (e.g., `config_exp1.yaml`)
2. **Set random seed**: Ensure `random_seed` in `config.yaml` matches the original run
3. **Use the same preprocessed data**: Point `datapath` to the same preprocessed directory
4. **Run with the saved config**:
   ```bash
   python train.py --config config_exp1.yaml
   ```

**Note**: Reproducibility also depends on:
- PyTorch version (use same version)
- CUDA version (may affect floating-point operations)
- Data preprocessing (use same `dataconfig.yaml`)

### Avoiding Overwriting Outputs

**Problem**: Multiple runs overwrite `model_out/<dataset_name>/model.pt` and plots.

**Solution**: Use unique output directory names. Modify `config.yaml`:
```yaml
training:
  model_path_out: "model_out/exp1_seed42_bs4/"  # Unique name per experiment
```

Or use job ID in SLURM (modify `setup_paths` in `helpers/helpers.py` or use environment variable):
```python
# In train.py or helpers.py:
import os
job_id = os.environ.get('SLURM_JOB_ID', 'local')
output_dir = f"model_out/exp1_job{job_id}/"
```

**Recommended**: Create a config per experiment:
- `configs/config_exp1.yaml`
- `configs/config_exp2.yaml`
- Each with unique `model_path_out`

## Ablation Study Guide

### Running Multiple Configs Concurrently

**Option 1: Use `--config` argument** (recommended):
```bash
# Create configs for different experiments
cp config.yaml configs/config_baseline.yaml
cp config.yaml configs/config_no_dropout.yaml
cp config.yaml configs/config_small_model.yaml

# Edit each config with different hyperparameters

# Run sequentially (or use SLURM array job):
python train.py --config configs/config_baseline.yaml
python train.py --config configs/config_no_dropout.yaml
python train.py --config configs/config_small_model.yaml
```

**Option 2: SLURM Array Job** (run multiple configs in parallel):

Create `run_ablation.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=egnn_ablation
#SBATCH --output=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/ablation_%A_%a.out
#SBATCH --error=/absolute/path/to/EMA_EXPERIMENT/pytorch_model/logs/ablation_%A_%a.err
#SBATCH --time=08:30:00
#SBATCH --partition=gpu-xl
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --array=0-3  # Run 4 configs (indices 0, 1, 2, 3)

module load gcc/11.3.0
module load python/3.10.4
module load cuda/12.1.1
source /path/to/your/venv/bin/activate

# Array of config files
CONFIGS=(
    "configs/config_baseline.yaml"
    "configs/config_no_dropout.yaml"
    "configs/config_small_model.yaml"
    "configs/config_large_model.yaml"
)

# Get config for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running config: $CONFIG (array task $SLURM_ARRAY_TASK_ID)"
echo "Start: $(date)"

cd /absolute/path/to/EMA_EXPERIMENT/pytorch_model
python train.py --config "$CONFIG" --cuda

echo "End: $(date)"
```

**Submit array job**:
```bash
sbatch run_ablation.sbatch
```

**Monitor all tasks**:
```bash
squeue -u $USER
```

**View specific task log**:
```bash
tail -f pytorch_model/logs/ablation_JOBID_0.out  # Task 0
tail -f pytorch_model/logs/ablation_JOBID_1.out  # Task 1
```

### Example Ablation Configs

Create `configs/` directory and example configs:

**`configs/config_baseline.yaml`**:
```yaml
model:
  hid_gnn_layer_dim: 128
  dropout_gnn: 0.1
training:
  lr: 0.001
  batch_size: 4
  model_path_out: "model_out/baseline/"
```

**`configs/config_no_dropout.yaml`**:
```yaml
model:
  hid_gnn_layer_dim: 128
  dropout_gnn: 0.0  # No dropout
training:
  lr: 0.001
  batch_size: 4
  model_path_out: "model_out/no_dropout/"
```

**`configs/config_small_model.yaml`**:
```yaml
model:
  hid_gnn_layer_dim: 64  # Smaller
  dropout_gnn: 0.1
training:
  lr: 0.001
  batch_size: 4
  model_path_out: "model_out/small_model/"
```

## Citation / Credits

### EGNN Paper

This implementation is based on the E(n) Equivariant Graph Neural Networks paper:

**Satorras, V. G., Hoogeboom, E., & Welling, M. (2021).**  
*E(n) Equivariant Graph Neural Networks.*  
ICML 2021.  
[arXiv:2102.09844](https://arxiv.org/abs/2102.09844)

### External Libraries

- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **einops**: [https://github.com/arogozhnikov/einops](https://github.com/arogozhnikov/einops)
- **tfrecord**: For reading TensorFlow TFRecord format

### Dataset

The deforming plate simulation dataset is provided as TFRecord files. Contact the project maintainers for dataset access.

---

**Last updated**: Fall 2025  
**Project**: EPFL CS-433 Project 2
