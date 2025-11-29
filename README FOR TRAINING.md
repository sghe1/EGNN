# MeshGraphNet Training and Evaluation Pipeline

This guide provides a complete pipeline for training and evaluating the MeshGraphNet model on the deforming plate dataset.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Training the Model](#training-the-model)
4. [Monitoring Training Progress](#monitoring-training-progress)
5. [Evaluating the Model](#evaluating-the-model)
6. [Viewing Results](#viewing-results)
7. [Configuration Options](#configuration-options)

---

## Prerequisites

### 1. Conda Environment

Ensure the `meshgraphnets` conda environment is set up and activated:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate meshgraphnets
```

### 2. Dataset

Ensure the deforming plate dataset is available in:
```
data/deforming_plate/
├── train.tfrecord
├── valid.tfrecord
├── test.tfrecord
└── meta.json
```

---

## Project Structure

```
MLproject2/
├── Project2/deepmind-research/meshgraphnets/  # Model code
├── data/deforming_plate/                      # Dataset
├── checkpoints/deforming_plate/               # Saved checkpoints
├── rollouts/                                  # Evaluation results
├── scripts/
│   ├── train_deforming_plate.sh              # Training script
│   └── eval_deforming_plate.sh               # Evaluation script
└── README.md                                  # This file
```

---

## Training the Model

### Basic Training (Default: 10,000 steps, 50% of dataset)

```bash
bash scripts/train_deforming_plate.sh
```

### Custom Training Steps

To train for a different number of steps:

```bash
bash scripts/train_deforming_plate.sh 5000  # Train for 5,000 steps
bash scripts/train_deforming_plate.sh 20000 # Train for 20,000 steps
```

### Training Configuration

The training script (`scripts/train_deforming_plate.sh`) is configured with:
- **Model**: `deforming_plate`
- **Dataset fraction**: `0.5` (50% of training data)
- **Default steps**: `10,000`
- **Normalization period**: First 100 steps (no training, just accumulating statistics)
- **Actual training**: Steps 100-10,000

### What Happens During Training

1. **Steps 0-100**: Normalization period
   - Model accumulates statistics for data normalization
   - No actual training occurs
   - Loss is not meaningful during this period

2. **Steps 100-10,000**: Actual training
   - Model learns to predict plate deformation
   - Loss decreases over time
   - Checkpoints saved every 600 seconds
   - TensorBoard summaries saved every 100 steps

### Training Output

You'll see output like:
```
I1123 13:13:49.255841 run_model.py:114] Step 0: Loss 50.0639
I1123 13:20:12.249876 run_model.py:114] Step 1000: Loss 0.147274
...
I1123 13:XX:XX.XXXXXX run_model.py:115] Training complete.
```

---

## Monitoring Training Progress

### Using TensorBoard

1. **Start TensorBoard** (in a separate terminal):

```bash
tensorboard --logdir=checkpoints/deforming_plate --port=6006
```

2. **Open in browser**:
   - Navigate to: http://localhost:6006
   - Go to the **"SCALARS"** tab

3. **View metrics**:
   - **Loss**: Training loss curve (logged every 100 steps)
   - **Learning Rate**: Learning rate schedule
   - **global_step/sec**: Training speed

### Viewing Loss from Logs

To see loss values from the training logs:

```bash
python3 view_loss.py
```

This will show:
- Loss values at different training steps
- Loss statistics (initial, final, improvement)
- Instructions for TensorBoard

---

## Evaluating the Model

### Run Evaluation

After training, evaluate the model:

```bash
bash scripts/eval_deforming_plate.sh
```

### Evaluation Configuration

The evaluation script:
- Uses the **validation** split by default
- Generates **10 rollouts** by default
- Saves results to: `rollouts/deforming_plate_rollout.pkl`

### Custom Evaluation

To evaluate with different settings:

```bash
# Evaluate on test split with 20 rollouts
bash scripts/eval_deforming_plate.sh test 20
```

### Evaluation Output

You'll see output like:
```
I1123 14:48:02.085502 run_model.py:133] Rollout trajectory 0
I1123 14:50:31.819319 run_model.py:133] Rollout trajectory 1
...
I1123 15:12:13.653093 run_model.py:138] mse_1_steps: 3.29947e-07
I1123 15:12:13.653244 run_model.py:138] mse_10_steps: 1.59113e-05
I1123 15:12:13.653297 run_model.py:138] mse_20_steps: 6.19957e-05
I1123 15:12:13.653340 run_model.py:138] mse_50_steps: 0.000381109
I1123 15:12:13.653381 run_model.py:138] mse_100_steps: 0.00152213
I1123 15:12:13.653419 run_model.py:138] mse_200_steps: 0.00610669
Rollouts saved to: rollouts/deforming_plate_rollout.pkl
```

---

## Viewing Results

### 1. View Rollout Summary

To see a detailed summary of the evaluation results:

```bash
python3 view_rollouts.py
```

This shows:
- Number of rollouts
- Trajectory shapes (timesteps × nodes × dimensions)
- Prediction errors at different horizons
- Error statistics across all rollouts

### 2. Access Rollout Data in Python

```python
import pickle
import numpy as np

# Load rollouts
data = pickle.load(open('rollouts/deforming_plate_rollout.pkl', 'rb'))

# Access first rollout
rollout = data[0]
gt_pos = rollout['gt_pos']      # Ground truth positions: (timesteps, nodes, 3)
pred_pos = rollout['pred_pos']  # Predicted positions: (timesteps, nodes, 3)
cells = rollout['cells']         # Mesh connectivity: (timesteps, num_cells, 4)
mesh_pos = rollout['mesh_pos']   # Mesh reference positions: (timesteps, nodes, 3)

# Example: Get position of node 100 at timestep 10
node_100_timestep_10 = gt_pos[10, 100, :]  # [x, y, z] coordinates
```

### 3. Check Training Checkpoints

Checkpoints are saved in:
```
checkpoints/deforming_plate/
├── checkpoint                    # Latest checkpoint info
├── model.ckpt-1000.data-*       # Model weights
├── model.ckpt-1000.index        # Index file
├── model.ckpt-1000.meta         # Graph definition
└── events.out.tfevents.*        # TensorBoard event files
```

To verify training completed:

```bash
python3 verify_training.py
```

---

## Configuration Options

### Training Script (`scripts/train_deforming_plate.sh`)

Key parameters:
- `--dataset_fraction=0.5`: Use 50% of training data (500 trajectories)
- `--num_training_steps=10000`: Train for 10,000 steps
- `--model=deforming_plate`: Use deforming plate model

### Evaluation Script (`scripts/eval_deforming_plate.sh`)

Key parameters:
- `--rollout_split=valid`: Use validation split (default: valid)
- `--num_rollouts=10`: Generate 10 rollouts (default: 10)

### Model Parameters

Located in `Project2/deepmind-research/meshgraphnets/run_model.py`:

```python
PARAMETERS = {
    'deforming_plate': dict(
        noise=0.003,           # Noise scale
        gamma=0.1,             # Noise gamma
        field='world_pos',     # Field to predict
        history=True,          # Use history
        size=3,               # Output size
        batch=1,              # Batch size
        model=deforming_plate_model,
        evaluator=deforming_plate_eval
    )
}
```

---

## Complete Pipeline Example

### Step 1: Train the Model

```bash
# Activate environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate meshgraphnets

# Train (this will take some time)
bash scripts/train_deforming_plate.sh
```

### Step 2: Monitor Training (in another terminal)

```bash
# Start TensorBoard
tensorboard --logdir=checkpoints/deforming_plate --port=6006

# Open http://localhost:6006 in browser
# Watch the loss decrease over time
```

### Step 3: Evaluate the Model

```bash
# Run evaluation
bash scripts/eval_deforming_plate.sh
```

### Step 4: View Results

```bash
# View rollout summary
python3 view_rollouts.py

# View loss statistics
python3 view_loss.py

# Verify training
python3 verify_training.py
```

---

## Troubleshooting

### Issue: "No module named 'sonnet'"

**Solution**: Make sure the conda environment is activated:
```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate meshgraphnets
```

### Issue: Only 2 loss values in TensorBoard

**Solution**: This happens if training stops too early. The normalization period is 100 steps, so you need at least 200+ steps to see training loss. Use:
```bash
bash scripts/train_deforming_plate.sh 10000  # Train for 10,000 steps
```

### Issue: Checkpoint not found

**Solution**: Make sure training completed successfully. Check:
```bash
ls -lh checkpoints/deforming_plate/
```

### Issue: Out of memory

**Solution**: Reduce batch size or dataset fraction in the training script.

---

## Expected Results

### Training Loss
- **Initial loss**: ~50 (at step 100, after normalization)
- **Final loss**: ~0.1-0.3 (after 10,000 steps)
- **Loss reduction**: ~99% improvement

### Evaluation Metrics
- **MSE @ 1 step**: ~1e-7 (very accurate for short-term)
- **MSE @ 10 steps**: ~1e-5
- **MSE @ 100 steps**: ~1e-3
- **MSE @ 200 steps**: ~1e-2

### Rollout Data
- **Number of rollouts**: 10
- **Timesteps per rollout**: 398
- **Nodes per mesh**: ~1645
- **File size**: ~400 MB

---

## Additional Resources

- **TensorBoard**: http://localhost:6006 (when running)
- **Checkpoints**: `checkpoints/deforming_plate/`
- **Rollouts**: `rollouts/deforming_plate_rollout.pkl`
- **Training logs**: Check console output or `/tmp/training.log`

---

## Quick Reference Commands

```bash
# Train
bash scripts/train_deforming_plate.sh

# Evaluate
bash scripts/eval_deforming_plate.sh

# View rollouts
python3 view_rollouts.py

# View loss
python3 view_loss.py

# TensorBoard
tensorboard --logdir=checkpoints/deforming_plate --port=6006

# Verify training
python3 verify_training.py
```

---

## Notes

- Training with 50% of the dataset (500 trajectories) is sufficient for good results
- The model uses 10% of the dataset for the original requirement (100 trajectories)
- Normalization period (first 100 steps) is necessary for proper data normalization
- Loss is logged to TensorBoard every 100 steps
- Checkpoints are saved every 600 seconds during training

