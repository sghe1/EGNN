# MeshGraphNet Training Results

## Training Summary

✅ **Training Status**: Completed Successfully

- **Model**: deforming_plate
- **Dataset**: deforming_plate (10% of full dataset = 100 trajectories)
- **Training Steps**: 1000 steps
- **Final Loss**: 0.0588012 (down from 9.94834 at step 0)
- **Training Speed**: ~4 steps/second

## Results Location

### 1. Checkpoints
**Location**: `checkpoints/deforming_plate/`

**Latest Checkpoint**: `model.ckpt-1001`
- `model.ckpt-1001.data-00000-of-00001` - Model weights (27 MB)
- `model.ckpt-1001.index` - Checkpoint index
- `model.ckpt-1001.meta` - Graph metadata

### 2. TensorBoard Logs
**Location**: `checkpoints/deforming_plate/`

**Files**: `events.out.tfevents.*`

**To View**:
```bash
tensorboard --logdir=checkpoints/deforming_plate
```
Then open http://localhost:6006/ in your browser.

**What to Look For**:
- **Scalars Tab**: Loss curve showing decrease from ~9.95 to ~0.06
- **Graphs Tab**: Model architecture visualization

### 3. Evaluation Rollouts (After Running Evaluation)
**Location**: `rollouts/deforming_plate_rollout.pkl`

**To Generate**:
```bash
bash scripts/eval_deforming_plate.sh
```

## Verification

Run the verification script to confirm everything is correct:
```bash
python verify_training.py
```

## Training Configuration

- **Model**: deforming_plate (MeshGraphNet)
- **Dataset**: deforming_plate
- **Dataset Fraction**: 0.1 (10% = 100 trajectories)
- **Training Steps**: 1000
- **Learning Rate**: Exponential decay starting at 1e-4
- **Batch Size**: 1
- **Environment**: meshgraphnets conda environment

## Notes

- The TensorBoard warnings about "more than one graph event" are **normal** and not errors
- They occur because TensorBoard found multiple graph definitions and uses the newest one
- Training completed without any actual errors

## Next Steps

1. **View Training Metrics**: Open TensorBoard at http://localhost:6006/
2. **Run Evaluation**: Generate rollouts with `bash scripts/eval_deforming_plate.sh`
3. **Continue Training**: Run `bash scripts/train_deforming_plate.sh 10000` for more steps
4. **Analyze Results**: Use the rollouts to visualize model predictions

