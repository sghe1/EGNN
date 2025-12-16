# Debug Run Guide - Sanity Check Training

This guide explains how to run a minimal sanity-check training run to verify the refactored EGNN code works correctly.

## Quick Start

Run this command from the `egnn-pytorch_new/` directory:

```bash
python egnn_train.py \
    --data_dir data/deforming_plate \
    --checkpoint_dir checkpoints/egnn_debug \
    --debug_one_traj \
    --debug_traj_id 0 \
    --debug_max_timesteps 4 \
    --debug_num_epochs 50 \
    --device cuda
```

Or if using CPU:

```bash
python egnn_train.py \
    --data_dir data/deforming_plate \
    --checkpoint_dir checkpoints/egnn_debug \
    --debug_one_traj \
    --debug_traj_id 0 \
    --debug_max_timesteps 4 \
    --debug_num_epochs 50 \
    --device cpu
```

## What This Does

1. **Loads only 1 trajectory** (trajectory ID = 0)
2. **Uses only 4 timesteps** (t = 0, 1, 2, 3)
   - Creates 3 training pairs: (0→1), (1→2), (2→3)
3. **Trains for 50 epochs**
4. **Forces batch_size=1** and **num_workers=0** for debug-friendly settings
5. **Disables warmup and stress threshold** to train on all available timesteps

## Expected Output

You should see:

```
======================================================================
DEBUG MODE ENABLED
======================================================================
  Using trajectory ID: 0
  Max timesteps: 4
  Number of epochs: 50
  Batch size: 1 (forced)
  Warmup fraction: 0.0 (disabled for debug)
  Min stress threshold: 0.0 (disabled for debug)
======================================================================

DEBUG: Trajectory loaded with 4 timesteps
DEBUG: Adjacency matrix shape: (4, N, N) (per-timestep)
DEBUG: Edge statistics (timestep 0):
  Number of nodes: <N>
  Number of edges: <E>
  Average degree: <degree>
  Base mesh edges: <E_mesh>
  Proximity edges (estimated): <E_prox>
  ✓ Proximity edges detected!

Feature dimension: 12 (expected: 12 = [mesh_pos(3), world_pos(3), node_type(2), stress(1), velocity(3)])
Target velocity shape: (4, N, 3) (expected: (T, N, 3))
Target stress shape: (4, N, 1) (expected: (T, N, 1))
Coordinates shape: (4, N, 3) (expected: (T, N, 3))
Adjacency matrix shape: (4, N, N) (expected: (T, N, N) for per-timestep)

DEBUG: First batch assertions:
  Features shape: (1, 4, N, 12), expected last dim = 12
  Adjacency shape: (4, N, N)
  ✓ Per-timestep adjacency matrices detected: (T=4, N=<N>)
  Edge stats (batch 0, timestep 0): <E> edges, avg degree: <degree>
```

## Verification Checklist

The debug run will automatically verify:

- ✅ Feature dimension is 12D
- ✅ Adjacency matrices are per-timestep (T, N, N)
- ✅ Proximity edges are detected (more edges than base mesh)
- ✅ Correct timestep slicing (4 timesteps)
- ✅ Correct trajectory loading (trajectory ID 0)

## Files Modified

1. **`egnn_train.py`**:
   - Added debug flags: `--debug_one_traj`, `--debug_traj_id`, `--debug_max_timesteps`, `--debug_num_epochs`
   - Added debug mode logic to override settings
   - Added assertions and logging for first batch
   - Force batch_size=1 and num_workers=0 in debug mode

2. **`defplate_dataset.py`**:
   - Added `debug_traj_id` and `debug_max_timesteps` parameters
   - Modified `__getitem__` to load specific trajectory and slice timesteps
   - Added debug logging

## Normal Training (Non-Debug)

To run normal training, simply omit the `--debug_one_traj` flag:

```bash
python egnn_train.py \
    --data_dir data/deforming_plate \
    --checkpoint_dir checkpoints/egnn \
    --dataset_fraction 0.1 \
    --num_epochs 10 \
    --batch_size 1
```

All debug flags are optional and only take effect when `--debug_one_traj` is specified.

## Troubleshooting

**Issue: "Trajectory ID X is out of range"**
- Solution: Use a smaller trajectory ID (try 0, 1, 2, etc.)

**Issue: "Expected 12D features, got 8D"**
- Solution: Make sure you're using the updated code with 12D features

**Issue: "No proximity edges detected"**
- Solution: This may be normal if `proximity_radius` is too small. Check the proximity_radius parameter in dataset initialization.

**Issue: Training loss is NaN**
- Solution: Check that normalization statistics are computed correctly. Try recomputing them.

---

**End of Guide**
