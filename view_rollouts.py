#!/usr/bin/env python3
"""View MeshGraphNet rollouts without visualization dependencies."""

import pickle
import numpy as np
import os
import sys

def print_rollout_summary(rollout_path):
    """Print detailed summary of rollout data."""
    print(f"\n{'='*70}")
    print(f"ROLLOUT DATA SUMMARY: {rollout_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(rollout_path):
        print(f"ERROR: File not found: {rollout_path}")
        return
    
    file_size = os.path.getsize(rollout_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB\n")
    
    with open(rollout_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Number of rollouts: {len(data)}")
    print(f"Data type: {type(data)}\n")
    
    if len(data) == 0:
        print("WARNING: Rollout data is empty!")
        return
    
    # Analyze first rollout in detail
    first_rollout = data[0]
    print(f"{'='*70}")
    print(f"DETAILED ANALYSIS - Rollout 0")
    print(f"{'='*70}\n")
    
    print("Keys in rollout data:")
    for key in first_rollout.keys():
        value = first_rollout[key]
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  - {key}: {type(value)}")
    
    print()
    
    # Extract data
    gt_pos = first_rollout['gt_pos']
    pred_pos = first_rollout['pred_pos']
    cells = first_rollout['cells']
    mesh_pos = first_rollout['mesh_pos']
    
    num_timesteps, num_nodes, dims = gt_pos.shape
    
    print(f"Trajectory shape: {num_timesteps} timesteps × {num_nodes} nodes × {dims} dimensions")
    print(f"Mesh cells shape: {cells.shape}")
    print(f"Mesh positions shape: {mesh_pos.shape}\n")
    
    # Compute errors
    print(f"{'='*70}")
    print("PREDICTION ERRORS")
    print(f"{'='*70}\n")
    
    # Overall error
    mse_overall = np.mean((gt_pos - pred_pos) ** 2)
    mae_overall = np.mean(np.abs(gt_pos - pred_pos))
    
    print(f"Overall (all timesteps):")
    print(f"  Mean Squared Error (MSE): {mse_overall:.6e}")
    print(f"  Mean Absolute Error (MAE): {mae_overall:.6e}")
    print()
    
    # Error at different horizons
    horizons = [1, 10, 20, 50, 100, 200]
    print("Error at different prediction horizons:")
    for h in horizons:
        if h < num_timesteps:
            mse_h = np.mean((gt_pos[1:h+1] - pred_pos[1:h+1]) ** 2)
            print(f"  MSE @ {h:3d} steps: {mse_h:.6e}")
    
    print()
    
    # Error per timestep
    mse_per_timestep = np.mean((gt_pos - pred_pos) ** 2, axis=(1, 2))
    print(f"Error progression:")
    print(f"  Initial (step 0): {mse_per_timestep[0]:.6e}")
    print(f"  Middle (step {num_timesteps//2}): {mse_per_timestep[num_timesteps//2]:.6e}")
    print(f"  Final (step {num_timesteps-1}): {mse_per_timestep[-1]:.6e}")
    
    print()
    
    # Statistics for all rollouts
    if len(data) > 1:
        print(f"{'='*70}")
        print(f"SUMMARY FOR ALL {len(data)} ROLLOUTS")
        print(f"{'='*70}\n")
        
        all_mse = []
        for i, rollout in enumerate(data):
            gt = rollout['gt_pos']
            pred = rollout['pred_pos']
            mse = np.mean((gt - pred) ** 2)
            all_mse.append(mse)
            print(f"Rollout {i:2d}: MSE = {mse:.6e}")
        
        print()
        print(f"Average MSE across all rollouts: {np.mean(all_mse):.6e}")
        print(f"Std dev of MSE: {np.std(all_mse):.6e}")
    
    print(f"\n{'='*70}")
    print("DATA ACCESS EXAMPLES")
    print(f"{'='*70}\n")
    print("To access the data in Python:")
    print("  import pickle")
    print("  data = pickle.load(open('rollouts/deforming_plate_rollout.pkl', 'rb'))")
    print("  ")
    print("  # Access rollout 0:")
    print("  rollout = data[0]")
    print("  gt_pos = rollout['gt_pos']      # Ground truth positions")
    print("  pred_pos = rollout['pred_pos']  # Predicted positions")
    print("  cells = rollout['cells']         # Mesh connectivity")
    print("  mesh_pos = rollout['mesh_pos']   # Mesh reference positions")
    print()
    print("  # Shape: (timesteps, num_nodes, 3)")
    print("  # Example: gt_pos[10, 100, :] gives position of node 100 at timestep 10")

if __name__ == '__main__':
    rollout_path = 'rollouts/deforming_plate_rollout.pkl'
    
    if len(sys.argv) > 1:
        rollout_path = sys.argv[1]
    
    print_rollout_summary(rollout_path)

