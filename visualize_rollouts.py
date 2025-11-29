#!/usr/bin/env python3
"""Visualize MeshGraphNet rollouts."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_rollout(rollout_data, rollout_idx=0, timesteps_to_show=None):
    """Visualize a single rollout trajectory."""
    if rollout_idx >= len(rollout_data):
        print(f"Error: Rollout index {rollout_idx} out of range (max: {len(rollout_data)-1})")
        return
    
    data = rollout_data[rollout_idx]
    gt_pos = data['gt_pos']  # Shape: (timesteps, num_nodes, 3)
    pred_pos = data['pred_pos']  # Shape: (timesteps, num_nodes, 3)
    mesh_pos = data['mesh_pos']  # Shape: (timesteps, num_nodes, 3)
    
    num_timesteps, num_nodes, _ = gt_pos.shape
    
    if timesteps_to_show is None:
        # Show first, middle, and last timesteps
        timesteps_to_show = [0, num_timesteps // 2, num_timesteps - 1]
    
    fig = plt.figure(figsize=(15, 5 * len(timesteps_to_show)))
    
    for i, t in enumerate(timesteps_to_show):
        if t >= num_timesteps:
            continue
            
        # Ground truth
        ax1 = fig.add_subplot(len(timesteps_to_show), 2, 2*i + 1, projection='3d')
        ax1.scatter(gt_pos[t, :, 0], gt_pos[t, :, 1], gt_pos[t, :, 2], 
                   c='blue', s=1, alpha=0.6)
        ax1.set_title(f'Ground Truth - Timestep {t}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Prediction
        ax2 = fig.add_subplot(len(timesteps_to_show), 2, 2*i + 2, projection='3d')
        ax2.scatter(pred_pos[t, :, 0], pred_pos[t, :, 1], pred_pos[t, :, 2], 
                   c='red', s=1, alpha=0.6)
        ax2.set_title(f'Prediction - Timestep {t}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Set same axis limits for comparison
        all_x = np.concatenate([gt_pos[t, :, 0], pred_pos[t, :, 0]])
        all_y = np.concatenate([gt_pos[t, :, 1], pred_pos[t, :, 1]])
        all_z = np.concatenate([gt_pos[t, :, 2], pred_pos[t, :, 2]])
        
        ax1.set_xlim([all_x.min(), all_x.max()])
        ax1.set_ylim([all_y.min(), all_y.max()])
        ax1.set_zlim([all_z.min(), all_z.max()])
        
        ax2.set_xlim([all_x.min(), all_x.max()])
        ax2.set_ylim([all_y.min(), all_y.max()])
        ax2.set_zlim([all_z.min(), all_z.max()])
    
    plt.tight_layout()
    return fig

def plot_error_over_time(rollout_data, rollout_idx=0):
    """Plot prediction error over time."""
    if rollout_idx >= len(rollout_data):
        print(f"Error: Rollout index {rollout_idx} out of range (max: {len(rollout_data)-1})")
        return
    
    data = rollout_data[rollout_idx]
    gt_pos = data['gt_pos']
    pred_pos = data['pred_pos']
    
    # Compute MSE per timestep
    mse_per_timestep = np.mean((gt_pos - pred_pos) ** 2, axis=(1, 2))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mse_per_timestep)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'Prediction Error Over Time - Rollout {rollout_idx}')
    ax.grid(True)
    
    return fig

def print_rollout_summary(rollout_path):
    """Print summary of rollout data."""
    print(f"\n{'='*60}")
    print(f"ROLLOUT SUMMARY: {rollout_path}")
    print(f"{'='*60}\n")
    
    with open(rollout_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Number of rollouts: {len(data)}")
    
    for i, rollout in enumerate(data):
        gt_pos = rollout['gt_pos']
        pred_pos = rollout['pred_pos']
        
        num_timesteps, num_nodes, _ = gt_pos.shape
        
        # Compute errors
        mse = np.mean((gt_pos - pred_pos) ** 2)
        mse_1 = np.mean((gt_pos[1:2] - pred_pos[1:2]) ** 2)
        mse_10 = np.mean((gt_pos[1:11] - pred_pos[1:11]) ** 2)
        mse_100 = np.mean((gt_pos[1:101] - pred_pos[1:101]) ** 2)
        
        print(f"\nRollout {i}:")
        print(f"  Timesteps: {num_timesteps}")
        print(f"  Nodes: {num_nodes}")
        print(f"  Overall MSE: {mse:.6e}")
        print(f"  MSE @ 1 step: {mse_1:.6e}")
        print(f"  MSE @ 10 steps: {mse_10:.6e}")
        print(f"  MSE @ 100 steps: {mse_100:.6e}")

if __name__ == '__main__':
    import sys
    
    rollout_path = 'rollouts/deforming_plate_rollout.pkl'
    
    if not os.path.exists(rollout_path):
        print(f"Error: Rollout file not found: {rollout_path}")
        sys.exit(1)
    
    # Print summary
    print_rollout_summary(rollout_path)
    
    # Load data
    with open(rollout_path, 'rb') as f:
        rollout_data = pickle.load(f)
    
    # Visualize first rollout
    print("\nGenerating visualization for rollout 0...")
    fig1 = visualize_rollout(rollout_data, rollout_idx=0)
    plt.savefig('rollouts/visualization_rollout_0.png', dpi=150, bbox_inches='tight')
    print("Saved: rollouts/visualization_rollout_0.png")
    
    # Plot error over time
    print("Generating error plot...")
    fig2 = plot_error_over_time(rollout_data, rollout_idx=0)
    plt.savefig('rollouts/error_over_time_rollout_0.png', dpi=150, bbox_inches='tight')
    print("Saved: rollouts/error_over_time_rollout_0.png")
    
    print("\nVisualization complete!")
    print("To view the plots, open:")
    print("  - rollouts/visualization_rollout_0.png")
    print("  - rollouts/error_over_time_rollout_0.png")

