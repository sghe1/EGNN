#!/usr/bin/env python3
"""
Compare predictions with ground truth values.
Visualizes and prints statistics for velocity and stress predictions.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_trajectory_data(trajectory_dir):
    """Load predictions and ground truth for a trajectory."""
    pred_vel = np.load(os.path.join(trajectory_dir, 'predictions_velocity.npy'))
    pred_stress = np.load(os.path.join(trajectory_dir, 'predictions_stress.npy'))
    true_vel = np.load(os.path.join(trajectory_dir, 'true_velocity.npy'))
    true_stress = np.load(os.path.join(trajectory_dir, 'true_stress.npy'))
    errors_vel = np.load(os.path.join(trajectory_dir, 'errors_velocity.npy'))
    errors_stress = np.load(os.path.join(trajectory_dir, 'errors_stress.npy'))
    
    return {
        'pred_vel': pred_vel,
        'pred_stress': pred_stress,
        'true_vel': true_vel,
        'true_stress': true_stress,
        'errors_vel': errors_vel,
        'errors_stress': errors_stress
    }

def print_statistics(data_dir):
    """Print detailed statistics for all trajectories."""
    summary_path = os.path.join(data_dir, 'summary.json')
    
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"\n{'='*70}")
    print("DETAILED STATISTICS")
    print(f"{'='*70}\n")
    
    print(f"Overall Metrics:")
    print(f"  Velocity - MSE: {summary['overall_velocity_mse']:.6f}, MAE: {summary['overall_velocity_mae']:.6f}")
    print(f"  Stress   - MSE: {summary['overall_stress_mse']:.6f}, MAE: {summary['overall_stress_mae']:.6f}")
    
    print(f"\nPer-Trajectory Metrics:")
    for traj in summary['trajectories']:
        print(f"\n  Trajectory {traj['trajectory_id']}:")
        print(f"    Shape: {traj['shape']['timesteps']} timesteps, {traj['shape']['nodes']} nodes")
        print(f"    Velocity - MSE: {traj['velocity_mse']:.6f}, MAE: {traj['velocity_mae']:.6f}")
        print(f"    Stress   - MSE: {traj['stress_mse']:.6f}, MAE: {traj['stress_mae']:.6f}")
    
    # Load and print per-trajectory detailed stats
    print(f"\n{'='*70}")
    print("PER-TRAJECTORY DETAILED STATISTICS")
    print(f"{'='*70}\n")
    
    for traj in summary['trajectories']:
        traj_id = traj['trajectory_id']
        traj_dir = os.path.join(data_dir, f'trajectory_{traj_id}')
        
        if not os.path.exists(traj_dir):
            continue
        
        data = load_trajectory_data(traj_dir)
        
        # Velocity statistics
        pred_vel = data['pred_vel']  # (T-1, N, 3)
        true_vel = data['true_vel']
        
        # Compute per-component statistics
        vel_errors = np.abs(pred_vel - true_vel)  # (T-1, N, 3)
        vel_magnitude_pred = np.linalg.norm(pred_vel, axis=-1)  # (T-1, N)
        vel_magnitude_true = np.linalg.norm(true_vel, axis=-1)
        
        print(f"Trajectory {traj_id} - Velocity:")
        print(f"  Component-wise MAE:")
        print(f"    X: {np.mean(vel_errors[:, :, 0]):.6f}")
        print(f"    Y: {np.mean(vel_errors[:, :, 1]):.6f}")
        print(f"    Z: {np.mean(vel_errors[:, :, 2]):.6f}")
        print(f"  Magnitude - Predicted: mean={np.mean(vel_magnitude_pred):.6f}, std={np.std(vel_magnitude_pred):.6f}")
        print(f"  Magnitude - True:      mean={np.mean(vel_magnitude_true):.6f}, std={np.std(vel_magnitude_true):.6f}")
        print(f"  Magnitude - Error:      mean={np.mean(np.abs(vel_magnitude_pred - vel_magnitude_true)):.6f}")
        
        # Stress statistics
        pred_stress = data['pred_stress']  # (T-1, N, 1)
        true_stress = data['true_stress']
        
        print(f"\nTrajectory {traj_id} - Stress:")
        print(f"  Predicted: mean={np.mean(pred_stress):.6f}, std={np.std(pred_stress):.6f}, min={np.min(pred_stress):.6f}, max={np.max(pred_stress):.6f}")
        print(f"  True:      mean={np.mean(true_stress):.6f}, std={np.std(true_stress):.6f}, min={np.min(true_stress):.6f}, max={np.max(true_stress):.6f}")
        print(f"  Error:     mean={np.mean(np.abs(pred_stress - true_stress)):.6f}, max={np.max(np.abs(pred_stress - true_stress)):.6f}")

def plot_comparison(data_dir, trajectory_id=0, save_plots=True):
    """Plot predictions vs ground truth for a trajectory."""
    traj_dir = os.path.join(data_dir, f'trajectory_{trajectory_id}')
    
    if not os.path.exists(traj_dir):
        print(f"Trajectory directory not found: {traj_dir}")
        return
    
    data = load_trajectory_data(traj_dir)
    
    pred_vel = data['pred_vel']  # (T-1, N, 3)
    true_vel = data['true_vel']
    pred_stress = data['pred_stress']  # (T-1, N, 1)
    true_stress = data['true_stress']
    
    T, N, _ = pred_vel.shape
    
    # Plot 1: Velocity magnitude over time (averaged over nodes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Velocity magnitude
    vel_mag_pred = np.linalg.norm(pred_vel, axis=-1)  # (T-1, N)
    vel_mag_true = np.linalg.norm(true_vel, axis=-1)
    vel_mag_pred_mean = np.mean(vel_mag_pred, axis=1)  # (T-1,)
    vel_mag_true_mean = np.mean(vel_mag_true, axis=1)
    vel_mag_pred_std = np.std(vel_mag_pred, axis=1)
    vel_mag_true_std = np.std(vel_mag_true, axis=1)
    
    ax = axes[0, 0]
    timesteps = np.arange(1, T+1)
    ax.plot(timesteps, vel_mag_pred_mean, 'r-', label='Predicted', linewidth=2)
    ax.fill_between(timesteps, vel_mag_pred_mean - vel_mag_pred_std, 
                    vel_mag_pred_mean + vel_mag_pred_std, alpha=0.3, color='red')
    ax.plot(timesteps, vel_mag_true_mean, 'b--', label='True', linewidth=2)
    ax.fill_between(timesteps, vel_mag_true_mean - vel_mag_true_std,
                    vel_mag_true_mean + vel_mag_true_std, alpha=0.3, color='blue')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity Magnitude')
    ax.set_title(f'Velocity Magnitude Over Time (Trajectory {trajectory_id})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity error over time
    ax = axes[0, 1]
    vel_error = np.mean(np.abs(vel_mag_pred - vel_mag_true), axis=1)
    ax.plot(timesteps, vel_error, 'g-', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'Velocity Error Over Time (Trajectory {trajectory_id})')
    ax.grid(True, alpha=0.3)
    
    # Stress over time (averaged over nodes)
    stress_pred_mean = np.mean(pred_stress[:, :, 0], axis=1)  # (T-1,)
    stress_true_mean = np.mean(true_stress[:, :, 0], axis=1)
    stress_pred_std = np.std(pred_stress[:, :, 0], axis=1)
    stress_true_std = np.std(true_stress[:, :, 0], axis=1)
    
    ax = axes[1, 0]
    ax.plot(timesteps, stress_pred_mean, 'r-', label='Predicted', linewidth=2)
    ax.fill_between(timesteps, stress_pred_mean - stress_pred_std,
                    stress_pred_mean + stress_pred_std, alpha=0.3, color='red')
    ax.plot(timesteps, stress_true_mean, 'b--', label='True', linewidth=2)
    ax.fill_between(timesteps, stress_true_mean - stress_true_std,
                    stress_true_mean + stress_true_std, alpha=0.3, color='blue')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Stress')
    ax.set_title(f'Stress Over Time (Trajectory {trajectory_id})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stress error over time
    ax = axes[1, 1]
    stress_error = np.mean(np.abs(pred_stress[:, :, 0] - true_stress[:, :, 0]), axis=1)
    ax.plot(timesteps, stress_error, 'g-', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'Stress Error Over Time (Trajectory {trajectory_id})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(traj_dir, f'comparison_plot_traj_{trajectory_id}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare predictions with ground truth')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test results (e.g., test_results/egnn_quick_test)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--trajectory_id', type=int, default=0,
                       help='Trajectory ID to plot (default: 0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        return
    
    # Print statistics
    print_statistics(args.data_dir)
    
    # Generate plots if requested
    if args.plot:
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print(f"{'='*70}\n")
        
        # Plot all trajectories
        summary_path = os.path.join(args.data_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            for traj in summary['trajectories']:
                traj_id = traj['trajectory_id']
                print(f"Plotting trajectory {traj_id}...")
                plot_comparison(args.data_dir, traj_id, save_plots=True)
        else:
            # Just plot the specified trajectory
            plot_comparison(args.data_dir, args.trajectory_id, save_plots=True)

if __name__ == '__main__':
    main()
