#!/usr/bin/env python3
"""
Batch visualization script to generate visualizations for multiple epochs and timesteps.
"""

import os
import sys
import subprocess
import argparse

def find_epochs(checkpoint_dir):
    """Find all epoch directories."""
    epochs = []
    if not os.path.exists(checkpoint_dir):
        return epochs
    
    for item in os.listdir(checkpoint_dir):
        if item.startswith('epoch_') and os.path.isdir(os.path.join(checkpoint_dir, item)):
            try:
                epoch_num = int(item.split('_')[1])
                epochs.append(epoch_num)
            except ValueError:
                continue
    
    return sorted(epochs)

def main():
    parser = argparse.ArgumentParser(description='Batch visualize predictions for multiple epochs and timesteps')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints/egnn/predictions',
        help='Directory containing prediction files'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/deforming_plate',
        help='Directory containing dataset (for loading cells)'
    )
    parser.add_argument(
        '--traj_idx',
        type=int,
        default=0,
        help='Trajectory index to visualize'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        nargs='+',
        default=[100, 200, 300, 400],
        help='Timesteps to visualize (default: 100 200 300 400)'
    )
    parser.add_argument(
        '--color',
        type=str,
        default='stress',
        choices=['stress', 'velocity', 'none', 'position_norm'],
        help='Color mode for visualization'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # Find all epochs
    epochs = find_epochs(args.checkpoint_dir)
    if not epochs:
        print(f"No epoch directories found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(epochs)} epochs: {epochs}")
    print(f"Will visualize timesteps: {args.timesteps}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the path to visualize_predictions.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualize_script = os.path.join(script_dir, 'visualize_predictions.py')
    
    if not os.path.exists(visualize_script):
        print(f"Error: visualize_predictions.py not found at {visualize_script}")
        return
    
    # Generate visualizations
    total = len(epochs) * len(args.timesteps)
    current = 0
    
    for epoch in epochs:
        for t in args.timesteps:
            current += 1
            output_file = os.path.join(args.output_dir, f'epoch_{epoch}_t_{t}_traj_{args.traj_idx}.png')
            
            print(f"\n[{current}/{total}] Generating: epoch {epoch}, timestep {t}")
            
            # Build command
            cmd = [
                'python', visualize_script,
                '--checkpoint_dir', args.checkpoint_dir,
                '--epoch', str(epoch),
                '--traj_idx', str(args.traj_idx),
                '--t', str(t),
                '--data_dir', args.data_dir,
                '--color', args.color,
                '--output', output_file
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"  ✓ Saved to: {output_file}")
            except subprocess.CalledProcessError as e:
                # Check if it's a timestep out of range error
                if "out of range" in e.stderr.lower() or "timestep" in e.stderr.lower():
                    print(f"  ⚠ Skipped: timestep {t} out of range for epoch {epoch}")
                else:
                    print(f"  ✗ Error generating visualization:")
                    print(f"    {e.stderr}")
                continue
    
    print(f"\n✓ Completed! Generated {current} visualizations in {args.output_dir}")

if __name__ == "__main__":
    main()
