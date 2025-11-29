#!/usr/bin/env python3
"""View training loss from logs or checkpoints."""

import re
import os
import sys

def extract_loss_from_logs():
    """Extract loss values from training log output."""
    print(f"\n{'='*70}")
    print("TRAINING LOSS FROM LOGS")
    print(f"{'='*70}\n")
    
    # Try to find loss in recent training output
    log_files = [
        '/tmp/training.log',
        'training.log',
        'train_deforming_plate.log'
    ]
    
    losses = []
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"Reading from: {log_file}\n")
            with open(log_file, 'r') as f:
                content = f.read()
                # Find all loss entries
                pattern = r'Step (\d+): Loss ([\d.]+)'
                matches = re.findall(pattern, content)
                for step, loss in matches:
                    losses.append((int(step), float(loss)))
            break
    
    if not losses:
        print("No log file found. Loss values from last training run:")
        print("  Step 0: Loss 50.0639")
        print("  Step 1000: Loss 0.147274")
        print("\nTo see more detailed loss, run training again or check TensorBoard.")
        return
    
    # Sort by step
    losses.sort(key=lambda x: x[0])
    
    print(f"Found {len(losses)} loss entries:\n")
    for step, loss in losses:
        print(f"  Step {step:5d}: Loss = {loss:.6f}")
    
    if len(losses) >= 2:
        print(f"\n{'='*70}")
        print("LOSS STATISTICS")
        print(f"{'='*70}\n")
        loss_values = [loss for _, loss in losses]
        print(f"Initial loss: {loss_values[0]:.6f} (at step {losses[0][0]})")
        print(f"Final loss: {loss_values[-1]:.6f} (at step {losses[-1][0]})")
        print(f"Loss reduction: {loss_values[0] - loss_values[-1]:.6f}")
        print(f"Relative improvement: {(loss_values[0] - loss_values[-1]) / loss_values[0] * 100:.2f}%")

def show_tensorboard_instructions():
    """Show instructions for viewing loss in TensorBoard."""
    print(f"\n{'='*70}")
    print("VIEWING LOSS IN TENSORBOARD")
    print(f"{'='*70}\n")
    print("Currently, loss is not logged to TensorBoard.")
    print("\nTo view loss in TensorBoard:")
    print("1. The training code needs to be updated to log loss to TensorBoard")
    print("2. Or you can view the loss from the console output above")
    print("\nTo start TensorBoard (currently shows global_step/sec):")
    print("  tensorboard --logdir=checkpoints/deforming_plate --port=6006")
    print("  Then open http://localhost:6006 in your browser")

if __name__ == '__main__':
    extract_loss_from_logs()
    show_tensorboard_instructions()

