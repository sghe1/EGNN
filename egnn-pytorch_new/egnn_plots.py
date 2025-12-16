"""
Plotting utilities for EGNN predictions.

This module contains functions for creating plots from model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_stress_scatter(stress_true, stress_pred, title="Stress Prediction vs True", output_path=None):
    """
    Create a scatter plot of predicted vs true stress with diagonal reference line.
    
    Args:
        stress_true: (N,) or (T, N) array of true stress values
        stress_pred: (N,) or (T, N) array of predicted stress values
        title: Plot title
        output_path: Optional path to save the plot
    """
    # Flatten if needed
    if len(stress_true.shape) > 1:
        stress_true = stress_true.flatten()
    if len(stress_pred.shape) > 1:
        stress_pred = stress_pred.flatten()
    
    # Remove any NaN or Inf values
    valid_mask = np.isfinite(stress_true) & np.isfinite(stress_pred)
    stress_true = stress_true[valid_mask]
    stress_pred = stress_pred[valid_mask]
    
    if len(stress_true) == 0:
        print("Warning: No valid stress values to plot")
        return
    
    # Compute statistics
    mse = np.mean((stress_pred - stress_true) ** 2)
    mae = np.mean(np.abs(stress_pred - stress_true))
    r2 = 1 - np.sum((stress_true - stress_pred) ** 2) / np.sum((stress_true - np.mean(stress_true)) ** 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(stress_true, stress_pred, alpha=0.3, s=1, c='blue', label='Predictions')
    
    # Diagonal reference line (perfect predictions)
    min_val = min(stress_true.min(), stress_pred.min())
    max_val = max(stress_true.max(), stress_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # Labels and title
    ax.set_xlabel('True Stress', fontsize=12)
    ax.set_ylabel('Predicted Stress', fontsize=12)
    ax.set_title(f'{title}\nMSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.4f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Add text box with statistics
    textstr = f'MSE: {mse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}\nN: {len(stress_true)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Stress scatter plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
