#!/usr/bin/env python3
"""
Visualize EGNN predictions from checkpoint directory.
Shows side-by-side comparison of true vs predicted mesh at a specific timestep.
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from data_loader_egnn import load_raw_trajectory_from_tfrecord


def load_npy(path):
    """Load numpy array or pickle file, handling both formats."""
    # Try pickle first (new format)
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Return list directly - no need to convert to numpy array
            # Lists work fine for indexing and avoid broadcasting issues
            return data
    except (pickle.UnpicklingError, EOFError, TypeError):
        # Fall back to numpy format (old format)
        arr = np.load(path, allow_pickle=True)
        # If it's an object array (list of arrays), return as-is
        if arr.dtype == object:
            return arr
        # If it's a regular array, check if it needs to be converted to list
        # Arrays saved with allow_pickle=True and dtype=object should be object arrays
        # But if it's a regular array with shape like (num_traj, T, N, ...), keep it as array
        # Don't squeeze - we need to preserve dimensions for indexing
        return arr


def visualize_mesh_pair(pos_true, pos_pred, stress_true=None, stress_pred=None,
                        cells=None, color_mode="stress", title_true="True", title_pred="Prediction"):
    """
    Visualize mesh pair side-by-side.
    
    Args:
        pos_true: (N, 3) true positions
        pos_pred: (N, 3) predicted positions
        stress_true: (N,) or (N, 1) true stress/intensity values
        stress_pred: (N,) or (N, 1) predicted stress/intensity values
        cells: (C, 4) cell connectivity (quads)
        color_mode: "stress", "velocity", "position_norm", or "none"
        title_true: Title for true mesh
        title_pred: Title for predicted mesh
    """
    # Ensure positions have correct shape (N, 3)
    pos_true = np.asarray(pos_true)
    pos_pred = np.asarray(pos_pred)
    if len(pos_true.shape) == 1:
        pos_true = pos_true.reshape(-1, 3)
    if len(pos_pred.shape) == 1:
        pos_pred = pos_pred.reshape(-1, 3)
    if pos_true.shape[1] != 3:
        # Might be transposed (3, N) -> (N, 3)
        if pos_true.shape[0] == 3:
            pos_true = pos_true.T
        else:
            raise ValueError(f"Unexpected pos_true shape: {pos_true.shape}")
    if pos_pred.shape[1] != 3:
        # Might be transposed (3, N) -> (N, 3)
        if pos_pred.shape[0] == 3:
            pos_pred = pos_pred.T
        else:
            raise ValueError(f"Unexpected pos_pred shape: {pos_pred.shape}")
    
    fig = plt.figure(figsize=(16, 8))
    
    # Prepare intensity values
    if color_mode != "none" and stress_true is not None and stress_pred is not None:
        intensity_true = stress_true.flatten() if len(stress_true.shape) > 1 else stress_true
        intensity_pred = stress_pred.flatten() if len(stress_pred.shape) > 1 else stress_pred
        
        # Debug: print stress ranges
        print(f"Stress ranges - True: min={intensity_true.min():.6f}, max={intensity_true.max():.6f}, "
              f"mean={intensity_true.mean():.6f}, std={intensity_true.std():.6f}")
        print(f"Stress ranges - Pred: min={intensity_pred.min():.6f}, max={intensity_pred.max():.6f}, "
              f"mean={intensity_pred.mean():.6f}, std={intensity_pred.std():.6f}")
        
        # Calculate separate color scales for true and predicted to show variation in each
        # This is important because predicted values might be in a very different range
        vmin_true = intensity_true.min()
        vmax_true = intensity_true.max()
        vmin_pred = intensity_pred.min()
        vmax_pred = intensity_pred.max()
        
        # If all values are the same, add a small range to make colormap work
        if abs(vmax_true - vmin_true) < 1e-10:
            print(f"Warning: All true stress values are nearly identical ({vmin_true:.6f}). Using small range for colormap.")
            vmin_true = vmin_true - abs(vmin_true) * 0.01 if abs(vmin_true) > 1e-10 else -1.0
            vmax_true = vmax_true + abs(vmax_true) * 0.01 if abs(vmax_true) > 1e-10 else 1.0
        
        if abs(vmax_pred - vmin_pred) < 1e-10:
            print(f"Warning: All predicted stress values are nearly identical ({vmin_pred:.6f}). Using small range for colormap.")
            vmin_pred = vmin_pred - abs(vmin_pred) * 0.01 if abs(vmin_pred) > 1e-10 else -1.0
            vmax_pred = vmax_pred + abs(vmax_pred) * 0.01 if abs(vmax_pred) > 1e-10 else 1.0
        
        # Use diverging colormap if values can be negative, otherwise sequential
        if vmin_true < 0 or vmin_pred < 0:
            cmap = 'RdBu_r'  # Red-Blue diverging (reversed)
        else:
            cmap = 'viridis'  # Sequential
    else:
        intensity_true = None
        intensity_pred = None
        vmin_true = 0
        vmax_true = 1
        vmin_pred = 0
        vmax_pred = 1
        cmap = 'viridis'
    
    # True mesh
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Debug: print position ranges
    print(f"True positions - X: [{pos_true[:, 0].min():.4f}, {pos_true[:, 0].max():.4f}], "
          f"Y: [{pos_true[:, 1].min():.4f}, {pos_true[:, 1].max():.4f}], "
          f"Z: [{pos_true[:, 2].min():.4f}, {pos_true[:, 2].max():.4f}], "
          f"Num points: {len(pos_true)}")
    
    # Draw mesh faces if cells are available (continuous surface)
    if cells is not None and len(cells) > 0:
        try:
            # Create Poly3DCollection for filled faces
            faces = []
            face_colors = []
            
            # Use all cells for better visualization
            max_cells = len(cells)
            
            for cell_idx, cell in enumerate(cells[:max_cells]):
                cell_indices = np.asarray(cell, dtype=int)
                # Ensure indices are within bounds
                valid_indices = cell_indices[cell_indices < len(pos_true)]
                if len(valid_indices) >= 3:
                    face_verts = pos_true[valid_indices]
                    
                    # For quads, split into two triangles
                    if len(face_verts) == 4:
                        # Two triangles: [0,1,2] and [0,2,3]
                        faces.append([face_verts[0], face_verts[1], face_verts[2]])
                        faces.append([face_verts[0], face_verts[2], face_verts[3]])
                        
                        # Get stress value for this cell (average of vertices)
                        if intensity_true is not None:
                            cell_stress_1 = intensity_true[valid_indices[:3]].mean()
                            cell_stress_2 = intensity_true[[valid_indices[0], valid_indices[2], valid_indices[3]]].mean()
                            face_colors.append(cell_stress_1)
                            face_colors.append(cell_stress_2)
                    elif len(face_verts) == 3:
                        faces.append([face_verts[0], face_verts[1], face_verts[2]])
                        if intensity_true is not None:
                            cell_stress = intensity_true[valid_indices].mean()
                            face_colors.append(cell_stress)
            
            print(f"DEBUG: Created {len(faces)} faces from {max_cells} cells")
            
            if len(faces) > 0:
                # Create Poly3DCollection
                if intensity_true is not None and len(face_colors) == len(faces):
                    # Map stress values to colors using colormap with TRUE scale
                    norm_true = mcolors.Normalize(vmin=vmin_true, vmax=vmax_true)
                    try:
                        colormap = plt.colormaps[cmap]
                    except (KeyError, AttributeError):
                        colormap = cm.get_cmap(cmap)
                    # Map colors and ensure they're in RGBA format
                    face_colors_mapped = []
                    for c in face_colors:
                        rgba = colormap(norm_true(c))
                        # Ensure it's a tuple/list of 4 values (RGBA)
                        if isinstance(rgba, tuple) and len(rgba) == 4:
                            face_colors_mapped.append(rgba)
                        elif isinstance(rgba, np.ndarray):
                            face_colors_mapped.append(tuple(rgba))
                        else:
                            # Convert to RGBA tuple
                            face_colors_mapped.append((rgba[0], rgba[1], rgba[2], 1.0))
                    
                    print(f"Created {len(faces)} faces with stress coloring (range: {min(face_colors):.2f} to {max(face_colors):.2f}, scale: {vmin_true:.2f} to {vmax_true:.2f})")
                    
                    # Use stress for coloring - create collection first, then set colors
                    face_collection = Poly3DCollection(faces, alpha=0.9, linewidths=0.0, edgecolors='none')
                    face_collection.set_facecolor(face_colors_mapped)
                    ax1.add_collection3d(face_collection)
                    
                    # Also add scatter points with stress coloring as backup/overlay for better visibility
                    scatter1 = ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                                          c=intensity_true, cmap=colormap, s=20, alpha=0.6, 
                                          edgecolors='none', vmin=vmin_true, vmax=vmax_true)
                    
                    # Create a ScalarMappable for the colorbar
                    sm = cm.ScalarMappable(cmap=colormap, norm=norm_true)
                    sm.set_array([])
                    cbar1 = plt.colorbar(sm, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                    print(f"DEBUG: Added Poly3DCollection with {len(face_collection._facecolors)} face colors + scatter overlay")
                else:
                    # No coloring, use uniform color
                    print(f"Warning: face_colors length ({len(face_colors) if intensity_true is not None else 0}) != faces length ({len(faces)})")
                    face_collection = Poly3DCollection(faces, alpha=1.0, linewidths=0.1, edgecolors='none', facecolors='lightblue')
                    ax1.add_collection3d(face_collection)
            else:
                print("Warning: No valid faces created, falling back to scatter plot")
                # Fallback to scatter if no valid faces
                if intensity_true is not None:
                    scatter1 = ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                                          c=intensity_true, cmap=cmap, s=50, alpha=1.0, 
                                          edgecolors='black', linewidths=0.2, vmin=vmin_true, vmax=vmax_true)
                    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                else:
                    ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2], 
                               c='blue', s=50, alpha=1.0, edgecolors='black', linewidths=0.2)
        except Exception as e:
            import traceback
            print(f"ERROR: Could not plot mesh faces: {e}")
            traceback.print_exc()
            # Fallback to scatter plot
            if intensity_true is not None:
                scatter1 = ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                                      c=intensity_true, cmap=cmap, s=50, alpha=1.0, 
                                      edgecolors='black', linewidths=0.2, vmin=vmin_true, vmax=vmax_true)
                cbar1 = plt.colorbar(scatter1, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
            else:
                ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2], 
                           c='blue', s=50, alpha=1.0, edgecolors='black', linewidths=0.2)
    else:
        # No cells available, use scatter plot
        if intensity_true is not None:
            scatter1 = ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                                  c=intensity_true, cmap=cmap, s=50, alpha=1.0, 
                                  edgecolors='black', linewidths=0.2, vmin=vmin_true, vmax=vmax_true)
            cbar1 = plt.colorbar(scatter1, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
        else:
            ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2], 
                       c='blue', s=50, alpha=1.0, edgecolors='black', linewidths=0.2)
    
    ax1.set_title(title_true, fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Predicted mesh
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Debug: print position ranges
    print(f"Pred positions - X: [{pos_pred[:, 0].min():.4f}, {pos_pred[:, 0].max():.4f}], "
          f"Y: [{pos_pred[:, 1].min():.4f}, {pos_pred[:, 1].max():.4f}], "
          f"Z: [{pos_pred[:, 2].min():.4f}, {pos_pred[:, 2].max():.4f}], "
          f"Num points: {len(pos_pred)}")
    
    # Draw mesh faces if cells are available (continuous surface)
    if cells is not None and len(cells) > 0:
        try:
            # Create Poly3DCollection for filled faces
            faces = []
            face_colors = []
            
            # Use all cells for better visualization
            max_cells = len(cells)
            
            for cell_idx, cell in enumerate(cells[:max_cells]):
                cell_indices = np.asarray(cell, dtype=int)
                # Ensure indices are within bounds
                valid_indices = cell_indices[cell_indices < len(pos_pred)]
                if len(valid_indices) >= 3:
                    face_verts = pos_pred[valid_indices]
                    
                    # For quads, split into two triangles
                    if len(face_verts) == 4:
                        # Two triangles: [0,1,2] and [0,2,3]
                        faces.append([face_verts[0], face_verts[1], face_verts[2]])
                        faces.append([face_verts[0], face_verts[2], face_verts[3]])
                        
                        # Get stress value for this cell (average of vertices)
                        if intensity_pred is not None:
                            cell_stress_1 = intensity_pred[valid_indices[:3]].mean()
                            cell_stress_2 = intensity_pred[[valid_indices[0], valid_indices[2], valid_indices[3]]].mean()
                            face_colors.append(cell_stress_1)
                            face_colors.append(cell_stress_2)
                    elif len(face_verts) == 3:
                        faces.append([face_verts[0], face_verts[1], face_verts[2]])
                        if intensity_pred is not None:
                            cell_stress = intensity_pred[valid_indices].mean()
                            face_colors.append(cell_stress)
            
            print(f"DEBUG: Created {len(faces)} faces from {max_cells} cells (predicted)")
            
            if len(faces) > 0:
                # Create Poly3DCollection
                if intensity_pred is not None and len(face_colors) == len(faces):
                    # Map stress values to colors using colormap with PREDICTED scale
                    norm_pred = mcolors.Normalize(vmin=vmin_pred, vmax=vmax_pred)
                    try:
                        colormap = plt.colormaps[cmap]
                    except (KeyError, AttributeError):
                        colormap = cm.get_cmap(cmap)
                    # Map colors and ensure they're in RGBA format
                    face_colors_mapped = []
                    for c in face_colors:
                        rgba = colormap(norm_pred(c))
                        # Ensure it's a tuple/list of 4 values (RGBA)
                        if isinstance(rgba, tuple) and len(rgba) == 4:
                            face_colors_mapped.append(rgba)
                        elif isinstance(rgba, np.ndarray):
                            face_colors_mapped.append(tuple(rgba))
                        else:
                            # Convert to RGBA tuple
                            face_colors_mapped.append((rgba[0], rgba[1], rgba[2], 1.0))
                    
                    print(f"Created {len(faces)} faces with stress coloring (range: {min(face_colors):.2f} to {max(face_colors):.2f}, scale: {vmin_pred:.2f} to {vmax_pred:.2f})")
                    
                    # Use stress for coloring - create collection first, then set colors
                    face_collection = Poly3DCollection(faces, alpha=0.9, linewidths=0.0, edgecolors='none')
                    face_collection.set_facecolor(face_colors_mapped)
                    ax2.add_collection3d(face_collection)
                    
                    # Also add scatter points with stress coloring as backup/overlay for better visibility
                    scatter2 = ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                                          c=intensity_pred, cmap=colormap, s=20, alpha=0.6,
                                          edgecolors='none', vmin=vmin_pred, vmax=vmax_pred)
                    
                    # Create a ScalarMappable for the colorbar
                    sm = cm.ScalarMappable(cmap=colormap, norm=norm_pred)
                    sm.set_array([])
                    cbar2 = plt.colorbar(sm, ax=ax2, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                    print(f"DEBUG: Added Poly3DCollection with {len(face_collection._facecolors)} face colors + scatter overlay (predicted)")
                else:
                    # No coloring, use uniform color
                    print(f"Warning: face_colors length ({len(face_colors) if intensity_pred is not None else 0}) != faces length ({len(faces)})")
                    face_collection = Poly3DCollection(faces, alpha=1.0, linewidths=0.1, edgecolors='none', facecolors='lightcoral')
                    ax2.add_collection3d(face_collection)
            else:
                # Fallback to scatter if no valid faces
                if intensity_pred is not None:
                    scatter2 = ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                                          c=intensity_pred, cmap=cmap, s=50, alpha=1.0,
                                          edgecolors='black', linewidths=0.2, vmin=vmin_pred, vmax=vmax_pred)
                    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                else:
                    ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 
                               c='red', s=50, alpha=1.0, edgecolors='black', linewidths=0.2)
        except Exception as e:
            print(f"Warning: Could not plot mesh faces: {e}")
            # Fallback to scatter plot
            if intensity_pred is not None:
                scatter2 = ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                                      c=intensity_pred, cmap=cmap, s=50, alpha=1.0,
                                      edgecolors='black', linewidths=0.2, vmin=vmin_pred, vmax=vmax_pred)
                cbar2 = plt.colorbar(scatter2, ax=ax2, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
            else:
                ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 
                           c='red', s=50, alpha=1.0, edgecolors='black', linewidths=0.2)
    else:
        # No cells available, use scatter plot
        if intensity_pred is not None:
            scatter2 = ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                                  c=intensity_pred, cmap=cmap, s=50, alpha=1.0,
                                  edgecolors='black', linewidths=0.2, vmin=vmin_pred, vmax=vmax_pred)
            cbar2 = plt.colorbar(scatter2, ax=ax2, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
        else:
            ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 
                       c='red', s=50, alpha=1.0, edgecolors='black', linewidths=0.2)
    
    ax2.set_title(title_pred, fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Check if predicted positions are in a very different range than true positions
    true_range = np.array([pos_true[:, 0].max() - pos_true[:, 0].min(),
                           pos_true[:, 1].max() - pos_true[:, 1].min(),
                           pos_true[:, 2].max() - pos_true[:, 2].min()])
    pred_range = np.array([pos_pred[:, 0].max() - pos_pred[:, 0].min(),
                           pos_pred[:, 1].max() - pos_pred[:, 1].min(),
                           pos_pred[:, 2].max() - pos_pred[:, 2].min()])
    
    # If ranges differ by more than 10x, use separate limits to avoid making one invisible
    use_separate_limits = np.any(pred_range > 10 * true_range) or np.any(true_range > 10 * pred_range)
    
    if use_separate_limits:
        print(f"WARNING: Predicted positions are in a very different range than true positions.")
        print(f"  True range: {true_range}")
        print(f"  Pred range: {pred_range}")
        print(f"  Using separate axis limits for each subplot.")
        
        # True mesh: use true position limits
        x_range_true = pos_true[:, 0].max() - pos_true[:, 0].min()
        y_range_true = pos_true[:, 1].max() - pos_true[:, 1].min()
        z_range_true = pos_true[:, 2].max() - pos_true[:, 2].min()
        padding = 0.1
        ax1.set_xlim([pos_true[:, 0].min() - padding * x_range_true, pos_true[:, 0].max() + padding * x_range_true])
        ax1.set_ylim([pos_true[:, 1].min() - padding * y_range_true, pos_true[:, 1].max() + padding * y_range_true])
        ax1.set_zlim([pos_true[:, 2].min() - padding * z_range_true, pos_true[:, 2].max() + padding * z_range_true])
        
        # Predicted mesh: use predicted position limits
        x_range_pred = pos_pred[:, 0].max() - pos_pred[:, 0].min()
        y_range_pred = pos_pred[:, 1].max() - pos_pred[:, 1].min()
        z_range_pred = pos_pred[:, 2].max() - pos_pred[:, 2].min()
        ax2.set_xlim([pos_pred[:, 0].min() - padding * x_range_pred, pos_pred[:, 0].max() + padding * x_range_pred])
        ax2.set_ylim([pos_pred[:, 1].min() - padding * y_range_pred, pos_pred[:, 1].max() + padding * y_range_pred])
        ax2.set_zlim([pos_pred[:, 2].min() - padding * z_range_pred, pos_pred[:, 2].max() + padding * z_range_pred])
    else:
        # Set same axis limits for comparison when ranges are similar
        all_x = np.concatenate([pos_true[:, 0], pos_pred[:, 0]])
        all_y = np.concatenate([pos_true[:, 1], pos_pred[:, 1]])
        all_z = np.concatenate([pos_true[:, 2], pos_pred[:, 2]])
        
        # Add some padding to axis limits
        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()
        z_range = all_z.max() - all_z.min()
        padding = 0.1  # 10% padding
        
        for ax in [ax1, ax2]:
            ax.set_xlim([all_x.min() - padding * x_range, all_x.max() + padding * x_range])
            ax.set_ylim([all_y.min() - padding * y_range, all_y.max() + padding * y_range])
            ax.set_zlim([all_z.min() - padding * z_range, all_z.max() + padding * z_range])
    
    # Set equal aspect ratio and viewing angle for both subplots
    for ax in [ax1, ax2]:
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig


def load_cells_from_trajectory(data_dir, traj_idx):
    """Load cells (connectivity) from a trajectory in the dataset."""
    train_tfrecord = os.path.join(data_dir, 'train.tfrecord')
    meta_path = os.path.join(data_dir, 'meta.json')
    
    if not os.path.exists(train_tfrecord):
        raise FileNotFoundError(f"Training data not found: {train_tfrecord}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    traj_dict = load_raw_trajectory_from_tfrecord(train_tfrecord, meta, traj_idx)
    cells = traj_dict['cells']  # (C, 4)
    return cells.astype(int)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize EGNN predictions from checkpoint directory'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints/egnn/predictions',
        help='Directory containing prediction files (e.g., checkpoints/egnn/predictions)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        required=True,
        help='Epoch number to visualize (e.g., 1, 2, 5)'
    )
    parser.add_argument(
        '--traj_idx',
        type=int,
        default=0,
        help='Trajectory index (default: 0)'
    )
    parser.add_argument(
        '--t',
        type=int,
        required=True,
        help='Time index to visualize (0-indexed within trajectory)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/deforming_plate',
        help='Directory containing dataset (for loading cells)'
    )
    parser.add_argument(
        '--color',
        type=str,
        default='stress',
        choices=['stress', 'velocity', 'none', 'position_norm'],
        help='Color mode for visualization'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (if None, displays interactively)'
    )
    
    args = parser.parse_args()
    
    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    epoch_dir = os.path.join(args.checkpoint_dir, f'epoch_{args.epoch}')
    
    if not os.path.exists(epoch_dir):
        raise FileNotFoundError(f"Epoch directory not found: {epoch_dir}")
    
    print(f"Loading predictions from: {epoch_dir}")
    print(f"Trajectory index: {args.traj_idx}, Timestep: {args.t}")
    
    # Load predictions (these are lists of arrays, one per trajectory)
    pos_true_list = load_npy(os.path.join(epoch_dir, 'true_positions.npy'))
    pos_pred_list = load_npy(os.path.join(epoch_dir, 'predictions_positions.npy'))
    
    # Handle both lists, object arrays, and regular arrays
    # Check if it's a list or object array (both support len() and indexing)
    if isinstance(pos_true_list, list) or (hasattr(pos_true_list, 'dtype') and pos_true_list.dtype == object):
        # List or object array: list of arrays, one per trajectory
        num_trajectories = len(pos_true_list)
        if args.traj_idx >= num_trajectories:
            raise ValueError(
                f"Trajectory index {args.traj_idx} out of range. "
                f"Available trajectories: 0-{num_trajectories-1}"
            )
        pos_true_seq = pos_true_list[args.traj_idx]  # (T-1, N, 3)
        pos_pred_seq = pos_pred_list[args.traj_idx]   # (T-1, N, 3)
    else:
        # Regular array: shape is (num_traj, T-1, N, 3)
        num_trajectories = pos_true_list.shape[0]
        if args.traj_idx >= num_trajectories:
            raise ValueError(
                f"Trajectory index {args.traj_idx} out of range. "
                f"Available trajectories: 0-{num_trajectories-1}"
            )
        pos_true_seq = pos_true_list[args.traj_idx]  # (T-1, N, 3)
        pos_pred_seq = pos_pred_list[args.traj_idx]   # (T-1, N, 3)
    
    # Check timestep index
    if args.t >= pos_true_seq.shape[0]:
        raise ValueError(
            f"Timestep {args.t} out of range. "
            f"Available timesteps: 0-{pos_true_seq.shape[0]-1}"
        )
    
    # Get cells from dataset
    try:
        cells = load_cells_from_trajectory(args.data_dir, args.traj_idx)
        print(f"Loaded {len(cells)} cells from trajectory {args.traj_idx}")
    except Exception as e:
        print(f"Warning: Could not load cells: {e}")
        print("Visualization will use scatter plot instead of mesh")
        cells = None
    
    # Optional stress
    stress_true_seq = None
    stress_pred_seq = None
    if args.color == 'stress':
        stress_true_list = load_npy(os.path.join(epoch_dir, 'true_stress.npy'))
        stress_pred_list = load_npy(os.path.join(epoch_dir, 'predictions_stress.npy'))
        # Handle both lists, object arrays, and regular arrays
        if isinstance(stress_true_list, list) or (hasattr(stress_true_list, 'dtype') and stress_true_list.dtype == object):
            stress_true_seq = stress_true_list[args.traj_idx]  # (T-1, N, 1)
            stress_pred_seq = stress_pred_list[args.traj_idx]   # (T-1, N, 1)
        else:
            stress_true_seq = stress_true_list[args.traj_idx]  # (T-1, N, 1)
            stress_pred_seq = stress_pred_list[args.traj_idx]   # (T-1, N, 1)
    
    # Optional velocity
    vel_true_seq = None
    vel_pred_seq = None
    if args.color == 'velocity':
        vel_true_list = load_npy(os.path.join(epoch_dir, 'true_velocity.npy'))
        vel_pred_list = load_npy(os.path.join(epoch_dir, 'predictions_velocity.npy'))
        # Handle both lists, object arrays, and regular arrays
        if isinstance(vel_true_list, list) or (hasattr(vel_true_list, 'dtype') and vel_true_list.dtype == object):
            vel_true_seq = vel_true_list[args.traj_idx]  # (T-1, N, 3)
            vel_pred_seq = vel_pred_list[args.traj_idx]   # (T-1, N, 3)
        else:
            vel_true_seq = vel_true_list[args.traj_idx]  # (T-1, N, 3)
            vel_pred_seq = vel_pred_list[args.traj_idx]   # (T-1, N, 3)
    
    # -------------------------------------------------
    # SELECT TIME STEP
    # -------------------------------------------------
    # Debug: print shapes
    print(f"DEBUG: pos_true_seq shape: {pos_true_seq.shape if hasattr(pos_true_seq, 'shape') else type(pos_true_seq)}")
    print(f"DEBUG: pos_pred_seq shape: {pos_pred_seq.shape if hasattr(pos_pred_seq, 'shape') else type(pos_pred_seq)}")
    
    t = args.t
    pos_true = pos_true_seq[t]  # (N, 3)
    pos_pred = pos_pred_seq[t]  # (N, 3)
    
    # Debug: print extracted shapes
    print(f"DEBUG: pos_true shape after indexing [t={t}]: {pos_true.shape if hasattr(pos_true, 'shape') else type(pos_true)}")
    print(f"DEBUG: pos_pred shape after indexing [t={t}]: {pos_pred.shape if hasattr(pos_pred, 'shape') else type(pos_pred)}")
    
    # Choose intensity field
    intensity_true = None
    intensity_pred = None
    
    if args.color == 'stress':
        if stress_true_seq is None or stress_pred_seq is None:
            raise ValueError("Stress files required for color='stress'")
        intensity_true = stress_true_seq[t]  # (N, 1) or (N,)
        intensity_pred = stress_pred_seq[t]  # (N, 1) or (N,)
        # Ensure 1D array for coloring
        if len(intensity_true.shape) > 1:
            intensity_true = intensity_true.squeeze()
        if len(intensity_pred.shape) > 1:
            intensity_pred = intensity_pred.squeeze()
        print(f"Stress range - True: [{intensity_true.min():.2f}, {intensity_true.max():.2f}], "
              f"Pred: [{intensity_pred.min():.2f}, {intensity_pred.max():.2f}]")
    elif args.color == 'velocity':
        if vel_true_seq is None or vel_pred_seq is None:
            raise ValueError("Velocity files required for color='velocity'")
        v_true = vel_true_seq[t]  # (N, 3)
        v_pred = vel_pred_seq[t]  # (N, 3)
        intensity_true = np.linalg.norm(v_true, axis=1)  # (N,)
        intensity_pred = np.linalg.norm(v_pred, axis=1)  # (N,)
    elif args.color == 'position_norm':
        intensity_true = np.linalg.norm(pos_true, axis=1)  # (N,)
        intensity_pred = np.linalg.norm(pos_pred, axis=1)  # (N,)
    # else: "none" - intensity remains None
    
    # -------------------------------------------------
    # VISUALIZE
    # -------------------------------------------------
    fig = visualize_mesh_pair(
        pos_true=pos_true,
        pos_pred=pos_pred,
        stress_true=intensity_true,
        stress_pred=intensity_pred,
        cells=cells,
        color_mode=args.color if args.color != 'none' else 'stress',
        title_true=f"True (t={t}, traj={args.traj_idx})",
        title_pred=f"Prediction (t={t}, traj={args.traj_idx})",
    )
    
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {args.output}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()
