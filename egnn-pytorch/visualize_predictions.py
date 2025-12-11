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
        
        # Debug: print stress ranges with detailed statistics
        true_min, true_max = intensity_true.min(), intensity_true.max()
        true_mean, true_std = intensity_true.mean(), intensity_true.std()
        pred_min, pred_max = intensity_pred.min(), intensity_pred.max()
        pred_mean, pred_std = intensity_pred.mean(), intensity_pred.std()
        
        print(f"Stress ranges - True: min={true_min:.6f}, max={true_max:.6f}, "
              f"mean={true_mean:.6f}, std={true_std:.6f}, non-zero count={np.count_nonzero(intensity_true)}/{len(intensity_true)}")
        print(f"Stress ranges - Pred: min={pred_min:.6f}, max={pred_max:.6f}, "
              f"mean={pred_mean:.6f}, std={pred_std:.6f}, non-zero count={np.count_nonzero(intensity_pred)}/{len(intensity_pred)}")
        
        # Check if true stress is all zero or very small
        if true_max < 1e-6:
            print(f"  ⚠ WARNING: Ground truth stress is essentially zero (max={true_max:.6f}). "
                  f"This is normal for early timesteps before deformation starts.")
        elif true_std < 1e-6:
            print(f"  ⚠ WARNING: Ground truth stress has no variation (std={true_std:.6f}). "
                  f"All values are approximately {true_mean:.6f}.")
        else:
            # Show percentiles for better understanding of distribution
            true_p25, true_p50, true_p75 = np.percentile(intensity_true, [25, 50, 75])
            print(f"  ✓ Ground truth stress distribution: p25={true_p25:.2f}, p50={true_p50:.2f}, p75={true_p75:.2f}")
        
        # Use the SAME color scale for both true and predicted stress
        # This allows direct comparison between true and predicted values
        vmin_shared = min(true_min, pred_min)
        vmax_shared = max(true_max, pred_max)
        
        # If all values are the same, add a small range to make colormap work
        if abs(vmax_shared - vmin_shared) < 1e-10:
            print(f"Warning: All stress values are nearly identical ({vmin_shared:.6f}). Using small range for colormap.")
            vmin_shared = vmin_shared - abs(vmin_shared) * 0.01 if abs(vmin_shared) > 1e-10 else -1.0
            vmax_shared = vmax_shared + abs(vmax_shared) * 0.01 if abs(vmax_shared) > 1e-10 else 1.0
        
        # Use shared scale for both
        vmin_true = vmin_pred = vmin_shared
        vmax_true = vmax_pred = vmax_shared
        
        print(f"Using SHARED colorbar range: [{vmin_shared:.2f}, {vmax_shared:.2f}]")
        
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
            print(f"DEBUG: intensity_true is None: {intensity_true is None}, len(face_colors)={len(face_colors)}, len(faces)={len(faces)}")
            if intensity_true is not None:
                print(f"DEBUG: intensity_true length={len(intensity_true)}, pos_true length={len(pos_true)}")
                if len(face_colors) > 0:
                    print(f"DEBUG: face_colors sample (first 5): {face_colors[:5]}")
            
            if len(faces) > 0:
                # Create Poly3DCollection
                if intensity_true is not None and len(face_colors) == len(faces):
                    # Map stress values to colors using colormap with TRUE scale
                    # Ensure vmin != vmax to avoid normalization issues
                    if abs(vmax_true - vmin_true) < 1e-10:
                        print(f"DEBUG: vmin_true == vmax_true ({vmin_true:.6f}), adjusting range")
                        vmax_true = vmin_true + 1.0 if abs(vmin_true) < 1e-6 else vmin_true * 1.01
                    
                    norm_true = mcolors.Normalize(vmin=vmin_true, vmax=vmax_true)
                    try:
                        colormap = plt.colormaps[cmap]
                    except (KeyError, AttributeError):
                        colormap = cm.get_cmap(cmap)
                    # Map colors and ensure they're in RGBA format
                    face_colors_mapped = []
                    for c in face_colors:
                        # Clamp value to [vmin, vmax] range
                        c_clamped = np.clip(c, vmin_true, vmax_true)
                        rgba = colormap(norm_true(c_clamped))
                        # Ensure it's a tuple/list of 4 values (RGBA)
                        if isinstance(rgba, tuple) and len(rgba) == 4:
                            face_colors_mapped.append(rgba)
                        elif isinstance(rgba, np.ndarray):
                            face_colors_mapped.append(tuple(rgba))
                        else:
                            # Convert to RGBA tuple
                            face_colors_mapped.append((rgba[0], rgba[1], rgba[2], 1.0))
                    
                    print(f"Created {len(faces)} faces with stress coloring (range: {min(face_colors):.6f} to {max(face_colors):.6f}, scale: {vmin_true:.6f} to {vmax_true:.6f})")
                    print(f"DEBUG: Mapped {len(face_colors_mapped)} colors, first color sample: {face_colors_mapped[0] if len(face_colors_mapped) > 0 else 'N/A'}")
                    
                    # Use stress for coloring - create collection first, then set colors
                    face_collection = Poly3DCollection(faces, alpha=0.9, linewidths=0.0, edgecolors='none')
                    face_collection.set_facecolor(face_colors_mapped)
                    ax1.add_collection3d(face_collection)
                    
                    # Create a ScalarMappable for the colorbar
                    sm = cm.ScalarMappable(cmap=colormap, norm=norm_true)
                    sm.set_array([])
                    cbar1 = plt.colorbar(sm, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                    print(f"DEBUG: Added Poly3DCollection with {len(face_collection._facecolors)} face colors (continuous mesh)")
                elif intensity_true is not None and len(face_colors) > 0:
                    # face_colors exists but length mismatch - try to use it anyway or fallback
                    print(f"WARNING: face_colors length ({len(face_colors)}) != faces length ({len(faces)}), attempting to use available colors")
                    # Try to use what we have
                    if len(face_colors) >= len(faces):
                        # Use first len(faces) colors
                        face_colors_use = face_colors[:len(faces)]
                    else:
                        # Repeat last color or use mean
                        mean_color = np.mean(face_colors) if len(face_colors) > 0 else 0.0
                        face_colors_use = list(face_colors) + [mean_color] * (len(faces) - len(face_colors))
                    
                    # Map colors
                    if abs(vmax_true - vmin_true) < 1e-10:
                        vmax_true = vmin_true + 1.0 if abs(vmin_true) < 1e-6 else vmin_true * 1.01
                    norm_true = mcolors.Normalize(vmin=vmin_true, vmax=vmax_true)
                    try:
                        colormap = plt.colormaps[cmap]
                    except (KeyError, AttributeError):
                        colormap = cm.get_cmap(cmap)
                    face_colors_mapped = [tuple(colormap(norm_true(np.clip(c, vmin_true, vmax_true)))) for c in face_colors_use]
                    face_collection = Poly3DCollection(faces, alpha=0.9, linewidths=0.0, edgecolors='none')
                    face_collection.set_facecolor(face_colors_mapped)
                    ax1.add_collection3d(face_collection)
                    sm = cm.ScalarMappable(cmap=colormap, norm=norm_true)
                    sm.set_array([])
                    cbar1 = plt.colorbar(sm, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                else:
                    # No coloring, use uniform color
                    print(f"WARNING: Cannot use stress coloring - intensity_true is None: {intensity_true is None}, face_colors length: {len(face_colors)}")
                    face_collection = Poly3DCollection(faces, alpha=1.0, linewidths=0.1, edgecolors='none', facecolors='lightblue')
                    ax1.add_collection3d(face_collection)
            else:
                print("Warning: No valid faces created, using uniform color mesh")
                # Use uniform color if face colors don't match
                face_collection = Poly3DCollection(faces, alpha=1.0, linewidths=0.1, edgecolors='none', facecolors='lightblue')
                ax1.add_collection3d(face_collection)
                if intensity_true is not None:
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm_true)
                    sm.set_array([])
                    cbar1 = plt.colorbar(sm, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
        except Exception as e:
            import traceback
            print(f"ERROR: Could not plot mesh faces: {e}")
            traceback.print_exc()
            # Fallback to scatter plot if mesh fails
            print("Falling back to scatter plot visualization")
            # Check if we have intensity data (defined in outer scope)
            if 'intensity_true' in locals() and intensity_true is not None:
                # Use the same colormap and scale as defined above
                scatter_cmap = cmap if 'cmap' in locals() else 'viridis'
                scatter_vmin = vmin_true if 'vmin_true' in locals() else intensity_true.min()
                scatter_vmax = vmax_true if 'vmax_true' in locals() else intensity_true.max()
                scatter1 = ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                                      c=intensity_true, cmap=scatter_cmap, s=50, alpha=1.0,
                                      edgecolors='black', linewidths=0.2, vmin=scatter_vmin, vmax=scatter_vmax)
                cbar1 = plt.colorbar(scatter1, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                print(f"DEBUG: Using scatter plot for true stress (N={len(pos_true)} points)")
            elif color_mode != "none" and stress_true is not None:
                # Try to use stress_true directly
                intensity_fallback = stress_true.flatten() if len(stress_true.shape) > 1 else stress_true
                scatter1 = ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                                      c=intensity_fallback, cmap='viridis', s=50, alpha=1.0,
                                      edgecolors='black', linewidths=0.2)
                cbar1 = plt.colorbar(scatter1, ax=ax1, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                print(f"DEBUG: Using scatter plot with fallback intensity (N={len(pos_true)} points)")
            else:
                ax1.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2], 
                           c='lightblue', s=50, alpha=1.0, edgecolors='black', linewidths=0.2)
                print("DEBUG: No intensity data available, using uniform color")
    else:
        # No cells available - cannot create continuous mesh
        raise ValueError("Cells are required for continuous mesh visualization. Please provide cells data.")
    
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
                    
                    # Create a ScalarMappable for the colorbar
                    sm = cm.ScalarMappable(cmap=colormap, norm=norm_pred)
                    sm.set_array([])
                    cbar2 = plt.colorbar(sm, ax=ax2, label='Stress' if color_mode == 'stress' else 'Intensity', shrink=0.8)
                    print(f"DEBUG: Added Poly3DCollection with {len(face_collection._facecolors)} face colors (continuous mesh, predicted)")
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
        default=100,
        help='Epoch number to visualize (e.g., 1, 2, 5). Default: 1'
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
        default=None,
        help='Time index to visualize (0-indexed within trajectory). If not provided, will visualize timesteps 0, 10, 20, ..., 100'
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
    parser.add_argument(
        '--scatter_plot',
        action='store_true',
        help='Generate scatter plot of predicted vs true stress (in addition to mesh visualization)'
    )
    
    args = parser.parse_args()
    
    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    # Resolve checkpoint directory path (handle relative paths)
    if not os.path.isabs(args.checkpoint_dir):
        # If relative path, resolve relative to script location or current working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try relative to script directory first, then current working directory
        predictions_base_dir = os.path.join(script_dir, args.checkpoint_dir)
        if not os.path.exists(predictions_base_dir):
            predictions_base_dir = os.path.abspath(args.checkpoint_dir)
    else:
        predictions_base_dir = args.checkpoint_dir
    
    # Find the epoch directory
    epoch_dir = os.path.join(predictions_base_dir, f'epoch_{args.epoch}')
    
    if not os.path.exists(epoch_dir):
        # Try to find available epochs
        if os.path.exists(predictions_base_dir):
            available_epochs = []
            for item in os.listdir(predictions_base_dir):
                if item.startswith('epoch_') and os.path.isdir(os.path.join(predictions_base_dir, item)):
                    try:
                        epoch_num = int(item.split('_')[1])
                        available_epochs.append(epoch_num)
                    except (ValueError, IndexError):
                        continue
            if available_epochs:
                available_epochs.sort()
                raise FileNotFoundError(
                    f"Epoch directory not found: {epoch_dir}\n"
                    f"Searched in: {predictions_base_dir}\n"
                    f"Available epochs: {available_epochs}\n"
                    f"Use --epoch to specify one of these epochs."
                )
        raise FileNotFoundError(
            f"Epoch directory not found: {epoch_dir}\n"
            f"Base directory: {predictions_base_dir} (exists: {os.path.exists(predictions_base_dir)})"
        )
    
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
    
    # Check timestep index (only if a specific timestep is provided)
    if args.t is not None and args.t >= pos_true_seq.shape[0]:
        raise ValueError(
            f"Timestep {args.t} out of range. "
            f"Available timesteps: 0-{pos_true_seq.shape[0]-1}"
        )
    
    # Get cells from dataset
    # Resolve data_dir path (handle relative paths)
    if not os.path.isabs(args.data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try relative to script directory first, then current working directory
        data_dir = os.path.join(script_dir, args.data_dir)
        if not os.path.exists(data_dir):
            data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = args.data_dir
    
    try:
        cells = load_cells_from_trajectory(data_dir, args.traj_idx)
        print(f"Loaded {len(cells)} cells from trajectory {args.traj_idx}")
    except Exception as e:
        import traceback
        print(f"ERROR: Could not load cells from {data_dir}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("\nCells are required for continuous mesh visualization.")
        print(f"Please ensure:")
        print(f"  1. Data directory exists: {data_dir}")
        print(f"  2. train.tfrecord and meta.json are in the data directory")
        print(f"  3. Trajectory {args.traj_idx} exists in the dataset")
        print(f"\nYou can specify a different data directory with --data_dir")
        raise
    
    # Optional stress
    stress_true_seq = None
    stress_pred_seq = None
    if args.color == 'stress':
        stress_true_path = os.path.join(epoch_dir, 'true_stress.npy')
        stress_pred_path = os.path.join(epoch_dir, 'predictions_stress.npy')
        print(f"Loading stress data from:")
        print(f"  True: {stress_true_path}")
        print(f"  Pred: {stress_pred_path}")
        
        if not os.path.exists(stress_true_path):
            raise FileNotFoundError(f"True stress file not found: {stress_true_path}")
        if not os.path.exists(stress_pred_path):
            raise FileNotFoundError(f"Predicted stress file not found: {stress_pred_path}")
        
        stress_true_list = load_npy(stress_true_path)
        stress_pred_list = load_npy(stress_pred_path)
        
        print(f"Loaded stress data - true_list type: {type(stress_true_list)}, pred_list type: {type(stress_pred_list)}")
        
        # Handle both lists, object arrays, and regular arrays
        if isinstance(stress_true_list, list) or (hasattr(stress_true_list, 'dtype') and stress_true_list.dtype == object):
            num_traj = len(stress_true_list)
            print(f"Stress data is list/object array with {num_traj} trajectories")
            if args.traj_idx >= num_traj:
                raise ValueError(f"Trajectory index {args.traj_idx} out of range (0-{num_traj-1})")
            stress_true_seq = stress_true_list[args.traj_idx]  # (T-1, N, 1)
            stress_pred_seq = stress_pred_list[args.traj_idx]   # (T-1, N, 1)
        else:
            num_traj = stress_true_list.shape[0]
            print(f"Stress data is regular array with shape {stress_true_list.shape}, {num_traj} trajectories")
            if args.traj_idx >= num_traj:
                raise ValueError(f"Trajectory index {args.traj_idx} out of range (0-{num_traj-1})")
            stress_true_seq = stress_true_list[args.traj_idx]  # (T-1, N, 1)
            stress_pred_seq = stress_pred_list[args.traj_idx]   # (T-1, N, 1)
        
        print(f"Extracted stress sequence - true shape: {stress_true_seq.shape if hasattr(stress_true_seq, 'shape') else type(stress_true_seq)}, "
              f"pred shape: {stress_pred_seq.shape if hasattr(stress_pred_seq, 'shape') else type(stress_pred_seq)}")
    
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
    # SELECT TIME STEPS
    # -------------------------------------------------
    # Debug: print shapes
    print(f"DEBUG: pos_true_seq shape: {pos_true_seq.shape if hasattr(pos_true_seq, 'shape') else type(pos_true_seq)}")
    print(f"DEBUG: pos_pred_seq shape: {pos_pred_seq.shape if hasattr(pos_pred_seq, 'shape') else type(pos_pred_seq)}")
    
    # Determine which timesteps to visualize
    if args.t is not None:
        # Single timestep specified
        timesteps_to_visualize = [args.t]
    else:
        # Default: visualize all available timesteps
        max_t = len(pos_true_seq) - 1  # Use all available timesteps
        timesteps_to_visualize = list(range(0, max_t + 1, 5))  # 0, 1, 2, ..., max_t
    
    print(f"Visualizing timesteps: {timesteps_to_visualize}")
    
    # Create output directory if saving files
    output_dir = None
    if args.output:
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Collect stress data for scatter plot if requested
    all_stress_true = []
    all_stress_pred = []
    
    # Visualize each timestep
    for t in timesteps_to_visualize:
        if t >= len(pos_true_seq):
            print(f"Skipping timestep {t} (exceeds available timesteps: {len(pos_true_seq)})")
            continue
        
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
            
            # Verify data integrity
            if not isinstance(intensity_true, np.ndarray):
                intensity_true = np.asarray(intensity_true)
            if not isinstance(intensity_pred, np.ndarray):
                intensity_pred = np.asarray(intensity_pred)
            
            # CRITICAL DEBUG: Print detailed statistics to diagnose uniform predictions
            print(f"\n=== Stress Data for t={t}, traj={args.traj_idx} ===")
            print(f"True stress shape: {intensity_true.shape}, dtype: {intensity_true.dtype}")
            print(f"Pred stress shape: {intensity_pred.shape}, dtype: {intensity_pred.dtype}")
            print(f"True stress - min={intensity_true.min():.6f}, max={intensity_true.max():.6f}, "
                  f"mean={intensity_true.mean():.6f}, std={intensity_true.std():.6f}, "
                  f"non-zero={np.count_nonzero(intensity_true)}/{len(intensity_true)}")
            print(f"Pred stress - min={intensity_pred.min():.6f}, max={intensity_pred.max():.6f}, "
                  f"mean={intensity_pred.mean():.6f}, std={intensity_pred.std():.6f}, "
                  f"non-zero={np.count_nonzero(intensity_pred)}/{len(intensity_pred)}")
            
            # Check for potential issues
            if np.all(intensity_true == 0):
                print(f"  ⚠ Ground truth stress is ALL ZERO - this is correct for early timesteps before deformation")
            elif np.allclose(intensity_true, intensity_true[0], atol=1e-6):
                print(f"  ⚠ Ground truth stress is UNIFORM (all values ≈ {intensity_true[0]:.6f})")
            else:
                # Show sample of unique values to verify variation
                unique_vals = np.unique(intensity_true)
                if len(unique_vals) <= 10:
                    print(f"  ✓ Ground truth stress has {len(unique_vals)} unique values: {unique_vals[:10]}")
                else:
                    print(f"  ✓ Ground truth stress has {len(unique_vals)} unique values (showing range)")
            
            # Check if predictions are uniform
            pred_range = intensity_pred.max() - intensity_pred.min()
            true_range = intensity_true.max() - intensity_true.min()
            pred_std = intensity_pred.std()
            true_std = intensity_true.std()
            
            if pred_range < 1e-6 or pred_std < 1e-6:
                print(f"  ⚠ CRITICAL: Predictions are UNIFORM in saved files!")
                print(f"     This confirms the problem is in MODEL PREDICTION, not visualization.")
            elif pred_range < 0.1 * true_range or pred_std < 0.1 * true_std:
                print(f"  ⚠ WARNING: Predictions show very little variation")
                print(f"     Range ratio: {pred_range/(true_range+1e-8):.4f}, Std ratio: {pred_std/(true_std+1e-8):.4f}")
            else:
                print(f"  ✓ Predictions show variation (range ratio: {pred_range/(true_range+1e-8):.4f})")
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
        
        # Collect stress data for scatter plot (if stress data is available)
        if args.scatter_plot and stress_true_seq is not None and stress_pred_seq is not None:
            stress_true_t = stress_true_seq[t]
            stress_pred_t = stress_pred_seq[t]
            if len(stress_true_t.shape) > 1:
                stress_true_t = stress_true_t.squeeze()
            if len(stress_pred_t.shape) > 1:
                stress_pred_t = stress_pred_t.squeeze()
            all_stress_true.append(stress_true_t)
            all_stress_pred.append(stress_pred_t)
        
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
        
        # Determine output path
        if args.output:
            # If single timestep specified, use provided path
            if args.t is not None:
                output_path = args.output
            else:
                # Multiple timesteps: add timestep to filename
                base, ext = os.path.splitext(args.output)
                output_path = f"{base}_t_{t:03d}{ext}"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    # Generate aggregated scatter plot if requested
    if args.scatter_plot and len(all_stress_true) > 0:
        # Concatenate all timesteps
        stress_true_all = np.concatenate(all_stress_true)
        stress_pred_all = np.concatenate(all_stress_pred)
        
        scatter_output = None
        if args.output:
            # Create scatter plot filename
            if args.t is not None:
                base, ext = os.path.splitext(args.output)
            else:
                # For multiple timesteps, use the base output name
                base, ext = os.path.splitext(args.output)
            scatter_output = f"{base}_stress_scatter{ext if ext else '.png'}"
        
        plot_stress_scatter(
            stress_true_all,
            stress_pred_all,
            title=f"Stress Prediction vs True (traj={args.traj_idx}, timesteps {timesteps_to_visualize[0]}-{timesteps_to_visualize[-1]})",
            output_path=scatter_output
        )


if __name__ == "__main__":
    main()
