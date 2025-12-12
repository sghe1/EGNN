#!/usr/bin/env python3
"""
Training script for EGNN on deforming plate dataset.
Uses data_loader_egnn.py to load data and MeshEGNN model.

Dataset Structure:
- Input features: [position(3), actuation(3), node_type_one_hot(2)] = 8 dims
- Targets: velocity (T, N, 3), stress (T, N, 1)
- Coordinates: world_pos (T, N, 3) - same as position in features

Normalization Pipeline (MeshGraphNet-style):
- ALL inputs are normalized BEFORE going into the model:
  * Positions: (pos - pos_mean) / pos_std
  * Actuation: (act - act_mean) / act_std
  * Node type: unchanged (one-hot)
- ALL targets are normalized BEFORE computing loss:
  * Velocity: vel / vel_std
  * Stress: (stress - stress_mean) / stress_std
- Model operates entirely in normalized space (O(1) magnitudes)
- Denormalization happens ONLY when saving predictions for visualization

Training Process:
- For each timestep t, model predicts normalized velocity and stress at time t
- Loss is computed on normalized values
- Denormalization only for saving predictions
"""

import os
import sys
import json
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from tfrecord.reader import tfrecord_loader

# Note: Velocities and stress are normalized by their means in the dataset
# No additional scaling is needed since normalization brings values to O(1) range

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
egnn_pytorch_dir = os.path.join(script_dir, 'egnn-pytorch')
sys.path.insert(0, script_dir)
sys.path.insert(0, egnn_pytorch_dir)

from data_loader_egnn import data_loader_egnn, trajectory_to_egnn_inputs, load_raw_trajectory_from_tfrecord, build_edges_from_cells
from tfrecord.reader import tfrecord_loader

# Import EGNN - use the correct implementation from myEGNN
from EGNN import MeshEGNN


class DeformingPlateDataset(Dataset):
    """Dataset for deforming plate trajectories."""
    
    def __init__(self, tfrecord_path, meta_path, num_trajectories=None, dataset_fraction=1.0, 
                 norm_stats_path=None, compute_norm_stats=True):
        """
        Dataset for deforming plate trajectories with MeshGraphNet-style normalization.
        
        All inputs (positions, actuation) and targets (velocity, stress) are normalized
        before being fed to the model. Denormalization happens only when saving predictions.
        
        Args:
            tfrecord_path: Path to TFRecord file
            meta_path: Path to meta.json
            num_trajectories: Number of trajectories to load (None = use dataset_fraction)
            dataset_fraction: Fraction of dataset to use (0.0 to 1.0)
            norm_stats_path: Path to JSON file with normalization statistics (if None, compute)
            compute_norm_stats: Whether to compute normalization statistics (if not loading from file)
        """
        self.tfrecord_path = tfrecord_path
        self.meta_path = meta_path
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # Determine number of trajectories
        if num_trajectories is None:
            if dataset_fraction is None:
                # If both are None, default to 1 trajectory
                num_trajectories = 1
            else:
                # Estimate from dataset - typically 1000 for deforming_plate
                estimated_total = 1000
                num_trajectories = int(estimated_total * dataset_fraction)
        
        self.requested_trajectories = max(1, num_trajectories)
        self.available_trajectories = self._count_available_trajectories(
            max_count=self.requested_trajectories
        )
        
        if self.available_trajectories == 0:
            raise ValueError(
                f"No trajectories found in TFRecord: {tfrecord_path}\n"
                "Ensure the dataset was uploaded/extracted correctly and the path is valid."
            )
        
        if self.available_trajectories < self.requested_trajectories:
            print(
                f"Requested {self.requested_trajectories} trajectories but only "
                f"{self.available_trajectories} available in {tfrecord_path}. "
                "Using the available count."
            )
        
        self.num_trajectories = min(self.requested_trajectories, self.available_trajectories)
        print(f"Loading {self.num_trajectories} trajectories from {tfrecord_path}")
        
        # Load or compute normalization statistics
        if norm_stats_path and os.path.exists(norm_stats_path):
            print(f"Loading normalization statistics from: {norm_stats_path}")
            self._load_norm_stats(norm_stats_path)
        elif compute_norm_stats:
            print("Computing normalization statistics...")
            self._compute_all_norm_stats()
            print("Normalization statistics computed:")
            print(f"  Position: mean={self.pos_mean.numpy()}, std={self.pos_std.numpy()}")
            print(f"  Actuation: mean={self.act_mean.numpy()}, std={self.act_std.numpy()}")
            print(f"  Velocity: std={self.vel_std:.6f} (scalar)")
            print(f"  Stress: mean={self.stress_mean:.6f}, std={self.stress_std:.6f}")
        else:
            # Default: no normalization (shouldn't be used in production)
            self._init_default_norm_stats()
    
    def _compute_all_norm_stats(self, sample_size=100):
        """
        Compute all normalization statistics following MeshGraphNet conventions.
        
        Computes:
        - pos_mean, pos_std: Per-component mean/std of positions (3,)
        - act_mean, act_std: Per-component mean/std of actuation (3,)
        - vel_std: Scalar std of velocity magnitudes
        - stress_mean, stress_std: Scalar mean/std of stress
        """
        import numpy as np
        all_positions = []
        all_actuation = []
        all_stress = []
        all_velocity = []
        
        # Sample trajectories to compute statistics
        sample_trajs = min(sample_size, self.available_trajectories)
        for idx in range(sample_trajs):
            traj_dict = load_raw_trajectory_from_tfrecord(
                self.tfrecord_path, self.meta, idx
            )
            world_pos = traj_dict['world_pos']  # (T, N, 3)
            stress = traj_dict['stress']  # (T, N, 1)
            actuation = traj_dict.get('actuation', None)
            
            # Collect positions (all timesteps)
            all_positions.append(world_pos.reshape(-1, 3))  # (T*N, 3)
            
            # Collect actuation (if available)
            if actuation is not None:
                if len(actuation.shape) == 2:  # (N, 3) - static
                    actuation = np.tile(actuation[np.newaxis, :, :], (world_pos.shape[0], 1, 1))
                all_actuation.append(actuation.reshape(-1, 3))  # (T*N, 3)
            
            # Collect stress - only from timesteps where mean stress > threshold
            # Only collect stress from timesteps where mean stress > 5000.0 (consistent with training threshold)
            for t in range(len(stress)):
                if stress[t].mean() > 5000.0:  # Only use high-stress timesteps
                    all_stress.append(stress[t].flatten())
            
            # Compute velocity (exclude t=0)
            vel = np.zeros_like(world_pos)
            if len(world_pos) > 1:
                vel[1:] = world_pos[1:] - world_pos[:-1]
            all_velocity.append(vel[1:].reshape(-1, 3))  # Skip t=0
        
        # Concatenate all samples
        all_positions = np.concatenate(all_positions, axis=0)  # (total_samples, 3)
        # Concatenate all stress values (only from high-stress timesteps)
        all_stress = np.concatenate(all_stress) if len(all_stress) > 0 else np.array([])  # (total_samples,) - only high-stress timesteps
        all_velocity = np.concatenate(all_velocity, axis=0)  # (total_samples, 3)
        
        # Position normalization: per-component mean and std
        self.pos_mean = torch.tensor(all_positions.mean(axis=0), dtype=torch.float32)  # (3,)
        self.pos_std = torch.tensor(all_positions.std(axis=0), dtype=torch.float32)  # (3,)
        # Avoid division by zero - use ddof=0 for population std (consistent with MeshGraphNet)
        if self.pos_std.min() < 1e-8:
            self.pos_std = torch.clamp(self.pos_std, min=1e-8)
        
        # Actuation normalization: per-component mean and std (or default if not available)
        if len(all_actuation) > 0:
            all_actuation = np.concatenate(all_actuation, axis=0)  # (total_samples, 3)
            self.act_mean = torch.tensor(all_actuation.mean(axis=0), dtype=torch.float32)  # (3,)
            self.act_std = torch.tensor(all_actuation.std(axis=0), dtype=torch.float32)  # (3,)
            self.act_std = torch.clamp(self.act_std, min=1e-8)
        else:
            # Actuation not available - use default (zero mean, unit std)
            self.act_mean = torch.zeros(3, dtype=torch.float32)
            self.act_std = torch.ones(3, dtype=torch.float32)
        
        # Velocity normalization: scalar std of velocity magnitudes
        velocity_magnitudes = np.linalg.norm(all_velocity, axis=1)  # (total_samples,)
        # Use 75th percentile for robustness
        vel_percentile = np.percentile(velocity_magnitudes, 75)
        self.vel_std = float(max(vel_percentile, 1e-5))  # Scalar, minimum 1e-5
        
        # Stress normalization: scalar mean and std
        # Only use stress from timesteps where mean stress > 5000.0 (consistent with training threshold)
        # This focuses normalization on high-stress regions where the model needs to be accurate
        if len(all_stress) > 0:
            stress_nonzero = all_stress[all_stress > 100.0]
            if len(stress_nonzero) > 0:
                self.stress_mean = float(np.median(stress_nonzero))  # Use median instead of mean
                self.stress_std = float(np.percentile(stress_nonzero, 75) - np.percentile(stress_nonzero, 25))  # IQR-based std
            else:
                self.stress_mean = 0.0
                self.stress_std = 1.0
            
            # Compute statistics for debugging
            print(f"Stress normalization computed from {len(all_stress)} samples (only timesteps with mean stress > 5000.0)")
            print(f"  Stress mean: {self.stress_mean:.2f}")
            print(f"  Stress std: {self.stress_std:.2f}")
            print(f"  Stress min: {all_stress.min():.2f}, max: {all_stress.max():.2f}")
        else:
            # Fallback: no stress data found
            print("  ⚠ WARNING: No stress values found! Using fallback normalization.")
            self.stress_mean = 0.0
            self.stress_std = 1.0
        # Ensure mean is non-negative (stress is always >= 0)
        if self.stress_mean < 0:
            self.stress_mean = 0.0  # Safety: stress cannot be negative
    
    def _init_default_norm_stats(self):
        """Initialize default normalization stats (no normalization)."""
        self.pos_mean = torch.zeros(3, dtype=torch.float32)
        self.pos_std = torch.ones(3, dtype=torch.float32)
        self.act_mean = torch.zeros(3, dtype=torch.float32)
        self.act_std = torch.ones(3, dtype=torch.float32)
        self.vel_std = 1.0
        self.stress_mean = 0.0
        self.stress_std = 1.0
    
    def _load_norm_stats(self, norm_stats_path):
        """Load normalization statistics from JSON file."""
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        
        self.pos_mean = torch.tensor(stats['pos_mean'], dtype=torch.float32)
        self.pos_std = torch.tensor(stats['pos_std'], dtype=torch.float32)
        self.act_mean = torch.tensor(stats['act_mean'], dtype=torch.float32)
        self.act_std = torch.tensor(stats['act_std'], dtype=torch.float32)
        self.vel_std = float(stats['vel_std'])
        self.stress_mean = float(stats['stress_mean'])
        self.stress_std = float(stats['stress_std'])
    
    def save_norm_stats(self, norm_stats_path):
        """Save normalization statistics to JSON file."""
        stats = {
            'pos_mean': self.pos_mean.tolist(),
            'pos_std': self.pos_std.tolist(),
            'act_mean': self.act_mean.tolist(),
            'act_std': self.act_std.tolist(),
            'vel_std': self.vel_std,
            'stress_mean': self.stress_mean,
            'stress_std': self.stress_std
        }
        os.makedirs(os.path.dirname(norm_stats_path), exist_ok=True)
        with open(norm_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved normalization statistics to: {norm_stats_path}")
    
    def _count_available_trajectories(self, max_count=None):
        """Count how many trajectories are in the TFRecord (up to max_count)."""
        try:
            loader = tfrecord_loader(self.tfrecord_path, index_path=None)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"TFRecord file not found: {self.tfrecord_path}. "
                "Upload or mount the deforming_plate dataset."
            ) from exc
        
        count = 0
        for _ in loader:
            count += 1
            if max_count is not None and count >= max_count:
                break
        
        return count
    
    def __len__(self):
        return self.num_trajectories
    
    def __getitem__(self, idx):
        """
        Load a trajectory and convert to EGNN format with normalized inputs.
        
        CRITICAL: All inputs (positions, actuation) are normalized BEFORE being fed to the model.
        Targets (velocity, stress) are returned in ORIGINAL scale (will be normalized in train_epoch).
        
        Returns:
            Dictionary with:
            - coors: (T, N, 3) - NORMALIZED node coordinates
            - feats: (T, N, 8) - NORMALIZED input features [pos_norm(3), act_norm(3), node_type_one_hot(2)]
            - adj_mat: (N, N) - adjacency matrix from mesh connectivity
            - edge_index: (E, 2) - edge connectivity
            - world_pos: (T, N, 3) - ORIGINAL world positions (for denormalization)
            - target_vel: (T, N, 3) - ORIGINAL target velocity (will be normalized in train_epoch)
            - target_stress: (T, N, 1) - ORIGINAL target stress (will be normalized in train_epoch)
        """
        if idx >= self.available_trajectories:
            raise IndexError(
                f"Trajectory index {idx} is out of range. "
                f"Only {self.available_trajectories} trajectories available in {self.tfrecord_path}."
            )
        traj_dict = load_raw_trajectory_from_tfrecord(
            self.tfrecord_path, self.meta, idx
        )
        
        # Get original data
        world_pos = torch.tensor(traj_dict['world_pos'], dtype=torch.float32)  # (T, N, 3) - ORIGINAL
        actuation = traj_dict.get('actuation', None)
        node_type = traj_dict['node_type']  # (N, 1)
        cells = traj_dict['cells']  # (C, 4)
        T, N, _ = world_pos.shape
        
        # Handle actuation
        if actuation is None:
            actuation = np.zeros((T, N, 3), dtype=np.float32)
        else:
            actuation = np.array(actuation, dtype=np.float32)
            if len(actuation.shape) == 2:  # (N, 3) - static
                actuation = np.tile(actuation[np.newaxis, :, :], (T, 1, 1))
        actuation = torch.tensor(actuation, dtype=torch.float32)  # (T, N, 3)
        
        # NORMALIZE INPUTS: Positions and actuation
        # Normalize positions: (pos - pos_mean) / pos_std
        coors_normalized = (world_pos - self.pos_mean.unsqueeze(0).unsqueeze(0)) / self.pos_std.unsqueeze(0).unsqueeze(0)  # (T, N, 3)
        
        # Normalize actuation: (act - act_mean) / act_std
        actuation_normalized = (actuation - self.act_mean.unsqueeze(0).unsqueeze(0)) / self.act_std.unsqueeze(0).unsqueeze(0)  # (T, N, 3)
        
        # Convert node_type to one-hot (not normalized)
        # According to meshgraphnets/common.py NodeType enum:
        # - NORMAL = 0 (plate nodes, where we compute loss)
        # - OBSTACLE = 1 (boundary nodes, no loss)
        # - HANDLE = 3 (actuator nodes, no loss)
        # Encoding: 0 -> [0, 0] (NORMAL/plate), 1 -> [1, 0] (OBSTACLE), 3 -> [0, 1] (HANDLE)
        # node_type == 0 (NORMAL) indicates nodes where we compute loss
        # node_type == 1 (OBSTACLE) or 3 (HANDLE) are boundary/actuator nodes (fixed, no loss)
        node_type_flat = node_type.flatten()  # (N,)
        node_type_one_hot = np.zeros((N, 2), dtype=np.float32)
        node_type_one_hot[node_type_flat == 0, :] = [0.0, 0.0]  # Type 0 -> [0, 0]
        node_type_one_hot[node_type_flat == 1, :] = [1.0, 0.0]  # Type 1 (plate) -> [1, 0]
        node_type_one_hot[node_type_flat == 3, :] = [0.0, 1.0]  # Type 3 -> [0, 1]
        node_type_one_hot = torch.tensor(node_type_one_hot, dtype=torch.float32)  # (N, 2)
        
        # Construct NORMALIZED features: [pos_norm(3), act_norm(3), node_type_one_hot(2)]
        feats_normalized = torch.cat([
            coors_normalized,           # (T, N, 3) - normalized positions
            actuation_normalized,       # (T, N, 3) - normalized actuation
            node_type_one_hot.unsqueeze(0).expand(T, -1, -1)  # (T, N, 2) - node type (not normalized)
        ], dim=-1)  # (T, N, 8)
        
        # Build edge connectivity
        edge_index = build_edges_from_cells(cells, num_nodes=N)
        num_nodes = coors_normalized.shape[1]
        adj_mat = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        if edge_index.shape[0] == 2:  # (2, E) format
            adj_mat[edge_index[0], edge_index[1]] = True
        else:  # (E, 2) format
            adj_mat[edge_index[:, 0], edge_index[:, 1]] = True
        adj_mat = adj_mat | adj_mat.t()  # Make symmetric
        
        # Compute targets in ORIGINAL scale (will be normalized in train_epoch)
        target_vel = compute_target_velocity(world_pos)  # (T, N, 3) - ORIGINAL
        target_stress = torch.tensor(traj_dict['stress'], dtype=torch.float32)  # (T, N, 1) - ORIGINAL
        
        # Verify shapes
        assert feats_normalized.shape[-1] == 8, f"Expected 8D features, got {feats_normalized.shape[-1]}D"
        assert target_vel.shape == (T, N, 3), f"Target velocity shape mismatch: {target_vel.shape}"
        assert target_stress.shape == (T, N, 1), f"Target stress shape mismatch: {target_stress.shape}"
        
        return {
            'coors': coors_normalized,      # (T, N, 3) - NORMALIZED coordinates
            'feats': feats_normalized,      # (T, N, 8) - NORMALIZED features [pos_norm(3), act_norm(3), node_type(2)]
            'adj_mat': adj_mat,             # (N, N) - adjacency matrix
            'edge_index': edge_index,       # (E, 2) - edge connectivity
            'world_pos': world_pos,         # (T, N, 3) - ORIGINAL positions (for denormalization)
            'target_vel': target_vel,       # (T, N, 3) - ORIGINAL velocity (will be normalized in train_epoch)
            'target_stress': target_stress  # (T, N, 1) - ORIGINAL stress (will be normalized in train_epoch)
        }


def compute_target_velocity(world_pos):
    """
    Compute target velocity from positions.
    For timestep t: vel[t] = pos[t] - pos[t-1]
    """
    T, N, _ = world_pos.shape
    vel = torch.zeros_like(world_pos)
    
    if T > 1:
        vel[1:] = world_pos[1:] - world_pos[:-1]
    
    return vel


def compute_target_stress(stress_array):
    """
    Convert stress array to tensor.
    """
    return torch.tensor(stress_array, dtype=torch.float32)


def custom_collate_fn(batch):
    """
    Custom collate function for variable-sized graphs.
    
    Implements graph batching by concatenating all graphs into one large graph
    with a block-diagonal adjacency matrix. This allows batching graphs with
    different numbers of nodes.
    
    Strategy:
    - If batch_size=1, return the single item directly
    - If batch_size > 1, always use graph batching (works for both same-size and different-size graphs):
      * Concatenate all node features, coordinates, targets along the node dimension
      * Create block-diagonal adjacency matrix (ensures nodes from different graphs don't interact)
      * Create batch_index to track which nodes belong to which graph
      * Process as single batch item (B=1) with total_N nodes
    """
    # CRITICAL: This function MUST be called by DataLoader
    # If you see errors about tensor size mismatches, it means default_collate was called instead
    # Validate batch format - ensure all items are dictionaries
    if not isinstance(batch, list):
        raise TypeError(f"Expected batch to be a list, got {type(batch)}")
    
    if len(batch) == 0:
        raise ValueError("Batch is empty")
    
    # Check that all items are dictionaries
    # If any item is not a dict, it means default_collate was called first (shouldn't happen)
    for i, item in enumerate(batch):
        if not isinstance(item, dict):
            raise TypeError(
                f"Batch item {i} is not a dictionary, got {type(item)}. "
                f"This suggests default_collate was called before custom_collate_fn. "
                f"Item: {item if not isinstance(item, torch.Tensor) else f'Tensor of shape {item.shape}'}"
            )
    
    # If batch_size=1, just return the single item
    if len(batch) == 1:
        return batch[0]
    
    # For batch_size > 1, always use graph batching
    # This is more robust and handles both same-size and different-size graphs
    # Graph batching concatenates all graphs into one large graph with block-diagonal adjacency
    # Standard approach: concatenate all graphs and process as single batch item (B=1)
    # Get dimensions
    num_graphs = len(batch)
    
    # Validate that all items have required keys
    required_keys = ['coors', 'feats', 'target_vel', 'target_stress', 'world_pos', 'adj_mat']
    for i, item in enumerate(batch):
        for key in required_keys:
            if key not in item:
                raise KeyError(f"Batch item {i} missing required key '{key}'. Available keys: {list(item.keys())}")
    
    T = batch[0]['coors'].shape[0]  # All should have same T
    node_counts = [item['coors'].shape[1] for item in batch]  # Different N per graph
    total_nodes = sum(node_counts)
    
    # Verify all have same T
    assert all(item['coors'].shape[0] == T for item in batch), \
        f"All trajectories must have same number of timesteps T, got {[item['coors'].shape[0] for item in batch]}"
    
    # Initialize concatenated tensors (B=1 for concatenated graph)
    device = batch[0]['coors'].device
    dtype = batch[0]['coors'].dtype
    
    coors_batched = torch.zeros(1, T, total_nodes, 3, dtype=dtype, device=device)
    feats_batched = torch.zeros(1, T, total_nodes, batch[0]['feats'].shape[2], dtype=dtype, device=device)
    target_vel_batched = torch.zeros(1, T, total_nodes, 3, dtype=dtype, device=device)
    target_stress_batched = torch.zeros(1, T, total_nodes, 1, dtype=dtype, device=device)
    world_pos_batched = torch.zeros(1, T, total_nodes, 3, dtype=dtype, device=device)
    
    # Create block-diagonal adjacency matrix (single batch item)
    adj_mat_batched = torch.zeros(1, total_nodes, total_nodes, dtype=torch.bool, device=device)
    
    # Create batch index: which graph does each node belong to?
    batch_index = torch.zeros(total_nodes, dtype=torch.long, device=device)
    
    # Concatenate each graph along the node dimension
    node_offset = 0
    for graph_idx, item in enumerate(batch):
        n_nodes = node_counts[graph_idx]
        
        # Handle input format: item might be (T, N, ...) or already have batch dimension
        coors_item = item['coors']
        feats_item = item['feats']
        target_vel_item = item['target_vel']
        target_stress_item = item['target_stress']
        world_pos_item = item['world_pos']
        adj_mat_item = item['adj_mat']
        
        # Remove batch dimension if present (should be single trajectory)
        if len(coors_item.shape) == 4:  # (1, T, N, 3)
            coors_item = coors_item[0]  # (T, N, 3)
            feats_item = feats_item[0]
            target_vel_item = target_vel_item[0]
            target_stress_item = target_stress_item[0]
            world_pos_item = world_pos_item[0]
        
        # Copy data into concatenated tensor (all graphs in single batch item)
        coors_batched[0, :, node_offset:node_offset+n_nodes, :] = coors_item
        feats_batched[0, :, node_offset:node_offset+n_nodes, :] = feats_item
        target_vel_batched[0, :, node_offset:node_offset+n_nodes, :] = target_vel_item
        target_stress_batched[0, :, node_offset:node_offset+n_nodes, :] = target_stress_item
        world_pos_batched[0, :, node_offset:node_offset+n_nodes, :] = world_pos_item
        
        # Add adjacency matrix as block (handle both 2D and 3D adj_mat)
        if len(adj_mat_item.shape) == 2:
            # (N, N) - direct assignment
            adj_mat_batched[0, node_offset:node_offset+n_nodes, node_offset:node_offset+n_nodes] = adj_mat_item
        elif len(adj_mat_item.shape) == 3:
            # (1, N, N) or (B_item, N, N) - take first
            adj_mat_batched[0, node_offset:node_offset+n_nodes, node_offset:node_offset+n_nodes] = adj_mat_item[0]
        else:
            raise ValueError(f"Unexpected adj_mat shape: {adj_mat_item.shape}")
        
        # Set batch index
        batch_index[node_offset:node_offset+n_nodes] = graph_idx
        
        node_offset += n_nodes
    
    return {
        'coors': coors_batched,  # (1, T, total_N, 3)
        'feats': feats_batched,  # (1, T, total_N, feat_dim)
        'adj_mat': adj_mat_batched,  # (1, total_N, total_N) - block diagonal
        'target_vel': target_vel_batched,  # (1, T, total_N, 3)
        'target_stress': target_stress_batched,  # (1, T, total_N, 1)
        'world_pos': world_pos_batched,  # (1, T, total_N, 3)
        'batch_index': batch_index,  # (total_N,) - which graph each node belongs to
        'node_counts': node_counts,  # List of node counts per graph (for loss computation)
    }


def save_predictions(model, dataloader, device, save_dir, epoch, dataset=None, num_trajectories=None, max_timesteps=None, use_autoregressive=True):
    """
    Save predictions and ground truth values for a subset of trajectories.
    
    CRITICAL: Model outputs are in NORMALIZED space. This function DENORMALIZES them
    before saving. Targets are in ORIGINAL scale and are saved as-is.
    
    Args:
        model: Trained model (outputs normalized predictions)
        dataloader: DataLoader with normalized inputs
        device: Device to run inference on
        save_dir: Directory to save predictions
        epoch: Current epoch number
        dataset: Dataset object (needed for normalization constants)
        num_trajectories: Number of trajectories to save (None = all)
        max_timesteps: Maximum number of timesteps to predict (None = all timesteps)
        use_autoregressive: If True, use autoregressive prediction (no teacher forcing).
                          If False, use teacher forcing (ground truth positions).
    """
    model.eval()
    
    if dataset is None:
        raise ValueError("dataset must be provided to save_predictions for denormalization")
    
    # Get normalization constants and move to device
    pos_mean = dataset.pos_mean.to(device)  # (3,)
    pos_std = dataset.pos_std.to(device)    # (3,)
    vel_std = torch.tensor(dataset.vel_std, device=device, dtype=torch.float32)  # scalar
    stress_mean = torch.tensor(dataset.stress_mean, device=device, dtype=torch.float32)  # scalar
    stress_std = torch.tensor(dataset.stress_std, device=device, dtype=torch.float32)  # scalar
    
    all_pred_vel = []
    all_pred_stress = []
    all_pred_pos = []
    all_true_vel = []
    all_true_stress = []
    all_true_pos = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Saving predictions (epoch {epoch+1})")
        for traj_idx, batch in enumerate(pbar):
            if num_trajectories is not None and traj_idx >= num_trajectories:
                break
                
            coors_seq = batch['coors'].to(device)  # (T, N, 3) - NORMALIZED
            feats_seq = batch['feats'].to(device)  # (T, N, 8) - NORMALIZED
            adj_mat = batch['adj_mat'].to(device)
            target_vel = batch['target_vel'].to(device)  # (T, N, 3) - ORIGINAL
            target_stress = batch['target_stress'].to(device)  # (T, N, 1) - ORIGINAL
            world_pos = batch.get('world_pos')  # (T, N, 3) - ORIGINAL
            if world_pos is None:
                raise KeyError(
                    "Batch is missing 'world_pos'. Ensure the dataset returns world positions."
                )
            world_pos = world_pos.to(device)
            
            # Handle batch dimension
            if len(coors_seq.shape) == 3:
                coors_seq = coors_seq.unsqueeze(0)
                feats_seq = feats_seq.unsqueeze(0)
                adj_mat = adj_mat.unsqueeze(0)
                target_vel = target_vel.unsqueeze(0)
                target_stress = target_stress.unsqueeze(0)
                world_pos = world_pos.unsqueeze(0)
            
            B, T, N, _ = coors_seq.shape
            
            # Determine end timestep: use max_timesteps if provided
            end_t = T if max_timesteps is None else min(T, max_timesteps)
            
            traj_pred_vel = []
            traj_pred_stress = []
            traj_pred_pos = []
            traj_true_vel = []
            traj_true_stress = []
            traj_true_pos = []
            
            # Initialize autoregressive state: start from t=0 ground truth
            # For autoregressive prediction, we'll update positions and features at each step
            if use_autoregressive:
                # Start with t=0 ground truth (normalized)
                current_coors_norm = coors_seq[:, 0].clone()  # (B, N, 3) - NORMALIZED
                current_feats_norm = feats_seq[:, 0].clone()  # (B, N, 8) - NORMALIZED
                # Features contain: [pos_norm(3), act_norm(3), node_type_one_hot(2)]
                # We'll update the position part (first 3 dims) at each step
                print(f"  Using AUTOREGRESSIVE prediction (no teacher forcing) for timesteps 1 to {end_t-1}")
            else:
                print(f"  Using TEACHER FORCING (ground truth positions) for timesteps 1 to {end_t-1}")
            
            # Process each timestep (starting from t=1)
            # Note: In autoregressive mode, we predict all timesteps regardless of stress threshold
            # This allows us to see predictions even on early timesteps with low stress
            for t in range(1, end_t):
                adj_mat_t = adj_mat
                target_vel_t = target_vel[:, t]  # (B, N, 3) - ORIGINAL
                target_stress_t = target_stress[:, t]  # (B, N, 1) - ORIGINAL
                target_pos_t = world_pos[:, t]  # (B, N, 3) - ORIGINAL
                
                if use_autoregressive:
                    # AUTOREGRESSIVE: Use predicted positions from previous step
                    coors_prev_norm = current_coors_norm  # (B, N, 3) - predicted from previous step
                    feats_prev_norm = current_feats_norm  # (B, N, 8) - updated from previous step
                else:
                    # TEACHER FORCING: Use ground truth positions and features
                    coors_prev_norm = coors_seq[:, t-1]  # (B, N, 3) - NORMALIZED
                    feats_prev_norm = feats_seq[:, t-1]   # (B, N, 8) - NORMALIZED
                
                # Forward pass: model outputs NORMALIZED predictions (should be masked for non-NORMAL nodes)
                pred_vel_norm, pred_stress_norm = model(feats_prev_norm, coors_prev_norm, adj_mat_t, v_init=None)
                
                # Verify masking is working
                # Velocity: only NORMAL nodes (node_type == 0) should have non-zero velocity
                # Stress: NORMAL (node_type == 0) or HANDLE (node_type == 3) nodes can have non-zero stress
                # OBSTACLE nodes (node_type == 1) should have zero for both
                node_type_one_hot_check = feats_prev_norm[:, :, 6:8]  # (B, N, 2)
                tolerance = 1e-6
                normal_mask_check = ((node_type_one_hot_check[:, :, 0].abs() < tolerance) & 
                                    (node_type_one_hot_check[:, :, 1].abs() < tolerance)).float()  # (B, N) - node_type == 0
                obstacle_mask_check = (node_type_one_hot_check[:, :, 0].abs() > 0.5).float()  # (B, N) - node_type == 1 (OBSTACLE)
                non_normal_vel_mask = 1.0 - normal_mask_check  # (B, N) - non-NORMAL nodes for velocity
                
                # Check velocity: non-NORMAL nodes should have zero velocity
                if non_normal_vel_mask.sum().item() > 0:
                    pred_vel_non_normal = pred_vel_norm[non_normal_vel_mask > 0.5]  # (num_non_normal, 3)
                    if pred_vel_non_normal.numel() > 0:
                        max_non_normal_vel = pred_vel_non_normal.abs().max().item()
                        if max_non_normal_vel > 1e-5:  # Should be essentially zero
                            # Force to zero as a safeguard
                            pred_vel_norm = pred_vel_norm * normal_mask_check.unsqueeze(-1).unsqueeze(-1)
                
                # Check stress: OBSTACLE nodes (node_type == 1) should have zero stress
                # HANDLE nodes (node_type == 3) can have non-zero stress, so we only check OBSTACLE
                if obstacle_mask_check.sum().item() > 0:
                    pred_stress_obstacle = pred_stress_norm.squeeze(-1)[obstacle_mask_check > 0.5]  # (num_obstacle,)
                    if pred_stress_obstacle.numel() > 0:
                        max_obstacle_stress = pred_stress_obstacle.abs().max().item()
                        if max_obstacle_stress > 1e-5:  # Should be essentially zero
                            # Force OBSTACLE nodes to zero as a safeguard
                            stress_mask_for_obstacle = 1.0 - obstacle_mask_check  # (B, N) - non-OBSTACLE nodes
                            pred_stress_norm = pred_stress_norm * stress_mask_for_obstacle.unsqueeze(-1)
                
                
                # DENORMALIZE predictions for saving
                # CRITICAL: Extract node_type to mask nodes after denormalization
                # The model should already mask predictions, but we double-check here for safety
                node_type_one_hot = feats_prev_norm[:, :, 6:8]  # (B, N, 2)
                tolerance = 1e-6
                # Velocity mask: only node_type == 0 (NORMAL nodes)
                velocity_node_mask = ((node_type_one_hot[:, :, 0].abs() < tolerance) & 
                                     (node_type_one_hot[:, :, 1].abs() < tolerance)).float()  # (B, N)
                velocity_node_mask = velocity_node_mask.unsqueeze(-1)  # (B, N, 1) for broadcasting
                
                # Stress mask: node_type == 0 (NORMAL) OR node_type == 3 (HANDLE)
                stress_node_mask = (node_type_one_hot[:, :, 0].abs() < tolerance).float()  # (B, N)
                stress_node_mask = stress_node_mask.unsqueeze(-1)  # (B, N, 1) for broadcasting
                
                # Velocity: v = v_norm * vel_std
                pred_vel_denorm = pred_vel_norm * vel_std  # (B, N, 3)
                # Ensure non-NORMAL nodes have zero velocity
                pred_vel_denorm = pred_vel_denorm * velocity_node_mask  # (B, N, 3)
                
                # Stress: s = s_norm * stress_std + stress_mean
                # Denormalize for NORMAL or HANDLE nodes (node_type == 0 or 3), set to 0 for others
                pred_stress_denorm = torch.where(
                    stress_node_mask > 0.5,  # If NORMAL or HANDLE node
                    pred_stress_norm * stress_std + stress_mean,  # Denormalize
                    torch.zeros_like(pred_stress_norm)  # Set to 0 for other nodes (OBSTACLE)
                )  # (B, N, 1)
                pred_stress_denorm = torch.clamp(pred_stress_denorm, min=0.0)  # Ensure stress >= 0 (von Mises stress)
                pred_stress_denorm = pred_stress_denorm.unsqueeze(-1)  # (B, N, 1)
                
                
                # Denormalize positions for computing predicted coordinates
                # coors_prev_norm is normalized, so: pos_prev = coors_prev_norm * pos_std + pos_mean
                pos_prev_denorm = coors_prev_norm * pos_std.unsqueeze(0).unsqueeze(0) + pos_mean.unsqueeze(0).unsqueeze(0)  # (B, N, 3)
                
                # Compute predicted coordinates: pos_pred = pos_prev + vel_pred
                pred_pos_denorm = pos_prev_denorm + pred_vel_denorm  # (B, N, 3)
                
                # AUTOREGRESSIVE UPDATE: Update state for next timestep
                if use_autoregressive and t < end_t - 1:  # Don't update on last timestep
                    # Update normalized coordinates for next step
                    current_coors_norm = (pred_pos_denorm - pos_mean.unsqueeze(0).unsqueeze(0)) / pos_std.unsqueeze(0).unsqueeze(0)  # (B, N, 3)
                    
                    # Update features: [pos_norm(3), act_norm(3), node_type_one_hot(2)]
                    # Update position part (first 3 dims) with new predicted positions
                    # Keep actuation (dims 3:6) and node_type (dims 6:8) unchanged
                    current_feats_norm = current_feats_norm.clone()  # (B, N, 8)
                    current_feats_norm[:, :, 0:3] = current_coors_norm  # Update position part
                    # Actuation and node_type remain the same (they're static or from t=0)
                
                # Convert to numpy and store (all in ORIGINAL scale)
                traj_pred_vel.append(pred_vel_denorm.cpu().numpy())
                traj_pred_stress.append(pred_stress_denorm.cpu().numpy())
                traj_pred_pos.append(pred_pos_denorm.cpu().numpy())
                traj_true_vel.append(target_vel_t.cpu().numpy())
                traj_true_stress.append(target_stress_t.cpu().numpy())
                traj_true_pos.append(target_pos_t.cpu().numpy())
            
            # Stack timesteps
            traj_pred_vel = np.stack(traj_pred_vel, axis=0)
            traj_pred_stress = np.stack(traj_pred_stress, axis=0)
            traj_pred_pos = np.stack(traj_pred_pos, axis=0)
            traj_true_vel = np.stack(traj_true_vel, axis=0)
            traj_true_stress = np.stack(traj_true_stress, axis=0)
            traj_true_pos = np.stack(traj_true_pos, axis=0)
            
            # Remove batch dimension if B=1
            if B == 1:
                traj_pred_vel = traj_pred_vel[:, 0]
                traj_pred_stress = traj_pred_stress[:, 0]
                traj_pred_pos = traj_pred_pos[:, 0]
                traj_true_vel = traj_true_vel[:, 0]
                traj_true_stress = traj_true_stress[:, 0]
                traj_true_pos = traj_true_pos[:, 0]
            
            all_pred_vel.append(traj_pred_vel)
            all_pred_stress.append(traj_pred_stress)
            all_pred_pos.append(traj_pred_pos)
            all_true_vel.append(traj_true_vel)
            all_true_stress.append(traj_true_stress)
            all_true_pos.append(traj_true_pos)
    
    # Save as numpy arrays
    # Note: Different trajectories may have different numbers of nodes (N),
    # so we save as a list of arrays rather than trying to stack into a single array
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save as lists of arrays using pickle (handles variable shapes across trajectories)
    # Use pickle directly to avoid numpy's array conversion issues
    with open(os.path.join(epoch_dir, 'predictions_velocity.npy'), 'wb') as f:
        pickle.dump(all_pred_vel, f)
    with open(os.path.join(epoch_dir, 'predictions_stress.npy'), 'wb') as f:
        pickle.dump(all_pred_stress, f)
    with open(os.path.join(epoch_dir, 'predictions_positions.npy'), 'wb') as f:
        pickle.dump(all_pred_pos, f)
    with open(os.path.join(epoch_dir, 'true_velocity.npy'), 'wb') as f:
        pickle.dump(all_true_vel, f)
    with open(os.path.join(epoch_dir, 'true_stress.npy'), 'wb') as f:
        pickle.dump(all_true_stress, f)
    with open(os.path.join(epoch_dir, 'true_positions.npy'), 'wb') as f:
        pickle.dump(all_true_pos, f)
    
    print(f"  -> Saved predictions to {epoch_dir}")
    
    model.train()  # Set back to training mode


def train_epoch(model, dataloader, optimizer, device, epoch, velocity_loss_weight=1.0, dataset=None, use_adaptive_scaling=False, max_timesteps=None, min_stress_threshold=5000.0, warmup_fraction=0.025):
    """
    Train for one epoch.
    
    Args:
        velocity_loss_weight: Weight for velocity loss to balance with stress loss.
                             Since velocity values are ~0.0002 and stress ~18000,
                             we need a large weight to balance the losses.
                             Only used if use_adaptive_scaling=False.
        use_adaptive_scaling: If True, use adaptive loss scaling with EMA instead of fixed weight.
    """
    model.train()
    total_loss = 0.0
    total_loss_vel = 0.0
    total_loss_stress = 0.0
    num_samples = 0
    
    # Initialize adaptive scaling parameters (using EMA with momentum=0.99)
    if use_adaptive_scaling:
        if not hasattr(model, 'vel_loss_scale'):
            model.vel_loss_scale = 1.0
        if not hasattr(model, 'stress_loss_scale'):
            model.stress_loss_scale = 1.0
        ema_momentum = 0.99  # Smooth updates
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        # Custom collate function always returns a dict (single item or batched)
        coors_seq = batch['coors'].to(device)  # (B, T, N, 3) or (T, N, 3) or (B, T, total_N, 3) for graph batching
        feats_seq = batch['feats'].to(device)  # (B, T, N, feat_dim) or (T, N, feat_dim) or (B, T, total_N, feat_dim)
        adj_mat = batch['adj_mat'].to(device)   # (B, N, N) or (N, N) or (B, total_N, total_N) for graph batching
        target_vel = batch['target_vel'].to(device)  # (B, T, N, 3) or (T, N, 3) or (B, T, total_N, 3)
        target_stress = batch['target_stress'].to(device)  # (B, T, N, 1) or (T, N, 1) or (B, T, total_N, 1)
        world_pos = batch['world_pos'].to(device)  # (B, T, N, 3) or (T, N, 3) or (B, T, total_N, 3)
        
        # Get batch_index and node_counts if available (from graph batching)
        batch_index = batch.get('batch_index')  # (total_N,) - which graph each node belongs to
        node_counts = batch.get('node_counts')  # List of node counts per graph
        if batch_index is not None:
            batch_index = batch_index.to(device)
        
        # Handle batch dimension
        if len(coors_seq.shape) == 3:  # (T, N, 3) - single trajectory
            coors_seq = coors_seq.unsqueeze(0)  # (1, T, N, 3)
            feats_seq = feats_seq.unsqueeze(0)  # (1, T, N, feat_dim)
            adj_mat = adj_mat.unsqueeze(0)      # (1, N, N)
            target_vel = target_vel.unsqueeze(0)  # (1, T, N, 3)
            target_stress = target_stress.unsqueeze(0)  # (1, T, N, 1)
            world_pos = world_pos.unsqueeze(0)  # (1, T, N, 3)
        
        B, T, N, _ = coors_seq.shape
        
        epoch_loss_batch = 0.0
        epoch_loss_vel_batch = 0.0
        epoch_loss_stress_batch = 0.0
        
        # Storage for rollout error computation (denormalized values)
        # Store predictions and targets for each valid timestep
        rollout_pred_vel = []  # List of (B, N, 3) tensors in denormalized space
        rollout_pred_stress = []  # List of (B, N, 1) tensors in denormalized space
        rollout_target_vel = []  # List of (B, N, 3) tensors in denormalized space
        rollout_target_stress = []  # List of (B, N, 1) tensors in denormalized space
        rollout_valid_timestep_indices = []  # Track which timesteps were valid (for indexing)
        rollout_velocity_mask_list = []  # Store velocity masks for each timestep
        rollout_stress_mask_list = []  # Store stress masks for each timestep
        
        # Process each timestep (starting from t=1)
        # TEACHER FORCING: Use ground truth positions at each timestep (not autoregressive)
        # This allows the model to learn single-step predictions correctly
        
        # CRITICAL FIX: Skip early timesteps and only train on high-stress timesteps
        # This prevents the model from learning to predict zero stress
        # These are now configurable via command-line arguments
        WARMUP_FRACTION = warmup_fraction  # Skip first fraction of trajectory (default: 2.5%)
        MIN_STRESS_THRESHOLD = min_stress_threshold  # Only train on timesteps with mean stress > threshold
        
        optimizer.zero_grad()
        
        valid_timesteps = 0  # Count timesteps with meaningful targets
        skipped_low_stress = 0  # Count timesteps skipped due to low stress
        
        # Determine end timestep: use max_timesteps if provided, otherwise use all timesteps
        end_t = T if max_timesteps is None else min(T, max_timesteps)
        
        # Start from warmup fraction to skip early timesteps where stress is near-zero
        # But adjust warmup if max_timesteps is too small to ensure we have valid timesteps
        if max_timesteps is not None and max_timesteps < T:
            # If max_timesteps is small, reduce warmup proportionally
            # Calculate warmup based on max_timesteps, not full trajectory length
            warmup_timesteps = max(0, int(max_timesteps * warmup_fraction))
            start_t = max(1, warmup_timesteps)
            # Ensure we have at least 1 timestep to train on
            if start_t >= end_t:
                # If warmup would eliminate all timesteps, start from t=1
                start_t = 1
        else:
            # Normal case: use full trajectory warmup
            start_t = max(1, int(T * warmup_fraction))
        
        # Final check: ensure we have valid timesteps
        if end_t <= start_t:
            if batch_idx == 0:
                print(f"\n⚠ ERROR: Invalid timestep range!")
                print(f"  start_t={start_t}, end_t={end_t}")
                print(f"  max_timesteps={max_timesteps} is too small!")
                print(f"  Need at least {start_t + 1} timesteps. Increase --max_timesteps to at least {start_t + 1}")
            # Set end_t to start_t + 1 as a fallback, but this will likely skip all timesteps due to stress threshold
            end_t = start_t + 1
        
        # Detect graph batching mode (when batch_index is available)
        is_graph_batching = (batch_index is not None and node_counts is not None)
        num_graphs_in_batch = len(node_counts) if is_graph_batching else B
        
        if batch_idx == 0 and epoch == 0:  # Only print once at start
            print(f"Training on timesteps: {start_t} to {end_t-1} (min stress threshold: {MIN_STRESS_THRESHOLD:.1f})")
            if is_graph_batching:
                print(f"  Graph batching mode: {num_graphs_in_batch} graphs in batch (total {N} nodes)")
        
        for t in range(start_t, end_t):
            # TEACHER FORCING: Use ground truth positions and features at time t-1
            # This is the correct approach for learning single-step predictions
            coors_t_minus_1 = coors_seq[:, t-1]  # Ground truth positions at t-1
            feats_t_minus_1 = feats_seq[:, t-1]   # Ground truth features at t-1
            adj_mat_t = adj_mat
            
            # Extract node type for masking (indices 6:8 are node_type_one_hot)
            # According to meshgraphnets/common.py: NORMAL=0, OBSTACLE=1, HANDLE=3
            # Encoding: 0 (NORMAL) -> [0, 0], 1 (OBSTACLE) -> [1, 0], 3 (HANDLE) -> [0, 1]
            node_type_one_hot = feats_t_minus_1[:, :, 6:8]  # (B, N, 2)
            
            # Targets at time t
            target_vel_t = target_vel[:, t]      # Velocity at time t
            target_stress_t = target_stress[:, t]  # Stress at time t
            
            # Create masks for velocity and stress losses
            # Velocity mask: only node_type == 0 (NORMAL nodes)
            # Encoding: 0 (NORMAL) -> [0, 0], 1 (OBSTACLE) -> [1, 0], 3 (HANDLE) -> [0, 1]
            velocity_mask = ((node_type_one_hot[:, :, 0].abs() < 1e-6) & (node_type_one_hot[:, :, 1].abs() < 1e-6)).float()  # (B, N) - node_type == 0 only
            
            # Stress mask: node_type == 0 (NORMAL) OR node_type == 3 (HANDLE)
            # node_type == 0: [0, 0], node_type == 3: [0, 1] -> both have one_hot[0] ≈ 0
            stress_mask = (node_type_one_hot[:, :, 0].abs() < 1e-6).float()  # (B, N) - node_type == 0 OR 3
            
            # PER-TRAJECTORY STRESS THRESHOLDING
            if is_graph_batching:
                # Graph batching mode: compute mean_stress per graph and filter
                # batch_index: (total_N,) maps each node to graph id [0..num_graphs-1]
                # For batch item 0 (since graph batching creates B=1), extract node-level data
                target_stress_t_flat = target_stress_t[0].squeeze(-1)  # (total_N,) - remove batch and feature dims
                stress_mask_flat = stress_mask[0]  # (total_N,)
                
                # Compute mean_stress per graph using vectorized operations
                # Group nodes by graph_id using batch_index
                mean_stress_per_graph = []
                node_offset = 0
                for g in range(num_graphs_in_batch):
                    n_nodes_g = node_counts[g]
                    # Get nodes belonging to graph g
                    node_indices_g = torch.arange(node_offset, node_offset + n_nodes_g, device=device)
                    # Extract stress and mask for this graph
                    stress_g = target_stress_t_flat[node_indices_g]  # (n_nodes_g,)
                    mask_g = stress_mask_flat[node_indices_g]  # (n_nodes_g,)
                    # Compute mean stress for this graph (only over masked nodes)
                    if mask_g.sum() > 0:
                        mean_stress_g = (stress_g.abs() * mask_g).sum().item() / (mask_g.sum().item() + 1e-8)
                    else:
                        mean_stress_g = 0.0
                    mean_stress_per_graph.append(mean_stress_g)
                    node_offset += n_nodes_g
                
                # Filter graphs that pass threshold
                valid_graphs = [g for g in range(num_graphs_in_batch) if mean_stress_per_graph[g] >= MIN_STRESS_THRESHOLD]
                
                # Skip timestep if no graphs are valid
                if len(valid_graphs) == 0:
                    skipped_low_stress += 1
                    if batch_idx == 0 and valid_timesteps == 0 and t == end_t - 1:
                        print(f"  ⚠ All timesteps have no graphs with stress >= {MIN_STRESS_THRESHOLD} - no training data")
                    continue
                
                valid_timesteps += 1
                
                # Debug print for first valid timestep
                if batch_idx == 0 and epoch == 0 and valid_timesteps == 1:
                    mean_stress_array = np.array(mean_stress_per_graph)
                    print(f"  Per-trajectory stress thresholding (t={t}):")
                    print(f"    Graphs in batch: {num_graphs_in_batch}")
                    print(f"    Graphs passing threshold: {len(valid_graphs)}/{num_graphs_in_batch}")
                    if len(mean_stress_per_graph) > 0:
                        print(f"    Mean stress per graph: min={mean_stress_array.min():.1f}, "
                              f"median={np.median(mean_stress_array):.1f}, max={mean_stress_array.max():.1f}")
            else:
                # Standard batching mode (batch_size=1 or same-size graphs): use existing logic
                # Check if this timestep has significant stress
                if stress_mask.sum() > 0:
                    mean_stress = (target_stress_t.squeeze(-1).abs() * stress_mask).sum().item() / (stress_mask.sum().item() + 1e-8)
                else:
                    mean_stress = 0.0
                
                if mean_stress < MIN_STRESS_THRESHOLD:
                    # Skip timesteps with low stress - avoid learning to predict zeros
                    skipped_low_stress += 1
                    # Warn if we're skipping all timesteps due to stress threshold
                    if batch_idx == 0 and valid_timesteps == 0 and t == end_t - 1:
                        print(f"  ⚠ All timesteps have stress < {MIN_STRESS_THRESHOLD} - no training data")
                    continue
                
                valid_timesteps += 1
                valid_graphs = list(range(B))  # All batch items are valid in standard mode
            
            # Forward pass: predict velocity and stress at time t given state at t-1
            # CRITICAL: Model receives NORMALIZED inputs (feats, coors) and outputs NORMALIZED predictions
            # Inputs are already normalized in __getitem__, so model operates in normalized space
            # Model outputs are in normalized space: pred_vel_norm, pred_stress_norm
            pred_vel_norm, pred_stress_norm = model(feats_t_minus_1, coors_t_minus_1, adj_mat_t, v_init=None)
            
            # Check for NaN in predictions
            if torch.isnan(pred_vel_norm).any() or torch.isnan(pred_stress_norm).any():
                continue
            
            # Validate masks
            velocity_mask_sum = velocity_mask.sum().item()
            stress_mask_sum = stress_mask.sum().item()
            if velocity_mask_sum < 1.0:
                # Velocity mask is invalid (0 nodes), this indicates node_type encoding issue
                print(f"  ⚠ WARNING: Invalid velocity mask (sum={velocity_mask_sum:.6f}), expected node_type==0 for velocity loss")
                # Fallback: use all nodes if mask is completely wrong
                velocity_mask = torch.ones_like(velocity_mask)
            if stress_mask_sum < 1.0:
                # Stress mask is invalid (0 nodes), this indicates node_type encoding issue
                print(f"  ⚠ WARNING: Invalid stress mask (sum={stress_mask_sum:.6f}), expected node_type==0 or 3 for stress loss")
                # Fallback: use all nodes if mask is completely wrong
                stress_mask = torch.ones_like(stress_mask)
            
            # NORMALIZE TARGETS: Targets come in original scale, normalize them for loss computation
            # This ensures loss is computed in normalized space (O(1) magnitudes)
            if dataset is not None:
                # Move normalization constants to device
                vel_std = torch.tensor(dataset.vel_std, device=device, dtype=torch.float32)
                stress_mean = torch.tensor(dataset.stress_mean, device=device, dtype=torch.float32)
                stress_std = torch.tensor(dataset.stress_std, device=device, dtype=torch.float32)
                
                # Normalize targets: v_norm = v / vel_std, s_norm = (s - stress_mean) / stress_std
                target_vel_t_norm = target_vel_t / vel_std  # (B, N, 3)
                target_stress_t_norm = (target_stress_t.squeeze(-1) - stress_mean) / stress_std  # (B, N)
                
                # Denormalize predictions for rollout error computation
                # pred_vel_norm: (B, N, 3) in normalized space
                # pred_stress_norm: (B, N, 1) in normalized space
                pred_vel_denorm = pred_vel_norm * vel_std  # (B, N, 3) - denormalized
                pred_stress_denorm = pred_stress_norm * stress_std + stress_mean  # (B, N, 1) - denormalized
            else:
                # Fallback: no normalization (shouldn't happen)
                target_vel_t_norm = target_vel_t
                target_stress_t_norm = target_stress_t.squeeze(-1)
                pred_vel_denorm = pred_vel_norm
                pred_stress_denorm = pred_stress_norm
            
            # Store denormalized predictions and targets for rollout error computation
            # Store as detached tensors to avoid memory issues
            rollout_pred_vel.append(pred_vel_denorm.detach().cpu())
            rollout_pred_stress.append(pred_stress_denorm.detach().cpu())
            rollout_target_vel.append(target_vel_t.detach().cpu())  # Already in original scale
            rollout_target_stress.append(target_stress_t.detach().cpu())  # Already in original scale
            rollout_valid_timestep_indices.append(t)
            rollout_velocity_mask_list.append(velocity_mask.detach().cpu())
            rollout_stress_mask_list.append(stress_mask.detach().cpu())
            
            # PER-TRAJECTORY LOSS COMPUTATION
            if is_graph_batching:
                # Graph batching mode: compute loss per graph, then average across valid graphs
                # Extract node-level data (batch item 0 since graph batching creates B=1)
                pred_vel_norm_flat = pred_vel_norm[0]  # (total_N, 3)
                pred_stress_norm_flat = pred_stress_norm[0].squeeze(-1)  # (total_N,)
                target_vel_t_norm_flat = target_vel_t_norm[0]  # (total_N, 3)
                target_stress_t_norm_flat = target_stress_t_norm[0]  # (total_N,)
                velocity_mask_flat = velocity_mask[0]  # (total_N,)
                stress_mask_flat = stress_mask[0]  # (total_N,)
                
                # Compute losses per graph
                loss_vel_per_graph = []
                loss_stress_per_graph = []
                
                node_offset = 0
                for g in range(num_graphs_in_batch):
                    n_nodes_g = node_counts[g]
                    node_indices_g = torch.arange(node_offset, node_offset + n_nodes_g, device=device)
                    
                    # Extract data for this graph
                    pred_vel_g = pred_vel_norm_flat[node_indices_g]  # (n_nodes_g, 3)
                    pred_stress_g = pred_stress_norm_flat[node_indices_g]  # (n_nodes_g,)
                    target_vel_g = target_vel_t_norm_flat[node_indices_g]  # (n_nodes_g, 3)
                    target_stress_g = target_stress_t_norm_flat[node_indices_g]  # (n_nodes_g,)
                    velocity_mask_g = velocity_mask_flat[node_indices_g]  # (n_nodes_g,)
                    stress_mask_g = stress_mask_flat[node_indices_g]  # (n_nodes_g,)
                    
                    # Velocity loss for this graph
                    vel_error_g = (pred_vel_g - target_vel_g) ** 2  # (n_nodes_g, 3)
                    vel_error_per_node_g = torch.sum(vel_error_g, dim=-1)  # (n_nodes_g,)
                    vel_error_masked_g = vel_error_per_node_g * velocity_mask_g  # (n_nodes_g,)
                    loss_vel_g = vel_error_masked_g.sum() / (velocity_mask_g.sum() + 1e-8)  # Scalar
                    
                    # Stress loss for this graph
                    stress_error_g = (pred_stress_g - target_stress_g) ** 2  # (n_nodes_g,)
                    stress_error_masked_g = stress_error_g * stress_mask_g  # (n_nodes_g,)
                    
                    # Penalties for this graph
                    target_stress_nonzero_g = (target_stress_g.abs() > 1e-6).float()  # (n_nodes_g,)
                    pred_stress_abs_g = pred_stress_g.abs()  # (n_nodes_g,)
                    pred_stress_near_zero_g = (pred_stress_abs_g < 0.01).float()  # (n_nodes_g,)
                    zero_prediction_penalty_g = (target_stress_nonzero_g * pred_stress_near_zero_g * stress_mask_g).sum() / (stress_mask_g.sum() + 1e-8)
                    
                    target_stress_abs_g = target_stress_g.abs()  # (n_nodes_g,)
                    severe_underestimate_g = (pred_stress_abs_g < 0.1 * target_stress_abs_g) & (target_stress_abs_g > 0.1)  # (n_nodes_g,)
                    underestimate_penalty_g = (severe_underestimate_g.float() * stress_mask_g).sum() / (stress_mask_g.sum() + 1e-8)
                    
                    penalty_weight_zero = 1.0
                    penalty_weight_under = 0.5
                    mean_target_sq_g = (target_stress_g ** 2 * stress_mask_g).sum() / (stress_mask_g.sum() + 1e-8)
                    zero_penalty_g = penalty_weight_zero * mean_target_sq_g * zero_prediction_penalty_g
                    under_penalty_g = penalty_weight_under * mean_target_sq_g * underestimate_penalty_g
                    
                    loss_stress_g = stress_error_masked_g.sum() / (stress_mask_g.sum() + 1e-8) + zero_penalty_g + under_penalty_g
                    
                    loss_vel_per_graph.append(loss_vel_g)
                    loss_stress_per_graph.append(loss_stress_g)
                    
                    node_offset += n_nodes_g
                
                # Average losses over valid graphs only (equal weight per trajectory)
                # Use torch.stack to maintain gradients properly
                if len(valid_graphs) > 0:
                    loss_vel_tensors = torch.stack([loss_vel_per_graph[g] for g in valid_graphs])
                    loss_stress_tensors = torch.stack([loss_stress_per_graph[g] for g in valid_graphs])
                    loss_vel = loss_vel_tensors.mean()
                    loss_stress = loss_stress_tensors.mean()
                else:
                    # Should not happen (we skip timestep if no valid graphs), but safety check
                    loss_vel = torch.tensor(0.0, device=device, requires_grad=True)
                    loss_stress = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # Standard batching mode: compute loss over all nodes (existing logic)
                # Velocity loss: compute per-node, then mask and average (on NORMALIZED values)
                # Both predictions and targets are in normalized space (O(1) magnitudes)
                # Use velocity_mask: only node_type == 0 (NORMAL nodes)
                vel_error = (pred_vel_norm - target_vel_t_norm) ** 2  # (B, N, 3)
                vel_error_per_node = torch.sum(vel_error, dim=-1)  # (B, N)
                vel_error_masked = vel_error_per_node * velocity_mask  # (B, N)
                loss_vel = vel_error_masked.sum() / (velocity_mask.sum() + 1e-8)  # Average over masked nodes
                
                # Stress loss: compute per-node, then mask and average (on NORMALIZED values)
                # Use stress_mask: node_type == 0 (NORMAL) OR node_type == 3 (HANDLE)
                # Velocity loss uses velocity_mask: only node_type == 0 (NORMAL nodes)
                # Boundary nodes (node_type == 1, OBSTACLE) are fixed and should not contribute to loss
                stress_error = (pred_stress_norm.squeeze(-1) - target_stress_t_norm) ** 2  # (B, N) - ensure shapes match
                stress_error_masked = stress_error * stress_mask  # (B, N) - use NORMAL or HANDLE nodes (node_type == 0 or 3)
                
                # CRITICAL FIX: Add penalty for predicting zero or very small stress when true stress is non-zero
                # This prevents the model from learning to predict zero for boundary NORMAL nodes
                # Only apply penalty to NORMAL nodes with non-zero true stress
                target_stress_nonzero = (target_stress_t_norm.abs() > 1e-6).float()  # (B, N) - nodes with non-zero true stress
                pred_stress_abs = pred_stress_norm.squeeze(-1).abs()  # (B, N) - absolute predicted stress
                
                # Penalty 1: Predictions that are exactly zero or very near zero (< 0.01 in normalized space)
                # This corresponds to ~0.01 * stress_std in original space, which is very small
                pred_stress_near_zero = (pred_stress_abs < 0.01).float()  # (B, N) - predictions near zero
                zero_prediction_penalty = (target_stress_nonzero * pred_stress_near_zero * stress_mask).sum() / (stress_mask.sum() + 1e-8)
                
                # Penalty 2: Predictions that are significantly smaller than target (underestimation)
                # If target is large but prediction is small, add penalty
                # Only penalize when prediction < 0.1 * target (severe underestimation)
                target_stress_abs = target_stress_t_norm.abs()  # (B, N)
                severe_underestimate = (pred_stress_abs < 0.1 * target_stress_abs) & (target_stress_abs > 0.1)  # (B, N)
                underestimate_penalty = (severe_underestimate.float() * stress_mask).sum() / (stress_mask.sum() + 1e-8)
                
                # Scale penalties to be comparable to MSE
                penalty_weight_zero = 1.0  # stronger penalty for zero predictions
                penalty_weight_under = 0.5  # Weight for underestimation penalty
                mean_target_sq = (target_stress_t_norm ** 2 * stress_mask).sum() / (stress_mask.sum() + 1e-8)
                zero_penalty = penalty_weight_zero * mean_target_sq * zero_prediction_penalty
                under_penalty = penalty_weight_under * mean_target_sq * underestimate_penalty
                
                # Normalize by weighted mask sum to account for boundary node weighting
                loss_stress = stress_error_masked.sum() / (stress_mask.sum() + 1e-8) + zero_penalty + under_penalty  # Average over plate nodes + penalties
            
            # CRITICAL DEBUG: Check if stress predictions are uniform BEFORE computing loss
            if batch_idx == 0 and valid_timesteps == 1:
                pred_stress_flat = pred_stress_norm.squeeze(-1)  # (B, N)
                pred_stress_std_all = pred_stress_flat.std().item()
                pred_stress_range_all = pred_stress_flat.max().item() - pred_stress_flat.min().item()
                
                if pred_stress_std_all < 0.01:
                    print(f"  ⚠ CRITICAL: Stress predictions are UNIFORM across ALL nodes!")
                    print(f"     std={pred_stress_std_all:.8f}, range={pred_stress_range_all:.8f}")
                    print(f"     mean={pred_stress_flat.mean().item():.6f}")
                    print(f"     This means the stress head is outputting the same value for all nodes!")
                    print(f"     Possible causes: vanishing gradients, dead ReLU, or initialization issue.")
            
            # CRITICAL DEBUG: Check stress predictions vs targets in normalized space
            # Also check boundary nodes specifically
            if batch_idx == 0 and valid_timesteps == 1:
                pred_stress_norm_masked = pred_stress_norm.squeeze(-1)[stress_mask > 0]  # (num_masked,)
                target_stress_norm_masked = target_stress_t_norm[stress_mask > 0]  # (num_masked,)
                
                # Identify boundary nodes by checking node positions (NORMAL nodes at edges)
                # This is approximate - we check if nodes are near min/max positions
                coors_t_minus_1_flat = coors_t_minus_1[0]  # (N, 3) - remove batch dim
                x_min, x_max = coors_t_minus_1_flat[:, 0].min(), coors_t_minus_1_flat[:, 0].max()
                y_min, y_max = coors_t_minus_1_flat[:, 1].min(), coors_t_minus_1_flat[:, 1].max()
                z_min, z_max = coors_t_minus_1_flat[:, 2].min(), coors_t_minus_1_flat[:, 2].max()
                x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
                tol = 0.02  # 2% tolerance for boundary detection
                
                boundary_x = ((coors_t_minus_1_flat[:, 0] - x_min) < tol * x_range) | ((x_max - coors_t_minus_1_flat[:, 0]) < tol * x_range)
                boundary_y = ((coors_t_minus_1_flat[:, 1] - y_min) < tol * y_range) | ((y_max - coors_t_minus_1_flat[:, 1]) < tol * y_range)
                boundary_z = ((coors_t_minus_1_flat[:, 2] - z_min) < tol * z_range) | ((z_max - coors_t_minus_1_flat[:, 2]) < tol * z_range)
                boundary_mask = (boundary_x | boundary_y | boundary_z).float()  # (N,)
                boundary_normal_mask = boundary_mask * stress_mask[0]  # (N,) - boundary NORMAL nodes
                
                if boundary_normal_mask.sum() > 0:
                    boundary_pred = pred_stress_norm.squeeze(-1)[0][boundary_normal_mask > 0.5]  # (num_boundary,)
                    boundary_target = target_stress_t_norm[0][boundary_normal_mask > 0.5]  # (num_boundary,)
                    boundary_zero_pred = (boundary_pred.abs() < 0.01).sum().item()
                    boundary_total = len(boundary_pred)
                    
                    print(f"  Boundary NORMAL nodes: {boundary_total}, zero predictions: {boundary_zero_pred} ({100*boundary_zero_pred/max(boundary_total,1):.1f}%)")
                    if boundary_total > 0:
                        print(f"    Boundary pred: min={boundary_pred.min().item():.4f}, max={boundary_pred.max().item():.4f}, mean={boundary_pred.mean().item():.4f}")
                        print(f"    Boundary target: min={boundary_target.min().item():.4f}, max={boundary_target.max().item():.4f}, mean={boundary_target.mean().item():.4f}")
                
                
                if pred_stress_norm_masked.numel() > 0:
                    print(f"\n  === STRESS DEBUG (t={t}, Normalized Space) ===")
                    print(f"  Normalization: mean={dataset.stress_mean:.2f}, std={dataset.stress_std:.2f}")
                    print(f"  Predicted stress (normalized, NORMAL nodes only):")
                    print(f"    min={pred_stress_norm_masked.min().item():.6f}, max={pred_stress_norm_masked.max().item():.6f}")
                    print(f"    mean={pred_stress_norm_masked.mean().item():.6f}, std={pred_stress_norm_masked.std().item():.6f}")
                    print(f"  Target stress (normalized, NORMAL nodes only):")
                    print(f"    min={target_stress_norm_masked.min().item():.6f}, max={target_stress_norm_masked.max().item():.6f}")
                    print(f"    mean={target_stress_norm_masked.mean().item():.6f}, std={target_stress_norm_masked.std().item():.6f}")
                    
                    # Check if predictions are uniform
                    pred_std = pred_stress_norm_masked.std().item()
                    target_std = target_stress_norm_masked.std().item()
                    pred_mean = pred_stress_norm_masked.mean().item()
                    target_mean = target_stress_norm_masked.mean().item()
                    
                    if pred_std < 0.01 * abs(pred_mean):
                        print(f"  ⚠ CRITICAL: Predictions are UNIFORM! std={pred_std:.8f}, mean={pred_mean:.6f}")
                        print(f"     Model is predicting the same value ({pred_mean:.6f}) for all nodes!")
                    
                    # Check prediction vs target statistics
                    pred_target_diff = (pred_stress_norm_masked - target_stress_norm_masked).abs()
                    mean_error = pred_target_diff.mean().item()
                    max_error = pred_target_diff.max().item()
                    print(f"  Prediction error (normalized):")
                    print(f"    mean_abs_error={mean_error:.6f}, max_abs_error={max_error:.6f}")
                    print(f"    RMSE={torch.sqrt((pred_stress_norm_masked - target_stress_norm_masked).pow(2).mean()).item():.6f}")
                    print(f"    Target std={target_std:.6f}, Pred std={pred_std:.6f}")
                    if pred_std < 0.1 * target_std:
                        print(f"    ⚠ WARNING: Prediction std is {target_std/pred_std:.1f}x smaller than target std!")
                        print(f"       Model is not learning the stress distribution, only predicting a constant offset.")
                    
                    # Check denormalized values
                    pred_stress_denorm = pred_stress_norm_masked * dataset.stress_std + dataset.stress_mean
                    target_stress_denorm = target_stress_norm_masked * dataset.stress_std + dataset.stress_mean
                    print(f"  Denormalized (for reference):")
                    print(f"    pred: mean={pred_stress_denorm.mean().item():.2f}, std={pred_stress_denorm.std().item():.2f}")
                    print(f"    target: mean={target_stress_denorm.mean().item():.2f}, std={target_stress_denorm.std().item():.2f}")
                    denorm_rmse = torch.sqrt((pred_stress_denorm - target_stress_denorm).pow(2).mean()).item()
                    print(f"    Denormalized RMSE={denorm_rmse:.2f}")
                    
                    print(f"  ====================================\n")
            
            # Combined loss: velocity + stress
            # Don't divide by valid_timesteps here - we'll accumulate and divide at the end
            # This allows proper gradient accumulation across all valid timesteps
            if use_adaptive_scaling:
                # Adaptive scaling: Update EMA of loss scales, then normalize both losses
                # Update scales using exponential moving average (only on first valid timestep of each batch)
                if valid_timesteps == 1:
                    # Initialize or update scales with EMA
                    current_vel_scale = max(loss_vel.item(), 0.1)  # Prevent division by zero
                    current_stress_scale = max(loss_stress.item(), 1.0)
                    
                    # EMA update: new_scale = momentum * old_scale + (1 - momentum) * current_scale
                    model.vel_loss_scale = ema_momentum * model.vel_loss_scale + (1 - ema_momentum) * current_vel_scale
                    model.stress_loss_scale = ema_momentum * model.stress_loss_scale + (1 - ema_momentum) * current_stress_scale
                
                # Normalize both losses to ~1.0 magnitude (both terms are O(1) after normalization)
                loss_vel_normalized = loss_vel / (model.vel_loss_scale + 1e-8)
                loss_stress_normalized = loss_stress / (model.stress_loss_scale + 1e-8)
                
                # Combine with equal importance (both are now O(1))
                loss = loss_vel_normalized + loss_stress_normalized
                
                # Track stress loss scale changes (minimal output)
                if batch_idx == 0 and valid_timesteps == 1 and epoch > 0 and hasattr(model, '_prev_stress_scale'):
                    scale_change = abs(model.stress_loss_scale - model._prev_stress_scale) / (model._prev_stress_scale + 1e-8)
                    if scale_change < 0.01:
                        print(f"  ⚠ Stress loss scale stuck (change: {scale_change*100:.2f}%) - stress loss not decreasing")
                if batch_idx == 0 and valid_timesteps == 1:
                    model._prev_stress_scale = model.stress_loss_scale
            else:
                # Fixed weight approach (original)
                loss = velocity_loss_weight * loss_vel + loss_stress
            
            # Backward pass (accumulate gradients across timesteps)
            loss.backward()
            
            # Accumulate losses (raw values, we'll average over valid_timesteps at the end)
            epoch_loss_batch += loss.item()
            epoch_loss_vel_batch += loss_vel.item()
            epoch_loss_stress_batch += loss_stress.item()
            
            # Track stress loss history to detect if it's stuck
            if batch_idx == 0 and valid_timesteps == 1:
                if not hasattr(model, '_stress_loss_history'):
                    model._stress_loss_history = []
                model._stress_loss_history.append(loss_stress.item())
                
                # Check if stress loss is decreasing (only warn if stuck)
                if len(model._stress_loss_history) > 10:
                    recent_losses = model._stress_loss_history[-10:]
                    if max(recent_losses) - min(recent_losses) < 1.0:  # Less than 1.0 change in last 10 steps
                        print(f"  ⚠ Stress loss STUCK at ~{loss_stress.item():.2f} (not learning)")
            
            # Check stress head gradients (essential for debugging stress learning)
            if batch_idx == 0 and t == min(5, end_t-1) and valid_timesteps > 0 and (epoch == 0 or epoch % 50 == 0):
                print("\n=== STRESS HEAD GRADIENT CHECK ===")
                
                # Stress head - check ALL layers
                stress_head_grads = []
                for i, layer in enumerate(model.stress_head):
                    if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                        stress_grad_norm = layer.weight.grad.norm().item()
                        stress_grad_mean = layer.weight.grad.abs().mean().item()
                        stress_head_grads.append((i, stress_grad_norm, stress_grad_mean))
                        print(f"  Stress head layer {i}: norm={stress_grad_norm:.8f}, mean_abs={stress_grad_mean:.8f}")
                        if stress_grad_norm < 1e-8:
                            print(f"    ⚠ Gradient near zero! (vanishing gradient)")
                
                if len(stress_head_grads) == 0:
                    print("  ⚠ CRITICAL: No gradients in stress head! Not learning!")
                else:
                    # Compare with velocity head for context
                    if hasattr(model, 'phi_v'):
                        phi_v_grads = []
                        for i, layer in enumerate(model.phi_v):
                            if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                                phi_v_grad_norm = layer.weight.grad.norm().item()
                                phi_v_grads.append((i, phi_v_grad_norm))
                        if phi_v_grads:
                            vel_max_grad = max([g[1] for g in phi_v_grads])
                            stress_max_grad = max([g[1] for g in stress_head_grads])
                            ratio = vel_max_grad / (stress_max_grad + 1e-10)
                            if ratio > 100:
                                print(f"  ⚠ Velocity gradients {ratio:.1f}x larger than stress - stress head may be under-learning")
                
                print("==================================\n")
        
        # Compute rollout errors on denormalized values (last 1 and last 50 timesteps)
        if len(rollout_pred_vel) > 0:
            # Stack all stored predictions and targets
            # Each is a list of (B, N, ...) tensors
            num_stored_timesteps = len(rollout_pred_vel)
            
            # Compute errors for last 1 and last 50 timesteps
            for horizon in [1, 50]:
                if num_stored_timesteps >= horizon:
                    # Get last 'horizon' timesteps
                    last_pred_vel = rollout_pred_vel[-horizon:]  # List of (B, N, 3)
                    last_pred_stress = rollout_pred_stress[-horizon:]  # List of (B, N, 1)
                    last_target_vel = rollout_target_vel[-horizon:]  # List of (B, N, 3)
                    last_target_stress = rollout_target_stress[-horizon:]  # List of (B, N, 1)
                    last_velocity_mask = rollout_velocity_mask_list[-horizon:]  # List of (B, N)
                    last_stress_mask = rollout_stress_mask_list[-horizon:]  # List of (B, N)
                    
                    # Compute velocity error (only on masked nodes)
                    vel_errors = []
                    stress_errors = []
                    
                    for i in range(horizon):
                        pred_v = last_pred_vel[i]  # (B, N, 3)
                        target_v = last_target_vel[i]  # (B, N, 3)
                        vel_mask = last_velocity_mask[i]  # (B, N)
                        
                        pred_s = last_pred_stress[i]  # (B, N, 1)
                        target_s = last_target_stress[i]  # (B, N, 1)
                        stress_mask = last_stress_mask[i]  # (B, N)
                        
                        # Velocity error: MSE on masked nodes only
                        vel_error = (pred_v - target_v) ** 2  # (B, N, 3)
                        vel_error_per_node = torch.sum(vel_error, dim=-1)  # (B, N)
                        vel_error_masked = vel_error_per_node * vel_mask  # (B, N)
                        if vel_mask.sum() > 0:
                            vel_errors.append(vel_error_masked.sum().item() / vel_mask.sum().item())
                        
                        # Stress error: MSE on masked nodes only
                        stress_error = (pred_s.squeeze(-1) - target_s.squeeze(-1)) ** 2  # (B, N)
                        stress_error_masked = stress_error * stress_mask  # (B, N)
                        if stress_mask.sum() > 0:
                            stress_errors.append(stress_error_masked.sum().item() / stress_mask.sum().item())
                    
                    # Average errors over the horizon
                    avg_vel_error = np.mean(vel_errors) if len(vel_errors) > 0 else 0.0
                    avg_stress_error = np.mean(stress_errors) if len(stress_errors) > 0 else 0.0
                    
                    # Display rollout errors (only for first batch of each epoch to avoid clutter)
                    if batch_idx == 0:
                        print(f"  Rollout error (denormalized, last {horizon} timestep{'s' if horizon > 1 else ''}):")
                        print(f"    Velocity MSE: {avg_vel_error:.6f} m²/s²")
                        print(f"    Stress MSE: {avg_stress_error:.6f} Pa²")
        
        # Gradient clipping and optimizer step after all timesteps in sequence
        if valid_timesteps > 0:  # Only step if we had valid timesteps
            # Average losses over valid timesteps for reporting
            avg_loss = epoch_loss_batch / valid_timesteps if valid_timesteps > 0 else 0.0
            avg_loss_vel = epoch_loss_vel_batch / valid_timesteps if valid_timesteps > 0 else 0.0
            avg_loss_stress = epoch_loss_stress_batch / valid_timesteps if valid_timesteps > 0 else 0.0
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Store averaged losses
            total_loss += avg_loss
            total_loss_vel += avg_loss_vel
            total_loss_stress += avg_loss_stress
            num_samples += 1
            
            # Debug: print accumulated losses
        else:
            # No valid timesteps - skip optimizer step
            # Reset optimizer to avoid issues
            optimizer.zero_grad()
        
        # Update progress bar
        if valid_timesteps > 0:
            avg_loss = epoch_loss_batch / valid_timesteps
            avg_loss_vel = epoch_loss_vel_batch / valid_timesteps
            avg_loss_stress = epoch_loss_stress_batch / valid_timesteps
            pbar.set_postfix({
                'loss': f'{avg_loss:.6f}',
                'vel': f'{avg_loss_vel:.6f}',
                'stress': f'{avg_loss_stress:.6f}',
                'valid_t': f'{valid_timesteps}/{end_t-1}'
            })
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_loss_vel = total_loss_vel / num_samples if num_samples > 0 else 0.0
    avg_loss_stress = total_loss_stress / num_samples if num_samples > 0 else 0.0
    
    return avg_loss, avg_loss_vel, avg_loss_stress


def main():
    parser = argparse.ArgumentParser(description='Train EGNN on deforming plate dataset')
    parser.add_argument('--data_dir', type=str, default='data/deforming_plate',
                       help='Directory containing dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/egnn',
                       help='Directory to save checkpoints')
    parser.add_argument('--dataset_fraction', type=float, default=0.1,
                       help='Fraction of dataset to use (0.0 to 1.0)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (usually 1 for trajectories)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for EGNN')
    parser.add_argument('--depth', type=int, default=4,
                       help='Depth of EGNN layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions and ground truth values')
    parser.add_argument('--save_predictions_every', type=int, default=1,
                       help='Save predictions every N epochs (default: 1, every epoch)')
    parser.add_argument('--num_trajectories_for_predictions', type=int, default=None,
                       help='Number of trajectories to save predictions for (default: all)')
    parser.add_argument('--velocity_loss_weight', type=float, default=50000.0,
                       help='Weight for velocity loss relative to stress loss (default: 50000.0 to balance with stress). Only used if --use_adaptive_scaling=False')
    parser.add_argument('--use_adaptive_scaling', action='store_true',
                       help='Use adaptive loss scaling with EMA instead of fixed velocity_loss_weight')
    parser.add_argument('--max_timesteps', type=int, default=None,
                       help='Maximum number of timesteps to use for training (default: None = use all 400 timesteps)')
    parser.add_argument('--min_stress_threshold', type=float, default=5000.0,
                       help='Minimum mean stress threshold for training on a timestep (default: 5000.0). Lower values allow training on early timesteps with low stress.')
    parser.add_argument('--warmup_fraction', type=float, default=0.025,
                       help='Fraction of timesteps to skip at the beginning (warmup period). Default: 0.025 (2.5%%). Set to 0.0 to train from t=1.')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    train_tfrecord = os.path.join(args.data_dir, 'train.tfrecord')
    meta_path = os.path.join(args.data_dir, 'meta.json')
    
    if not os.path.exists(train_tfrecord):
        raise FileNotFoundError(f"Training data not found: {train_tfrecord}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    dataset = DeformingPlateDataset(
        train_tfrecord, meta_path, 
        dataset_fraction=args.dataset_fraction
    )
    
    # Ensure custom_collate_fn is callable
    if not callable(custom_collate_fn):
        raise TypeError(f"custom_collate_fn must be callable, got {type(custom_collate_fn)}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid issues with tfrecord loading
        collate_fn=custom_collate_fn  # Handle variable-sized graphs
    )
    
    # Verify the DataLoader is using our custom collate function
    if dataloader.collate_fn is not custom_collate_fn:
        raise RuntimeError(
            f"DataLoader is not using custom_collate_fn! "
            f"Expected {custom_collate_fn}, got {dataloader.collate_fn}"
        )
    
    # Save normalization statistics to JSON file for consistency
    norm_stats_path = os.path.join(args.checkpoint_dir, 'norm_stats.json')
    dataset.save_norm_stats(norm_stats_path)
    
    # Create a separate dataloader for saving predictions (no shuffle)
    # Use same normalization stats as training dataset
    # CRITICAL FIX: Ensure eval_dataset has enough trajectories for predictions
    # If num_trajectories_for_predictions is specified, use at least that many
    # Otherwise, use the same number as training dataset
    if args.save_predictions:
        if args.num_trajectories_for_predictions is not None:
            # Use at least num_trajectories_for_predictions trajectories
            eval_num_trajectories = max(args.num_trajectories_for_predictions, len(dataset))
        else:
            # Use same as training dataset
            eval_num_trajectories = len(dataset)
        
        eval_dataset = DeformingPlateDataset(
            train_tfrecord, meta_path, 
            num_trajectories=eval_num_trajectories,  # Explicitly set number of trajectories
            dataset_fraction=None,  # Don't use fraction, use explicit num_trajectories
            norm_stats_path=norm_stats_path,  # Load from saved file
            compute_norm_stats=False  # Use saved stats
        )
        print(f"Eval dataset for predictions: {len(eval_dataset)} trajectories "
              f"(will save {args.num_trajectories_for_predictions if args.num_trajectories_for_predictions else len(eval_dataset)} trajectories)")
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn  # Handle variable-sized graphs
        )
    else:
        eval_dataset = None
        eval_dataloader = None
    
    # Determine feature dimensions from first sample
    sample = dataset[0]
    feat_dim = sample['feats'].shape[-1]  # Feature dimension
    print(f"Feature dimension: {feat_dim} (expected: 8 = [position(3), actuation(3), node_type_one_hot(2)])")
    assert feat_dim == 8, f"Expected 8D input features, got {feat_dim}D"
    
    # Verify target shapes
    print(f"Target velocity shape: {sample['target_vel'].shape} (expected: (T, N, 3))")
    print(f"Target stress shape: {sample['target_stress'].shape} (expected: (T, N, 1))")
    print(f"Coordinates shape: {sample['coors'].shape} (expected: (T, N, 3))")
    print(f"Cells shape: {sample.get('cells', 'N/A')} (for connectivity only)")
    
    # Create model following paper architecture
    # Get average number of nodes for C initialization (C = 1/(N-1))
    sample_coors = sample['coors']
    num_nodes_avg = int(sample_coors.shape[1])  # Use actual number of nodes from first sample
    
    model = MeshEGNN(
        in_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_nodes_avg=num_nodes_avg
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    train_losses = []
    train_losses_vel = []
    train_losses_stress = []
    best_loss = float('inf')
    
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Dataset fraction: {args.dataset_fraction*100:.1f}% ({len(dataset)} trajectories)")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    if args.max_timesteps is not None:
        print(f"Max timesteps per trajectory: {args.max_timesteps} (out of 400 total)")
    else:
        print(f"Max timesteps per trajectory: All 400 timesteps")
    print(f"Warmup fraction: {args.warmup_fraction*100:.1f}% (skipping first {int(400 * args.warmup_fraction)} timesteps)")
    print(f"Min stress threshold: {args.min_stress_threshold:.1f}")
    if args.use_adaptive_scaling:
        print(f"Loss scaling: ADAPTIVE (EMA-based)")
    else:
        print(f"Velocity loss weight: {args.velocity_loss_weight}")
    print("-" * 70)
    
    for epoch in range(args.num_epochs):
        avg_loss, avg_loss_vel, avg_loss_stress = train_epoch(
            model, dataloader, optimizer, device, epoch, args.velocity_loss_weight, 
            dataset=dataset, use_adaptive_scaling=args.use_adaptive_scaling,
            max_timesteps=args.max_timesteps, min_stress_threshold=args.min_stress_threshold,
            warmup_fraction=args.warmup_fraction
        )
        train_losses.append(avg_loss)
        train_losses_vel.append(avg_loss_vel)
        train_losses_stress.append(avg_loss_stress)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Total Loss: {avg_loss:.6f} "
              f"(Vel: {avg_loss_vel:.6f}, Stress: {avg_loss_stress:.6f})")
        
        # Save predictions if requested
        if args.save_predictions and (epoch + 1) % args.save_predictions_every == 0:
            predictions_dir = os.path.join(args.checkpoint_dir, 'predictions')
            try:
                save_predictions(
                    model, eval_dataloader, device, predictions_dir, epoch,
                    dataset=eval_dataset,  # Pass dataset for denormalization
                    num_trajectories=args.num_trajectories_for_predictions,
                    max_timesteps=args.max_timesteps,  # Use same max_timesteps as training
                    use_autoregressive=True  # Use autoregressive prediction (no teacher forcing) for evaluation
                )
            except Exception as e:
                print(f"  ⚠ Warning: Failed to save predictions for epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  -> Saved best model (loss: {avg_loss:.6f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, checkpoint_path)
    
    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final loss: {train_losses[-1]:.6f}")
    print(f"Final velocity loss: {train_losses_vel[-1]:.6f}")
    print(f"Final stress loss: {train_losses_stress[-1]:.6f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()

