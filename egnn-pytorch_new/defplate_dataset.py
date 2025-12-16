"""
Dataset class for deforming plate trajectories with normalization.

This module contains the DeformingPlateDataset class which handles:
- Loading trajectories from TFRecord files
- Computing and storing normalization statistics
- Normalizing inputs (positions, actuation) and preparing targets
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tfrecord.reader import tfrecord_loader

from data_loader_egnn import load_raw_trajectory_from_tfrecord, build_edges_from_cells, add_proximity_edges


def compute_velocity_features(world_pos, node_type):
    """
    Compute velocity features at time t as per report:
    - For plate nodes (NORMAL, node_type==0): backward difference x(t) - x(t-1)
    - For actuator nodes (HANDLE, node_type==3): forward difference x(t+1) - x(t)
    
    Args:
        world_pos: (T, N, 3) tensor of world positions
        node_type: (N,) tensor of node types (0=NORMAL, 1=OBSTACLE, 3=HANDLE)
    
    Returns:
        velocity: (T, N, 3) tensor of velocity features
    """
    T, N, _ = world_pos.shape
    vel = torch.zeros_like(world_pos)
    node_type_flat = node_type.flatten() if node_type.ndim > 1 else node_type
    
    # Plate nodes (NORMAL, node_type==0): backward difference
    plate_mask = (node_type_flat == 0)
    if T > 1:
        vel[1:, plate_mask] = world_pos[1:, plate_mask] - world_pos[:-1, plate_mask]
        # For t=0, set velocity to zero for plate nodes
        vel[0, plate_mask] = 0.0
    
    # Actuator nodes (HANDLE, node_type==3): forward difference
    actuator_mask = (node_type_flat == 3)
    if T > 1:
        vel[:-1, actuator_mask] = world_pos[1:, actuator_mask] - world_pos[:-1, actuator_mask]
        # For last timestep, use backward difference as fallback
        vel[-1, actuator_mask] = world_pos[-1, actuator_mask] - world_pos[-2, actuator_mask]
    
    return vel


def compute_target_velocity(world_pos):
    """
    Compute target velocity for next timestep (t+1).
    For all nodes: vel[t+1] = pos[t+1] - pos[t]
    This is the standard Lagrangian velocity to predict.
    """
    T, N, _ = world_pos.shape
    vel = torch.zeros_like(world_pos)
    
    if T > 1:
        vel[:-1] = world_pos[1:] - world_pos[:-1]
        # For last timestep, use backward difference
        vel[-1] = world_pos[-1] - world_pos[-2]
    
    return vel


class DeformingPlateDataset(Dataset):
    """Dataset for deforming plate trajectories."""
    
    def __init__(self, tfrecord_path, meta_path, num_trajectories=None, dataset_fraction=1.0, 
                 norm_stats_path=None, compute_norm_stats=True, proximity_radius=0.1, max_proximity_edges_per_node=None,
                 debug_traj_id=None, debug_max_timesteps=None):
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
            proximity_radius: float, maximum distance for world-space proximity edges (report requirement)
            max_proximity_edges_per_node: optional int, maximum proximity edges per node (for memory)
            debug_traj_id: DEBUG: Specific trajectory ID to load (None = use num_trajectories/dataset_fraction)
            debug_max_timesteps: DEBUG: Maximum timesteps to use (None = use all timesteps)
        """
        self.tfrecord_path = tfrecord_path
        self.meta_path = meta_path
        self.proximity_radius = proximity_radius
        self.max_proximity_edges_per_node = max_proximity_edges_per_node
        self.debug_traj_id = debug_traj_id
        self.debug_max_timesteps = debug_max_timesteps
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # DEBUG MODE: If debug_traj_id is specified, load only that trajectory
        if debug_traj_id is not None:
            # Verify trajectory exists
            available_count = self._count_available_trajectories(max_count=debug_traj_id + 1)
            if debug_traj_id >= available_count:
                raise IndexError(
                    f"DEBUG: Trajectory ID {debug_traj_id} is out of range. "
                    f"Only {available_count} trajectories available."
                )
            self.num_trajectories = 1
            self.requested_trajectories = 1
            self.available_trajectories = available_count
            print(f"DEBUG: Loading trajectory ID {debug_traj_id} only")
            if debug_max_timesteps is not None:
                print(f"DEBUG: Will slice to first {debug_max_timesteps} timesteps")
        else:
            # Normal mode: Determine number of trajectories
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
            print(f"  Mesh position: mean={self.mesh_pos_mean.numpy()}, std={self.mesh_pos_std.item():.6f} (isotropic)")
            print(f"  World position: mean={self.pos_mean.numpy()}, std={self.pos_std.item():.6f} (isotropic)")
            print(f"  Velocity: mean={self.vel_mean.numpy()}, std={self.vel_std:.6f} (isotropic)")
            print(f"  Stress: mean={self.stress_mean:.6f}, std={self.stress_std:.6f}")
        else:
            # Default: no normalization (shouldn't be used in production)
            self._init_default_norm_stats()
    
    def _compute_all_norm_stats(self, sample_size=100):
        """
        Compute all normalization statistics following report requirements.
        
        As per report: "mesh-space position, world-space position, velocity are normalized 
        to zero mean and unit variance. The std for 3D vectors is computed globally across 
        all 3 components (same scalar scaling applied to x/y/z) to preserve direction."
        
        Computes:
        - mesh_pos_mean, mesh_pos_std: Isotropic (scalar std) for mesh positions
        - pos_mean, pos_std: Isotropic (scalar std) for world positions
        - vel_mean, vel_std: Isotropic (scalar std) for velocity
        - stress_mean, stress_std: Standard mean/std (not median/IQR)
        """
        import numpy as np
        all_mesh_positions = []
        all_positions = []
        all_stress = []
        all_velocity = []
        
        # Sample trajectories to compute statistics
        sample_trajs = min(sample_size, self.available_trajectories)
        for idx in range(sample_trajs):
            traj_dict = load_raw_trajectory_from_tfrecord(
                self.tfrecord_path, self.meta, idx
            )
            world_pos = traj_dict['world_pos']  # (T, N, 3)
            mesh_pos = traj_dict['mesh_pos']  # (N, 3) - static
            stress = traj_dict['stress']  # (T, N, 1)
            node_type = traj_dict['node_type']  # (N,)
            
            # Collect mesh positions (static, same for all timesteps)
            all_mesh_positions.append(mesh_pos)  # (N, 3)
            
            # Collect world positions (all timesteps)
            all_positions.append(world_pos.reshape(-1, 3))  # (T*N, 3)
            
            # Collect stress from ALL timesteps (report doesn't specify filtering)
            all_stress.append(stress.flatten())  # (T*N,)
            
            # Compute velocity features (backward for plate, forward for actuator)
            vel = compute_velocity_features(
                torch.tensor(world_pos, dtype=torch.float32),
                torch.tensor(node_type, dtype=torch.long)
            ).numpy()
            all_velocity.append(vel.reshape(-1, 3))  # (T*N, 3)
        
        # Concatenate all samples
        all_mesh_positions = np.concatenate(all_mesh_positions, axis=0)  # (total_samples, 3)
        all_positions = np.concatenate(all_positions, axis=0)  # (total_samples, 3)
        all_stress = np.concatenate(all_stress)  # (total_samples,)
        all_velocity = np.concatenate(all_velocity, axis=0)  # (total_samples, 3)
        
        # Mesh position normalization: isotropic (scalar std across all 3 components)
        # Mean as vector (3,), but std as scalar computed over flattened values
        self.mesh_pos_mean = torch.tensor(all_mesh_positions.mean(axis=0), dtype=torch.float32)  # (3,)
        mesh_pos_flat = all_mesh_positions.flatten()  # (total_samples * 3,)
        mesh_pos_std_scalar = float(np.std(mesh_pos_flat))
        self.mesh_pos_std = torch.tensor(max(mesh_pos_std_scalar, 1e-8), dtype=torch.float32)  # Scalar
        
        # World position normalization: isotropic (scalar std across all 3 components)
        # As per report: "std computed globally across all 3 components (same scalar scaling)"
        self.pos_mean = torch.tensor(all_positions.mean(axis=0), dtype=torch.float32)  # (3,)
        pos_flat = all_positions.flatten()  # (total_samples * 3,)
        pos_std_scalar = float(np.std(pos_flat))
        self.pos_std = torch.tensor(max(pos_std_scalar, 1e-8), dtype=torch.float32)  # Scalar
        
        # Velocity normalization: isotropic (scalar std across all 3 components)
        self.vel_mean = torch.tensor(all_velocity.mean(axis=0), dtype=torch.float32)  # (3,)
        vel_flat = all_velocity.flatten()  # (total_samples * 3,)
        vel_std_scalar = float(np.std(vel_flat))
        self.vel_std = float(max(vel_std_scalar, 1e-8))  # Scalar
        
        # Stress normalization: standard mean/std (not median/IQR)
        # As per report: "stress normalized" (implies standard normalization)
        self.stress_mean = float(np.mean(all_stress))
        self.stress_std = float(np.std(all_stress))
        if self.stress_std < 1e-8:
            self.stress_std = 1.0
        if self.stress_mean < 0:
            self.stress_mean = 0.0  # Safety: stress cannot be negative
        
        print(f"Normalization statistics computed:")
        print(f"  Mesh position: mean={self.mesh_pos_mean.numpy()}, std={self.mesh_pos_std.item():.6f} (isotropic)")
        print(f"  World position: mean={self.pos_mean.numpy()}, std={self.pos_std.item():.6f} (isotropic)")
        print(f"  Velocity: mean={self.vel_mean.numpy()}, std={self.vel_std:.6f} (isotropic)")
        print(f"  Stress: mean={self.stress_mean:.6f}, std={self.stress_std:.6f}")
    
    def _init_default_norm_stats(self):
        """Initialize default normalization stats (no normalization)."""
        self.mesh_pos_mean = torch.zeros(3, dtype=torch.float32)
        self.mesh_pos_std = torch.tensor(1.0, dtype=torch.float32)  # Scalar
        self.pos_mean = torch.zeros(3, dtype=torch.float32)
        self.pos_std = torch.tensor(1.0, dtype=torch.float32)  # Scalar
        self.vel_mean = torch.zeros(3, dtype=torch.float32)
        self.vel_std = 1.0  # Scalar
        self.stress_mean = 0.0
        self.stress_std = 1.0
    
    def _load_norm_stats(self, norm_stats_path):
        """Load normalization statistics from JSON file."""
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        
        self.mesh_pos_mean = torch.tensor(stats['mesh_pos_mean'], dtype=torch.float32)
        # Handle both old (list) and new (scalar) formats
        mesh_pos_std_val = stats['mesh_pos_std']
        if isinstance(mesh_pos_std_val, list):
            # Old format: convert to scalar (take mean or first value)
            self.mesh_pos_std = torch.tensor(float(np.mean(mesh_pos_std_val)), dtype=torch.float32)
        else:
            self.mesh_pos_std = torch.tensor(float(mesh_pos_std_val), dtype=torch.float32)
        
        self.pos_mean = torch.tensor(stats['pos_mean'], dtype=torch.float32)
        # Handle both old (list) and new (scalar) formats
        pos_std_val = stats['pos_std']
        if isinstance(pos_std_val, list):
            # Old format: convert to scalar (take mean or first value)
            self.pos_std = torch.tensor(float(np.mean(pos_std_val)), dtype=torch.float32)
        else:
            self.pos_std = torch.tensor(float(pos_std_val), dtype=torch.float32)
        
        self.vel_mean = torch.tensor(stats.get('vel_mean', [0.0, 0.0, 0.0]), dtype=torch.float32)
        self.vel_std = float(stats['vel_std'])
        self.stress_mean = float(stats['stress_mean'])
        self.stress_std = float(stats['stress_std'])
    
    def save_norm_stats(self, norm_stats_path):
        """Save normalization statistics to JSON file."""
        stats = {
            'mesh_pos_mean': self.mesh_pos_mean.tolist(),
            'mesh_pos_std': self.mesh_pos_std.item(),  # Scalar
            'pos_mean': self.pos_mean.tolist(),
            'pos_std': self.pos_std.item(),  # Scalar
            'vel_mean': self.vel_mean.tolist(),
            'vel_std': self.vel_std,  # Scalar
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
        # DEBUG MODE: If debug_traj_id is specified, always load that trajectory
        if self.debug_traj_id is not None:
            actual_idx = self.debug_traj_id
        else:
            actual_idx = idx
        """
        Load a trajectory and convert to EGNN format with normalized inputs.
        
        As per report: Node features at time t consist of:
        - Mesh-space position (3): static reference configuration
        - World-space position (3): current position, centered per timestep by subtracting centroid
        - Node type (2): one-hot encoding
        - Von Mises stress (1): scalar stress at time t
        - Lagrangian velocity (3): backward diff for plate, forward diff for actuator
        
        Total: 12D features
        
        Returns:
            Dictionary with:
            - coors: (T, N, 3) - NORMALIZED node coordinates (centered world positions)
            - feats: (T, N, 12) - NORMALIZED input features [mesh_pos(3), world_pos(3), node_type(2), stress(1), velocity(3)]
            - adj_mat: (T, N, N) - adjacency matrices per timestep (mesh + proximity edges)
            - edge_index: (E, 2) - base mesh edge connectivity
            - world_pos: (T, N, 3) - ORIGINAL world positions (for denormalization)
            - target_vel: (T, N, 3) - ORIGINAL target velocity (will be normalized in train_epoch)
            - target_stress: (T, N, 1) - ORIGINAL target stress (will be normalized in train_epoch)
        """
        if actual_idx >= self.available_trajectories:
            raise IndexError(
                f"Trajectory index {actual_idx} is out of range. "
                f"Only {self.available_trajectories} trajectories available in {self.tfrecord_path}."
            )
        traj_dict = load_raw_trajectory_from_tfrecord(
            self.tfrecord_path, self.meta, actual_idx
        )
        
        # Get original data
        world_pos = torch.tensor(traj_dict['world_pos'], dtype=torch.float32)  # (T, N, 3) - ORIGINAL
        mesh_pos = torch.tensor(traj_dict['mesh_pos'], dtype=torch.float32)  # (N, 3) - static
        stress = torch.tensor(traj_dict['stress'], dtype=torch.float32)  # (T, N, 1) - ORIGINAL
        node_type = traj_dict['node_type']  # (N, 1) or (N,)
        cells = traj_dict['cells']  # (C, 4)
        T_full, N, _ = world_pos.shape
        
        # DEBUG MODE: Slice timesteps if debug_max_timesteps is specified
        if self.debug_max_timesteps is not None:
            T = min(self.debug_max_timesteps, T_full)
            world_pos = world_pos[:T]  # (T, N, 3)
            stress = stress[:T]  # (T, N, 1)
            print(f"DEBUG: Sliced trajectory from {T_full} to {T} timesteps")
        else:
            T = T_full
        
        # Flatten node_type if needed
        node_type_flat = node_type.flatten() if node_type.ndim > 1 else node_type
        node_type_tensor = torch.tensor(node_type_flat, dtype=torch.long)  # (N,)
        
        # Convert node_type to one-hot (2D encoding: NORMAL=[0,0], OBSTACLE=[1,0], HANDLE=[0,1])
        node_type_one_hot = np.zeros((N, 2), dtype=np.float32)
        node_type_one_hot[node_type_flat == 0, :] = [0.0, 0.0]  # Type 0 (NORMAL) -> [0, 0]
        node_type_one_hot[node_type_flat == 1, :] = [1.0, 0.0]  # Type 1 (OBSTACLE) -> [1, 0]
        node_type_one_hot[node_type_flat == 3, :] = [0.0, 1.0]  # Type 3 (HANDLE) -> [0, 1]
        node_type_one_hot = torch.tensor(node_type_one_hot, dtype=torch.float32)  # (N, 2)
        
        # Compute velocity features (backward for plate, forward for actuator)
        velocity_features = compute_velocity_features(world_pos, node_type_tensor)  # (T, N, 3) - ORIGINAL
        
        # Center world positions per timestep by subtracting centroid (as per report)
        world_pos_centered = world_pos.clone()
        for t in range(T):
            centroid = world_pos[t].mean(dim=0)  # (3,)
            world_pos_centered[t] = world_pos[t] - centroid
        
        # Normalize mesh positions: (mesh_pos - mesh_pos_mean) / mesh_pos_std (isotropic)
        mesh_pos_normalized = (mesh_pos - self.mesh_pos_mean) / self.mesh_pos_std  # (N, 3)
        
        # Normalize world positions: (world_pos_centered - pos_mean) / pos_std (isotropic)
        # pos_mean is (3,), pos_std is scalar
        world_pos_normalized = (world_pos_centered - self.pos_mean.unsqueeze(0).unsqueeze(0)) / self.pos_std  # (T, N, 3)
        
        # Normalize velocity: (velocity - vel_mean) / vel_std (isotropic)
        # vel_mean is (3,), vel_std is scalar
        velocity_normalized = (velocity_features - self.vel_mean.unsqueeze(0).unsqueeze(0)) / self.vel_std  # (T, N, 3)
        
        # Normalize stress: (stress - stress_mean) / stress_std
        stress_normalized = (stress - self.stress_mean) / self.stress_std  # (T, N, 1)
        
        # Construct 12D features: [mesh_pos(3), world_pos(3), node_type(2), stress(1), velocity(3)]
        feats_normalized = torch.cat([
            mesh_pos_normalized.unsqueeze(0).expand(T, -1, -1),  # (T, N, 3) - mesh position (static, repeated)
            world_pos_normalized,                                 # (T, N, 3) - centered world position
            node_type_one_hot.unsqueeze(0).expand(T, -1, -1),   # (T, N, 2) - node type (not normalized)
            stress_normalized,                                    # (T, N, 1) - normalized stress
            velocity_normalized                                   # (T, N, 3) - normalized velocity
        ], dim=-1)  # (T, N, 12)
        
        # Build base mesh edge connectivity
        base_edge_index = build_edges_from_cells(cells, num_nodes=N)  # (E_mesh, 2)
        
        # Add proximity edges per timestep (dynamic, recomputed for each t)
        # As per report: "graph is augmented with world-space proximity edges"
        adj_mat_list = []
        edge_index_list = []
        for t in range(T):
            # Add proximity edges using world positions at time t
            edge_index_t = add_proximity_edges(
                base_edge_index,
                world_pos[t].numpy(),  # (N, 3) - use original (not centered) positions for distance
                proximity_radius=self.proximity_radius,
                max_edges_per_node=self.max_proximity_edges_per_node
            )  # (E_total, 2)
            
            # Build adjacency matrix for this timestep
            adj_mat_t = torch.zeros(N, N, dtype=torch.bool)
            if edge_index_t.shape[0] > 0:
                if edge_index_t.shape[0] == 2 and edge_index_t.shape[1] > 0:
                    adj_mat_t[edge_index_t[0], edge_index_t[1]] = True
                else:
                    adj_mat_t[edge_index_t[:, 0], edge_index_t[:, 1]] = True
            adj_mat_t = adj_mat_t | adj_mat_t.t()  # Make symmetric
            
            adj_mat_list.append(adj_mat_t)
            edge_index_list.append(edge_index_t)
        
        # Stack adjacency matrices: (T, N, N)
        adj_mat = torch.stack(adj_mat_list, dim=0)  # (T, N, N)
        
        # Compute targets in ORIGINAL scale (will be normalized in train_epoch)
        target_vel = compute_target_velocity(world_pos)  # (T, N, 3) - ORIGINAL
        target_stress = stress  # (T, N, 1) - ORIGINAL (already loaded)
        
        # Verify shapes
        assert feats_normalized.shape[-1] == 12, f"Expected 12D features, got {feats_normalized.shape[-1]}D"
        assert target_vel.shape == (T, N, 3), f"Target velocity shape mismatch: {target_vel.shape}"
        assert target_stress.shape == (T, N, 1), f"Target stress shape mismatch: {target_stress.shape}"
        assert adj_mat.shape == (T, N, N), f"Adjacency matrix shape mismatch: {adj_mat.shape}"
        
        return {
            'coors': world_pos_normalized,  # (T, N, 3) - NORMALIZED centered coordinates
            'feats': feats_normalized,      # (T, N, 12) - NORMALIZED features
            'adj_mat': adj_mat,             # (T, N, N) - adjacency matrices per timestep
            'edge_index': base_edge_index,  # (E_mesh, 2) - base mesh edges
            'world_pos': world_pos,         # (T, N, 3) - ORIGINAL positions (for denormalization)
            'target_vel': target_vel,       # (T, N, 3) - ORIGINAL velocity (will be normalized in train_epoch)
            'target_stress': target_stress  # (T, N, 1) - ORIGINAL stress (will be normalized in train_epoch)
        }
