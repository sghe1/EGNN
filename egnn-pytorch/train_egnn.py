#!/usr/bin/env python3
"""
Training script for EGNN on deforming plate dataset.
Uses data_loader_egnn.py to load data and MeshEGNN model.
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

# Scaling constants
VELOCITY_SCALE = 1e4  # rescales tiny velocities (~1e-4) to O(1) for stable training

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
egnn_pytorch_dir = os.path.join(script_dir, 'egnn-pytorch')
sys.path.insert(0, script_dir)
sys.path.insert(0, egnn_pytorch_dir)

from data_loader_egnn import data_loader_egnn, trajectory_to_egnn_inputs, load_raw_trajectory_from_tfrecord
from tfrecord.reader import tfrecord_loader

# Import EGNN - try different paths
try:
    from myEGNN.EGNN import MeshEGNN
except ImportError:
    try:
        # Try importing from egnn-pytorch package
        from egnn_pytorch.egnn_pytorch import EGNN_Network
    except ImportError:
        # Try relative import
        sys.path.insert(0, os.path.join(egnn_pytorch_dir, 'egnn_pytorch'))
        from egnn_pytorch import EGNN_Network
    
    import torch.nn as nn
    
    import torch.nn.functional as F
    
    class MeshEGNN(nn.Module):
        """
        Mesh EGNN following the paper formulas:
        - v_i^{l+1} = phi_v(h_i^l) * v_i^init + C * sum_j (x_i^l - x_j^l) * phi_x(m_ij)
        - x_i^{l+1} = x_i^l + v_i^{l+1}
        """
        def __init__(self, in_dim, hidden_dim, out_dim=None, edge_dim=0, depth=4, C=1.0):
            super().__init__()
            self.input_mlp  = nn.Linear(in_dim, hidden_dim)
            self.egnn       = EGNN_Network(
                num_tokens=None,
                dim=hidden_dim,
                depth=depth,
                edge_dim=edge_dim,
                only_sparse_neighbors=True,
            )
            
            # phi_v: MLP that takes h and outputs a 1D vector to modulate initial velocity
            self.phi_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # φ_e: edge/message MLP, builds m_ij from (h_i, h_j, ||x_i - x_j||^2, a_ij)
            # Here we have no explicit edge attributes a_ij, so we omit them.
            self.phi_e = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # φ_x: takes m_ij and outputs a scalar weight (Eq. (7))
            # Add tanh to constrain output to [-1, 1] to prevent explosion
            self.phi_x = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()  # Constrain to [-1, 1] to prevent large neighbor terms
            )
            
            # Constant C (can be learnable) - initialize to small value to match velocity scale
            # Velocities are ~1e-4, so C should be small (e.g., 1e-4 or 1e-5)
            self.C = nn.Parameter(torch.tensor(C * 1e-4, dtype=torch.float32))  # Scale down C
            
            # Stress head (unchanged)
            self.stress_head = nn.Linear(hidden_dim, 1)
            
            # Extract initial velocity from features: feats = [node_type(1), vel(3), acc(3), stress(1)]
            self.vel_init_start_idx = 1  # After node_type
            self.vel_init_end_idx = 4    # vel has 3 dims

        def forward(self, feats, coors, adj_mat, edges=None):
            """
            Args:
                feats: (B, N, in_dim) - features including [node_type(1), vel(3), acc(3), stress(1)]
                coors: (B, N, 3) - current node coordinates
                adj_mat: (B, N, N) or (N, N) - adjacency matrix
            Returns:
                pred_vel: (B, N, 3) - predicted velocity
                pred_stress: (B, N, 1) - predicted stress
                pred_coors: (B, N, 3) - predicted coordinates (coors + pred_vel)
            """
            B, N, _ = coors.shape
            
            # Extract initial velocity from features
            v_init = feats[:, :, self.vel_init_start_idx:self.vel_init_end_idx]  # (B, N, 3)
            
            # Project features to hidden dimension and pass through EGNN
            h = self.input_mlp(feats)  # (B, N, hidden_dim)
            h, _ = self.egnn(h, coors, adj_mat=adj_mat, edges=edges, mask=None)
            
            # Compute messages m_ij for all pairs (we'll use only neighbors via adj_mat)
            # For each node i, compute interaction with all neighbors j
            # Expand h for pairwise computation
            h_i = h.unsqueeze(2)  # (B, N, 1, hidden_dim)
            h_j = h.unsqueeze(1)  # (B, 1, N, hidden_dim)
            
            # Compute relative positions and squared distances
            coors_i = coors.unsqueeze(2)  # (B, N, 1, 3)
            coors_j = coors.unsqueeze(1)  # (B, 1, N, 3)
            rel_pos = coors_i - coors_j  # (B, N, N, 3)
            sq_dist = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)  # (B, N, N, 1)
            
            # Concatenate h_i, h_j, and squared distance to form input to φ_e
            h_i_expanded = h_i.expand(-1, -1, N, -1)   # (B, N, N, hidden_dim)
            h_j_expanded = h_j.expand(-1, N, -1, -1)   # (B, N, N, hidden_dim)
            message_input = torch.cat([h_i_expanded, h_j_expanded, sq_dist], dim=-1)  # (B, N, N, 2*hidden_dim+1)

            # m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
            m_ij = self.phi_e(message_input)          # (B, N, N, hidden_dim)

            # φ_x(m_ij) -> scalar per edge
            phi_x_output = self.phi_x(m_ij).squeeze(-1)  # (B, N, N)
            
            # Apply adjacency mask (only consider neighbors)
            if len(adj_mat.shape) == 2:
                adj_mat = adj_mat.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
            adj_mask = adj_mat.float()  # (B, N, N)
            
            # Mask out self-connections (j != i)
            identity = torch.eye(N, device=adj_mat.device, dtype=torch.float32)
            if len(identity.shape) == 2:
                identity = identity.unsqueeze(0).expand(B, -1, -1)
            adj_mask = adj_mask * (1 - identity)  # Remove self-connections
            
            # Compute neighbor interaction term: sum_j (x_i - x_j) * phi_x(m_ij)
            # rel_pos: (B, N, N, 3), phi_x_output: (B, N, N)
            phi_x_masked = phi_x_output.unsqueeze(-1) * adj_mask.unsqueeze(-1)  # (B, N, N, 1)
            neighbor_term = torch.sum(rel_pos * phi_x_masked, dim=2)  # (B, N, 3)
            
            # Normalize neighbor_term by number of neighbors to prevent explosion
            # Count neighbors per node
            num_neighbors = adj_mask.sum(dim=2, keepdim=True)  # (B, N, 1)
            num_neighbors = torch.clamp(num_neighbors, min=1.0)  # Avoid division by zero
            neighbor_term = neighbor_term / num_neighbors  # Normalize by neighbor count
            
            # Compute scalar gate φ_v(h_i^l) (γ_i in Eq. (7))
            # Add sigmoid to constrain gamma to [0, 1] to prevent velocity explosion
            gamma = torch.sigmoid(self.phi_v(h))  # (B, N, 1) - now in [0, 1]

            # Velocity prediction (Eq. (7)):
            # v_i^{l+1} = φ_v(h_i^l) v_i^{init} + C ∑_{j≠i} (x_i^l - x_j^l) φ_x(m_ij)
            pred_vel = gamma * v_init + self.C * neighbor_term   # (B, N, 3)
            
            # Stress prediction from node embeddings
            pred_stress = self.stress_head(h)
            
            # Coordinate prediction: x_i^{l+1} = x_i^l + v_i^{l+1}
            pred_coors = coors + pred_vel
            
            return pred_vel, pred_stress, pred_coors


class DeformingPlateDataset(Dataset):
    """Dataset for deforming plate trajectories."""
    
    def __init__(self, tfrecord_path, meta_path, num_trajectories=None, dataset_fraction=1.0):
        self.tfrecord_path = tfrecord_path
        self.meta_path = meta_path
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # Determine number of trajectories
        if num_trajectories is None:
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
        """Load a trajectory and convert to EGNN format."""
        if idx >= self.available_trajectories:
            raise IndexError(
                f"Trajectory index {idx} is out of range. "
                f"Only {self.available_trajectories} trajectories available in {self.tfrecord_path}."
            )
        traj_dict = load_raw_trajectory_from_tfrecord(
            self.tfrecord_path, self.meta, idx
        )
        
        coors_seq, feats_seq, edge_index = trajectory_to_egnn_inputs(traj_dict)
        
        # Convert edge_index to adjacency matrix for EGNN
        num_nodes = coors_seq.shape[1]
        adj_mat = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        if edge_index.shape[0] == 2:  # (2, E) format
            adj_mat[edge_index[0], edge_index[1]] = True
        else:  # (E, 2) format
            adj_mat[edge_index[:, 0], edge_index[:, 1]] = True
        
        # Make symmetric (undirected graph)
        adj_mat = adj_mat | adj_mat.t()
        
        # Compute target velocity and stress
        world_pos = torch.tensor(traj_dict['world_pos'], dtype=torch.float32)
        target_vel = compute_target_velocity(world_pos)
        target_stress = torch.tensor(traj_dict['stress'], dtype=torch.float32)  # (T, N, 1)
        
        return {
            'coors': coors_seq,      # (T, N, 3)
            'feats': feats_seq,      # (T, N, feat_dim)
            'adj_mat': adj_mat,      # (N, N)
            'edge_index': edge_index, # (2, E) or (E, 2)
            'world_pos': world_pos,  # (T, N, 3)
            'target_vel': target_vel,  # (T, N, 3)
            'target_stress': target_stress  # (T, N, 1)
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


def save_predictions(model, dataloader, device, save_dir, epoch, num_trajectories=None):
    """
    Save predictions and ground truth values for a subset of trajectories.
    """
    model.eval()
    
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
                
            coors_seq = batch['coors'].to(device)
            feats_seq = batch['feats'].to(device)
            adj_mat = batch['adj_mat'].to(device)
            target_vel = batch['target_vel'].to(device)
            target_stress = batch['target_stress'].to(device)
            world_pos = batch.get('world_pos')
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
            
            traj_pred_vel = []
            traj_pred_stress = []
            traj_pred_pos = []
            traj_true_vel = []
            traj_true_stress = []
            traj_true_pos = []
            
            # Process each timestep (starting from t=1)
            for t in range(1, T):
                feats_prev = feats_seq[:, t - 1]
                coors_prev = coors_seq[:, t - 1]
                adj_mat_t = adj_mat
                target_vel_t = target_vel[:, t]
                target_stress_t = target_stress[:, t]
                target_pos_t = world_pos[:, t]
                
                # Forward pass
                # Returns: (pred_vel, pred_stress, pred_coors) where:
                #   pred_vel follows: v_i^{l+1} = phi_v(h_i^l) * v_i^init + C * sum_j (x_i - x_j) * phi_x(m_ij)
                #   pred_coors = coors_prev + pred_vel
                pred_vel, pred_stress, pred_coors = model(feats_prev, coors_prev, adj_mat_t)
                
                # Convert to numpy and store
                traj_pred_vel.append(pred_vel.cpu().numpy())
                traj_pred_stress.append(pred_stress.cpu().numpy())
                traj_pred_pos.append(pred_coors.cpu().numpy())
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


def train_epoch(model, dataloader, optimizer, device, epoch, velocity_loss_weight=2.0):
    """
    Train for one epoch.
    
    Args:
        velocity_loss_weight: Weight for velocity loss to balance with stress loss.
                             Since velocity values are ~0.0002 and stress ~18000,
                             we need a large weight to balance the losses.
    """
    model.train()
    total_loss = 0.0
    total_loss_vel = 0.0
    total_loss_stress = 0.0
    total_loss_pos = 0.0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        coors_seq = batch['coors'].to(device)  # (B, T, N, 3) or (T, N, 3)
        feats_seq = batch['feats'].to(device)  # (B, T, N, feat_dim) or (T, N, feat_dim)
        adj_mat = batch['adj_mat'].to(device)   # (B, N, N) or (N, N)
        target_vel = batch['target_vel'].to(device)  # (B, T, N, 3) or (T, N, 3)
        target_stress = batch['target_stress'].to(device)  # (B, T, N, 1) or (T, N, 1)
        world_pos = batch['world_pos'].to(device)  # (B, T, N, 3) or (T, N, 3)
        
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
        epoch_loss_pos_batch = 0.0
        
        # Process each timestep (starting from t=1 since we need previous position for velocity)
        for t in range(1, T):
            # Previous timestep inputs (to predict state at t)
            feats_prev = feats_seq[:, t - 1]      # (B, N, feat_dim)
            coors_prev = coors_seq[:, t - 1]      # (B, N, 3)
            adj_mat_t = adj_mat            # (B, N, N) or (N, N)
            target_vel_t = target_vel[:, t]  # (B, N, 3)
            target_stress_t = target_stress[:, t]  # (B, N, 1)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Model expects: feats (B, N, feat_dim), coors (B, N, 3), adj_mat (B, N, N)
            # Returns: (pred_vel, pred_stress, pred_coors) where:
            #   pred_vel follows: v_i^{l+1} = phi_v(h_i^l) * v_i^init + C * sum_j (x_i - x_j) * phi_x(m_ij)
            #   pred_coors = coors_prev + pred_vel
            pred_vel, pred_stress, pred_coors = model(feats_prev, coors_prev, adj_mat_t)
            
            # Extract node_type from features (first dimension, index 0)
            # node_type is stored as float in features: [node_type(1), vel(3), acc(3), stress(1)]
            # node_type values: typically 0=boundary, 1=normal, 2=inflow, 3=outflow, etc.
            node_type = feats_prev[:, :, 0]  # (B, N)
            # Create mask for node_type == 1 (normal nodes only)
            # Loss is computed only on normal nodes, excluding boundary and special nodes
            node_mask = (node_type == 1.0).float()  # (B, N)
            
            # Compute losses with weighting (apply velocity scaling)
            # Only compute loss on nodes where node_type == 1
            scaled_pred_vel = pred_vel * VELOCITY_SCALE
            scaled_target_vel = target_vel_t * VELOCITY_SCALE
            
            # Velocity loss: compute per-node, then mask and average
            vel_error = (scaled_pred_vel - scaled_target_vel) ** 2  # (B, N, 3)
            vel_error_per_node = torch.sum(vel_error, dim=-1)  # (B, N)
            vel_error_masked = vel_error_per_node * node_mask  # (B, N)
            loss_vel = vel_error_masked.sum() / (node_mask.sum() + 1e-8)  # Average over masked nodes
            
            # Stress loss: compute per-node, then mask and average
            stress_error = (pred_stress.squeeze(-1) - target_stress_t.squeeze(-1)) ** 2  # (B, N)
            stress_error_masked = stress_error * node_mask  # (B, N)
            loss_stress = stress_error_masked.sum() / (node_mask.sum() + 1e-8)  # Average over masked nodes
            
            # Combined loss: velocity + stress
            loss = velocity_loss_weight * loss_vel + loss_stress
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss_batch += loss.item()
            epoch_loss_vel_batch += loss_vel.item()
            epoch_loss_stress_batch += loss_stress.item()
            num_samples += 1
        
        total_loss += epoch_loss_batch
        total_loss_vel += epoch_loss_vel_batch
        total_loss_stress += epoch_loss_stress_batch
        
        # Update progress bar
        if batch_idx % 10 == 0:
            avg_loss = epoch_loss_batch / (T - 1) if T > 1 else 0.0
            avg_loss_vel = epoch_loss_vel_batch / (T - 1) if T > 1 else 0.0
            avg_loss_stress = epoch_loss_stress_batch / (T - 1) if T > 1 else 0.0
            pbar.set_postfix({
                'loss': f'{avg_loss:.6f}',
                'vel': f'{avg_loss_vel:.6f}',
                'stress': f'{avg_loss_stress:.6f}'
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
    parser.add_argument('--velocity_loss_weight', type=float, default=1.0,
                       help='Weight for velocity loss relative to stress loss (default: 1.0)')
    
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
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid issues with tfrecord loading
    )
    
    # Create a separate dataloader for saving predictions (no shuffle)
    eval_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    ) if args.save_predictions else None
    
    # Determine feature dimensions from first sample
    sample = dataset[0]
    feat_dim = sample['feats'].shape[-1]  # Feature dimension
    print(f"Feature dimension: {feat_dim}")
    
    # Create model
    # Output: velocity (3D) + stress (1D) = 4D
    model = MeshEGNN(
        in_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        edge_dim=0,
        depth=args.depth
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
    print(f"Velocity loss weight: {args.velocity_loss_weight}")
    print("-" * 70)
    
    for epoch in range(args.num_epochs):
        avg_loss, avg_loss_vel, avg_loss_stress = train_epoch(
            model, dataloader, optimizer, device, epoch, args.velocity_loss_weight
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
                    num_trajectories=args.num_trajectories_for_predictions
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

