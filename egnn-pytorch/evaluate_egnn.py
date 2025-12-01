#!/usr/bin/env python3
"""
Evaluation script for EGNN on deforming plate dataset.
Generates predictions for velocity and stress and saves them for comparison with true values.
"""

import os
import sys
import json
import pickle
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
egnn_pytorch_dir = os.path.join(script_dir, 'egnn-pytorch')
sys.path.insert(0, script_dir)
sys.path.insert(0, egnn_pytorch_dir)

from data_loader_egnn import data_loader_egnn, trajectory_to_egnn_inputs, load_raw_trajectory_from_tfrecord
from train_egnn import DeformingPlateDataset, compute_target_velocity, MeshEGNN

def evaluate_model(model, dataloader, device, save_dir, num_trajectories=None):
    """
    Evaluate model and save predictions along with true values.
    
    Returns:
        predictions: dict with keys 'velocity', 'stress', 'true_velocity', 'true_stress'
    """
    model.eval()
    
    all_pred_vel = []
    all_pred_stress = []
    all_pred_pos = []
    all_true_vel = []
    all_true_stress = []
    all_true_pos = []
    all_trajectory_ids = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for traj_idx, batch in enumerate(pbar):
            if num_trajectories is not None and traj_idx >= num_trajectories:
                break
                
            coors_seq = batch['coors'].to(device)  # (B, T, N, 3) or (T, N, 3)
            feats_seq = batch['feats'].to(device)  # (B, T, N, feat_dim) or (T, N, feat_dim)
            adj_mat = batch['adj_mat'].to(device)   # (B, N, N) or (N, N)
            target_vel = batch['target_vel'].to(device)  # (B, T, N, 3) or (T, N, 3)
            target_stress = batch['target_stress'].to(device)  # (B, T, N, 1) or (T, N, 1)
            world_pos = batch.get('world_pos')
            if world_pos is None:
                raise KeyError(
                    "Batch is missing 'world_pos'. Ensure the dataset returns world positions."
                )
            world_pos = world_pos.to(device)
            
            # Handle batch dimension
            if len(coors_seq.shape) == 3:  # (T, N, 3) - single trajectory
                coors_seq = coors_seq.unsqueeze(0)  # (1, T, N, 3)
                feats_seq = feats_seq.unsqueeze(0)  # (1, T, N, feat_dim)
                adj_mat = adj_mat.unsqueeze(0)      # (1, N, N)
                target_vel = target_vel.unsqueeze(0)  # (1, T, N, 3)
                target_stress = target_stress.unsqueeze(0)  # (1, T, N, 1)
                world_pos = world_pos.unsqueeze(0)
            
            B, T, N, _ = coors_seq.shape
            
            # Store predictions for this trajectory
            traj_pred_vel = []
            traj_pred_stress = []
            traj_pred_pos = []
            traj_true_vel = []
            traj_true_stress = []
            traj_true_pos = []
            
            # Process each timestep (starting from t=1)
            for t in range(1, T):
                feats_prev = feats_seq[:, t - 1]      # (B, N, feat_dim)
                coors_prev = coors_seq[:, t - 1]      # (B, N, 3)
                adj_mat_t = adj_mat            # (B, N, N) or (N, N)
                target_vel_t = target_vel[:, t]  # (B, N, 3)
                target_stress_t = target_stress[:, t]  # (B, N, 1)
                target_pos_t = world_pos[:, t]  # (B, N, 3)
                
                # Forward pass
                # Returns: (pred_vel, pred_stress, pred_coors) where:
                #   pred_vel follows: v_i^{l+1} = phi_v(h_i^l) * v_i^init + C * sum_j (x_i - x_j) * phi_x(m_ij)
                #   pred_coors = coors_prev + pred_vel
                pred_vel, pred_stress, pred_coors = model(feats_prev, coors_prev, adj_mat_t)
                
                # Convert to numpy and store
                traj_pred_vel.append(pred_vel.cpu().numpy())  # (B, N, 3)
                traj_pred_stress.append(pred_stress.cpu().numpy())  # (B, N, 1)
                traj_pred_pos.append(pred_coors.cpu().numpy())  # (B, N, 3)
                traj_true_vel.append(target_vel_t.cpu().numpy())  # (B, N, 3)
                traj_true_stress.append(target_stress_t.cpu().numpy())  # (B, N, 1)
                traj_true_pos.append(target_pos_t.cpu().numpy())  # (B, N, 3)
            
            # Stack timesteps: (T-1, B, N, ...)
            traj_pred_vel = np.stack(traj_pred_vel, axis=0)  # (T-1, B, N, 3)
            traj_pred_stress = np.stack(traj_pred_stress, axis=0)  # (T-1, B, N, 1)
            traj_pred_pos = np.stack(traj_pred_pos, axis=0)  # (T-1, B, N, 3)
            traj_true_vel = np.stack(traj_true_vel, axis=0)  # (T-1, B, N, 3)
            traj_true_stress = np.stack(traj_true_stress, axis=0)  # (T-1, B, N, 1)
            traj_true_pos = np.stack(traj_true_pos, axis=0)  # (T-1, B, N, 3)
            
            # Remove batch dimension if B=1: (T-1, N, ...)
            if B == 1:
                traj_pred_vel = traj_pred_vel[:, 0]  # (T-1, N, 3)
                traj_pred_stress = traj_pred_stress[:, 0]  # (T-1, N, 1)
                traj_pred_pos = traj_pred_pos[:, 0]  # (T-1, N, 3)
                traj_true_vel = traj_true_vel[:, 0]  # (T-1, N, 3)
                traj_true_stress = traj_true_stress[:, 0]  # (T-1, N, 1)
                traj_true_pos = traj_true_pos[:, 0]  # (T-1, N, 3)
            
            all_pred_vel.append(traj_pred_vel)
            all_pred_stress.append(traj_pred_stress)
            all_pred_pos.append(traj_pred_pos)
            all_true_vel.append(traj_true_vel)
            all_true_stress.append(traj_true_stress)
            all_true_pos.append(traj_true_pos)
            all_trajectory_ids.append(traj_idx)
            
            # Update progress
            pbar.set_postfix({'traj': traj_idx + 1})
    
    # Save predictions
    print(f"\nSaving predictions to {save_dir}...")
    
    # Save as numpy arrays
    # Save as lists of arrays using pickle (handles variable shapes across trajectories)
    # Use pickle directly to avoid numpy's array conversion issues
    with open(os.path.join(save_dir, 'predictions_velocity.npy'), 'wb') as f:
        pickle.dump(all_pred_vel, f)
    with open(os.path.join(save_dir, 'predictions_stress.npy'), 'wb') as f:
        pickle.dump(all_pred_stress, f)
    with open(os.path.join(save_dir, 'predictions_positions.npy'), 'wb') as f:
        pickle.dump(all_pred_pos, f)
    with open(os.path.join(save_dir, 'true_velocity.npy'), 'wb') as f:
        pickle.dump(all_true_vel, f)
    with open(os.path.join(save_dir, 'true_stress.npy'), 'wb') as f:
        pickle.dump(all_true_stress, f)
    with open(os.path.join(save_dir, 'true_positions.npy'), 'wb') as f:
        pickle.dump(all_true_pos, f)
    
    # Compute and save metrics
    metrics = compute_metrics(all_pred_vel, all_pred_stress, all_true_vel, all_true_stress)
    
    # Save metrics as JSON
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Predictions saved to {save_dir}")
    print(f"✓ Metrics saved to {os.path.join(save_dir, 'metrics.json')}")
    print(f"\nMetrics:")
    print(f"  Velocity MSE: {metrics['velocity_mse']:.6f}")
    print(f"  Velocity MAE: {metrics['velocity_mae']:.6f}")
    print(f"  Stress MSE: {metrics['stress_mse']:.6f}")
    print(f"  Stress MAE: {metrics['stress_mae']:.6f}")
    
    return {
        'predictions_velocity': all_pred_vel,
        'predictions_stress': all_pred_stress,
        'predictions_positions': all_pred_pos,
        'true_velocity': all_true_vel,
        'true_stress': all_true_stress,
        'true_positions': all_true_pos,
        'metrics': metrics
    }


def compute_metrics(pred_vel, pred_stress, true_vel, true_stress):
    """Compute evaluation metrics."""
    # Flatten all trajectories
    all_pred_vel = np.concatenate([v.flatten() for v in pred_vel])
    all_pred_stress = np.concatenate([s.flatten() for s in pred_stress])
    all_true_vel = np.concatenate([v.flatten() for v in true_vel])
    all_true_stress = np.concatenate([s.flatten() for s in true_stress])
    
    # Compute MSE and MAE
    vel_mse = np.mean((all_pred_vel - all_true_vel) ** 2)
    vel_mae = np.mean(np.abs(all_pred_vel - all_true_vel))
    stress_mse = np.mean((all_pred_stress - all_true_stress) ** 2)
    stress_mae = np.mean(np.abs(all_pred_stress - all_true_stress))
    
    return {
        'velocity_mse': float(vel_mse),
        'velocity_mae': float(vel_mae),
        'stress_mse': float(stress_mse),
        'stress_mae': float(stress_mae),
        'num_trajectories': len(pred_vel)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate EGNN on deforming plate dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data_dir', type=str, default='data/deforming_plate',
                       help='Directory containing dataset')
    parser.add_argument('--output_dir', type=str, default='results/egnn',
                       help='Directory to save predictions')
    parser.add_argument('--dataset_fraction', type=float, default=0.1,
                       help='Fraction of dataset to evaluate (0.0 to 1.0)')
    parser.add_argument('--num_trajectories', type=int, default=None,
                       help='Number of trajectories to evaluate (overrides dataset_fraction)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    train_tfrecord = os.path.join(args.data_dir, 'train.tfrecord')
    meta_path = os.path.join(args.data_dir, 'meta.json')
    
    if not os.path.exists(train_tfrecord):
        raise FileNotFoundError(f"Training data not found: {train_tfrecord}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    dataset = DeformingPlateDataset(
        train_tfrecord, meta_path, 
        dataset_fraction=args.dataset_fraction if args.num_trajectories is None else 1.0
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Determine feature dimensions
    sample = dataset[0]
    feat_dim = sample['feats'].shape[-1]
    print(f"Feature dimension: {feat_dim}")
    
    # Create model
    model = MeshEGNN(
        in_dim=feat_dim,
        hidden_dim=128,  # Should match training config
        out_dim=4,  # velocity (3D) + stress (1D)
        edge_dim=0,
        depth=4
    ).to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
    
    # Evaluate
    num_traj = args.num_trajectories if args.num_trajectories is not None else len(dataset)
    results = evaluate_model(model, dataloader, device, args.output_dir, num_trajectories=num_traj)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Evaluated {len(results['predictions_velocity'])} trajectories")
    print(f"  Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

