#!/usr/bin/env python3
"""
Quick test script to train EGNN on 3 trajectories and compare predictions with ground truth.
This verifies the model is working correctly before running on the full dataset.
"""

import os
import sys
import json
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

from data_loader_egnn import load_raw_trajectory_from_tfrecord, trajectory_to_egnn_inputs
from train_egnn import DeformingPlateDataset, compute_target_velocity, MeshEGNN, train_epoch

def evaluate_and_save(model, dataloader, device, output_dir, num_trajectories=3):
    """Evaluate model and save predictions with ground truth for comparison."""
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    with torch.no_grad():
        print(f"\nEvaluating on {num_trajectories} trajectories...")
        for traj_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if traj_idx >= num_trajectories:
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
            traj_errors_vel = []
            traj_errors_stress = []
            traj_pos_errors = []
            
            # Process each timestep
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
                
                # Compute errors
                error_vel = torch.abs(pred_vel - target_vel_t)
                error_stress = torch.abs(pred_stress - target_stress_t)
                error_pos = torch.abs(pred_coors - target_pos_t)
                
                # Convert to numpy
                traj_pred_vel.append(pred_vel.cpu().numpy())
                traj_pred_stress.append(pred_stress.cpu().numpy())
                traj_pred_pos.append(pred_coors.cpu().numpy())
                traj_true_vel.append(target_vel_t.cpu().numpy())
                traj_true_stress.append(target_stress_t.cpu().numpy())
                traj_true_pos.append(target_pos_t.cpu().numpy())
                traj_errors_vel.append(error_vel.cpu().numpy())
                traj_errors_stress.append(error_stress.cpu().numpy())
                traj_pos_errors.append(error_pos.cpu().numpy())
            
            # Stack timesteps
            traj_pred_vel = np.stack(traj_pred_vel, axis=0)  # (T-1, B, N, 3)
            traj_pred_stress = np.stack(traj_pred_stress, axis=0)  # (T-1, B, N, 1)
            traj_pred_pos = np.stack(traj_pred_pos, axis=0)
            traj_true_vel = np.stack(traj_true_vel, axis=0)
            traj_true_stress = np.stack(traj_true_stress, axis=0)
            traj_true_pos = np.stack(traj_true_pos, axis=0)
            traj_errors_vel = np.stack(traj_errors_vel, axis=0)
            traj_errors_stress = np.stack(traj_errors_stress, axis=0)
            traj_pos_errors = np.stack(traj_pos_errors, axis=0)
            
            # Remove batch dimension if B=1
            if B == 1:
                traj_pred_vel = traj_pred_vel[:, 0]  # (T-1, N, 3)
                traj_pred_stress = traj_pred_stress[:, 0]  # (T-1, N, 1)
                traj_pred_pos = traj_pred_pos[:, 0]
                traj_true_vel = traj_true_vel[:, 0]
                traj_true_stress = traj_true_stress[:, 0]
                traj_true_pos = traj_true_pos[:, 0]
                traj_errors_vel = traj_errors_vel[:, 0]
                traj_errors_stress = traj_errors_stress[:, 0]
                traj_pos_errors = traj_pos_errors[:, 0]
            
            # Compute metrics for this trajectory
            vel_mse = np.mean((traj_pred_vel - traj_true_vel) ** 2)
            vel_mae = np.mean(np.abs(traj_pred_vel - traj_true_vel))
            stress_mse = np.mean((traj_pred_stress - traj_true_stress) ** 2)
            stress_mae = np.mean(np.abs(traj_pred_stress - traj_true_stress))
            pos_mse = np.mean((traj_pred_pos - traj_true_pos) ** 2)
            pos_mae = np.mean(np.abs(traj_pred_pos - traj_true_pos))
            
            # Save individual trajectory
            traj_dir = os.path.join(output_dir, f'trajectory_{traj_idx}')
            os.makedirs(traj_dir, exist_ok=True)
            
            np.save(os.path.join(traj_dir, 'predictions_velocity.npy'), traj_pred_vel)
            np.save(os.path.join(traj_dir, 'predictions_stress.npy'), traj_pred_stress)
            np.save(os.path.join(traj_dir, 'predictions_positions.npy'), traj_pred_pos)
            np.save(os.path.join(traj_dir, 'true_velocity.npy'), traj_true_vel)
            np.save(os.path.join(traj_dir, 'true_stress.npy'), traj_true_stress)
            np.save(os.path.join(traj_dir, 'true_positions.npy'), traj_true_pos)
            np.save(os.path.join(traj_dir, 'errors_velocity.npy'), traj_errors_vel)
            np.save(os.path.join(traj_dir, 'errors_stress.npy'), traj_errors_stress)
            np.save(os.path.join(traj_dir, 'errors_positions.npy'), traj_pos_errors)
            
            traj_metrics = {
                'trajectory_id': traj_idx,
                'velocity_mse': float(vel_mse),
                'velocity_mae': float(vel_mae),
                'stress_mse': float(stress_mse),
                'stress_mae': float(stress_mae),
                'position_mse': float(pos_mse),
                'position_mae': float(pos_mae),
                'shape': {
                    'timesteps': int(traj_pred_vel.shape[0]),
                    'nodes': int(traj_pred_vel.shape[1])
                }
            }
            all_results.append(traj_metrics)
            
            print(f"\nTrajectory {traj_idx}:")
            print(f"  Velocity - MSE: {vel_mse:.6f}, MAE: {vel_mae:.6f}")
            print(f"  Stress   - MSE: {stress_mse:.6f}, MAE: {stress_mae:.6f}")
            print(f"  Position - MSE: {pos_mse:.6f}, MAE: {pos_mae:.6f}")
            print(f"  Saved to: {traj_dir}")
    
    # Save summary
    summary = {
        'num_trajectories': len(all_results),
        'trajectories': all_results,
        'overall_velocity_mse': float(np.mean([r['velocity_mse'] for r in all_results])),
        'overall_velocity_mae': float(np.mean([r['velocity_mae'] for r in all_results])),
        'overall_stress_mse': float(np.mean([r['stress_mse'] for r in all_results])),
        'overall_stress_mae': float(np.mean([r['stress_mae'] for r in all_results])),
        'overall_position_mse': float(np.mean([r['position_mse'] for r in all_results])),
        'overall_position_mae': float(np.mean([r['position_mae'] for r in all_results]))
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Overall Velocity - MSE: {summary['overall_velocity_mse']:.6f}, MAE: {summary['overall_velocity_mae']:.6f}")
    print(f"Overall Stress   - MSE: {summary['overall_stress_mse']:.6f}, MAE: {summary['overall_stress_mae']:.6f}")
    print(f"Overall Position - MSE: {summary['overall_position_mse']:.6f}, MAE: {summary['overall_position_mae']:.6f}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"Summary saved to: {os.path.join(output_dir, 'summary.json')}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Quick test: Train and evaluate EGNN on 3 trajectories')
    parser.add_argument('--data_dir', type=str, default='data/deforming_plate',
                       help='Directory containing dataset')
    parser.add_argument('--output_dir', type=str, default='test_results/egnn_quick_test',
                       help='Directory to save test results')
    parser.add_argument('--num_trajectories', type=int, default=3,
                       help='Number of trajectories to test on')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs (quick test)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for EGNN')
    parser.add_argument('--depth', type=int, default=4,
                       help='Depth of EGNN layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--velocity_loss_weight', type=float, default=1.0,
                       help='Weight for velocity loss relative to stress loss (default: 1.0)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Testing on {args.num_trajectories} trajectories")
    
    # Load dataset (only the trajectories we need)
    train_tfrecord = os.path.join(args.data_dir, 'train.tfrecord')
    meta_path = os.path.join(args.data_dir, 'meta.json')
    
    if not os.path.exists(train_tfrecord):
        raise FileNotFoundError(f"Training data not found: {train_tfrecord}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    # Create dataset with only the number of trajectories we need
    dataset = DeformingPlateDataset(
        train_tfrecord, meta_path,
        num_trajectories=args.num_trajectories
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    eval_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Get feature dimension
    sample = dataset[0]
    feat_dim = sample['feats'].shape[-1]
    print(f"Feature dimension: {feat_dim}")
    
    # Create model
    model = MeshEGNN(
        in_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        edge_dim=0,
        depth=args.depth
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"\nTraining for {args.num_epochs} epochs...")
    print(f"Velocity loss weight: {args.velocity_loss_weight}")
    print("-" * 70)
    
    for epoch in range(args.num_epochs):
        avg_loss, avg_loss_vel, avg_loss_stress = train_epoch(
            model, train_dataloader, optimizer, device, epoch, args.velocity_loss_weight
        )
        print(f"Epoch {epoch+1}/{args.num_epochs} - Total Loss: {avg_loss:.6f} "
              f"(Vel: {avg_loss_vel:.6f}, Stress: {avg_loss_stress:.6f})")
    
    # Evaluate and save predictions
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")
    summary = evaluate_and_save(model, eval_dataloader, device, args.output_dir, args.num_trajectories)
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print("\nYou can now inspect the predictions and compare with ground truth.")
    print("Each trajectory has its own directory with:")
    print("  - predictions_velocity.npy")
    print("  - predictions_stress.npy")
    print("  - predictions_positions.npy")
    print("  - true_velocity.npy")
    print("  - true_stress.npy")
    print("  - true_positions.npy")
    print("  - errors_velocity.npy")
    print("  - errors_stress.npy")
    print("  - errors_positions.npy")


if __name__ == '__main__':
    main()
