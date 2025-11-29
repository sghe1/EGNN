import torch
import torch.nn as nn
import torch.nn.functional as F
import sidechainnet as scn
from myEGNN.EGNN import MeshEGNN
from Project2.data_loader_egnn import data_loader_egnn

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Loading the data

tfrecord_path = "data/deforming_plate/train.tfrecord"
meta_path = "data/deforming_plate/meta.json"

coors, feats, edge_index = data_loader_egnn(
    tfrecord_path, meta_path, traj_index=0
)
    
BATCH_SIZE = 1
data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

# Model configuration
ACCEL_DIM = 3  
STRESS_DIM = 1  
OUT_DIM = ACCEL_DIM + STRESS_DIM 

# Creating the model
model = MeshEGNN(in_dim=1, hidden_dim=128, out_dim=OUT_DIM, edge_dim=1)
model = model.to(device)
model.train()  

# Training the model
num_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss tracking
train_losses = []

# Expected trajectory structure:
# - 1000 trajectories in training set
# - Each trajectory has 400 timestamps
# - Each timestamp is a graph with nodes, coordinates, features, etc.

NUM_TIMESTAMPS = 400  

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_samples = 0  
    
    for traj_idx, trajectory in enumerate(data['train']):
        try:
            # Each trajectory contains data for all 400 timestamps
            # Expected shapes:
            # - feats: [400, num_nodes, in_dim] or [num_nodes, in_dim] (if static)
            # - coors: [400, num_nodes, 3]
            # - adj_mat: [num_nodes, num_nodes] (usually static across timestamps)
            # - acceleration: [400, num_nodes, 3]
            # - stress: [400, num_nodes, 1]
            
            # Load trajectory data
            if hasattr(trajectory, 'feats'):
                traj_feats = trajectory.feats.to(device) if torch.is_tensor(trajectory.feats) else torch.tensor(trajectory.feats, device=device)
            else:
                raise AttributeError("Trajectory does not have 'feats' attribute.")
            
            if hasattr(trajectory, 'coors'):
                traj_coors = trajectory.coors.to(device) if torch.is_tensor(trajectory.coors) else torch.tensor(trajectory.coors, device=device)
            else:
                raise AttributeError("Trajectory does not have 'coors' attribute.")
            
            if hasattr(trajectory, 'adj_mat'):
                traj_adj_mat = trajectory.adj_mat.to(device) if torch.is_tensor(trajectory.adj_mat) else torch.tensor(trajectory.adj_mat, device=device)
            else:
                raise AttributeError("Trajectory does not have 'adj_mat' attribute.")
            
            traj_edges = None
            if hasattr(trajectory, 'edges'):
                traj_edges = trajectory.edges.to(device) if torch.is_tensor(trajectory.edges) else torch.tensor(trajectory.edges, device=device)
            
            # Load targets
            traj_accel = getattr(trajectory, "acceleration")
            if torch.is_tensor(traj_accel):
                traj_accel = traj_accel.to(device)
            else:
                traj_accel = torch.tensor(traj_accel, device=device, dtype=torch.float32)
            
            traj_stress = getattr(trajectory, "stress")
            if torch.is_tensor(traj_stress):
                traj_stress = traj_stress.to(device)
            else:
                traj_stress = torch.tensor(traj_stress, device=device, dtype=torch.float32)
            
            # Determine number of timestamps in this trajectory
            # Check if data has time dimension
            if len(traj_coors.shape) == 3:
                # Shape: [num_timestamps, num_nodes, 3]
                num_timestamps = traj_coors.shape[0]
            elif len(traj_coors.shape) == 2:
                # Shape: [num_nodes, 3] - single timestamp (unlikely for trajectory)
                num_timestamps = 1
            else:
                raise ValueError(f"Unexpected coors shape: {traj_coors.shape}")
            
            # Process each timestamp in the trajectory
            for t in range(num_timestamps):
                if len(traj_feats.shape) == 3:
                    feats = traj_feats[t]  # [num_nodes, in_dim]
                else:
                    feats = traj_feats  # Static features [num_nodes, in_dim]
                
                coors = traj_coors[t]  # [num_nodes, 3]
                
                # Adjacency matrix is usually static (same for all timestamps)
                if len(traj_adj_mat.shape) == 3:
                    adj_mat = traj_adj_mat[t]  # [num_nodes, num_nodes]
                else:
                    adj_mat = traj_adj_mat  # [num_nodes, num_nodes]
                
                edges = None
                if traj_edges is not None:
                    if len(traj_edges.shape) == 4:
                        edges = traj_edges[t]  # [num_nodes, num_nodes, edge_dim]
                    elif len(traj_edges.shape) == 3:
                        edges = traj_edges[t]  # [num_nodes, num_nodes] or [num_edges, edge_dim]
                    else:
                        edges = traj_edges  # Static edges
                
                # Extract targets at timestamp t
                if len(traj_accel.shape) == 3:
                    target_accel = traj_accel[t]  # [num_nodes, 3]
                else:
                    target_accel = traj_accel  # [num_nodes, 3]
                
                if len(traj_stress.shape) == 3:
                    target_stress = traj_stress[t]  # [num_nodes, 1] or [num_nodes, STRESS_DIM]
                elif len(traj_stress.shape) == 2:
                    target_stress = traj_stress[t]  # [num_nodes] -> will be reshaped
                else:
                    target_stress = traj_stress  # [num_nodes, STRESS_DIM]
                
                # Add batch dimension: [1, num_nodes, ...]
                feats = feats.unsqueeze(0)  # [1, num_nodes, in_dim]
                coors = coors.unsqueeze(0)  # [1, num_nodes, 3]
                adj_mat = adj_mat.unsqueeze(0)  # [1, num_nodes, num_nodes]
                target_accel = target_accel.unsqueeze(0)  # [1, num_nodes, 3]
                
                # Handle edges batch dimension if needed
                if edges is not None:
                    if len(edges.shape) == 2:
                        edges = edges.unsqueeze(0)  # [1, num_nodes, num_nodes] or [1, num_edges, edge_dim]
                    elif len(edges.shape) == 3:
                        edges = edges.unsqueeze(0)  # [1, num_nodes, num_nodes, edge_dim]
                
                # Handle stress shape
                if len(target_stress.shape) == 1:
                    target_stress = target_stress.unsqueeze(0).unsqueeze(-1)  # [1, num_nodes, 1]
                elif len(target_stress.shape) == 2:
                    target_stress = target_stress.unsqueeze(0)  # [1, num_nodes, STRESS_DIM]
                
                # Zero gradients before forward pass
                optimizer.zero_grad()
                
                # Forward pass
                pred, coors_out = model(feats, coors, adj_mat, edges)
                
                # Split predictions into acceleration and stress
                pred_accel = pred[:, :, :ACCEL_DIM]  # [1, num_nodes, 3]
                pred_stress = pred[:, :, ACCEL_DIM:]  # [1, num_nodes, STRESS_DIM]
                
                # Compute loss
                loss_accel = F.mse_loss(pred_accel, target_accel)
                loss_stress = F.mse_loss(pred_stress, target_stress)
                loss = loss_accel + loss_stress
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                epoch_loss += loss.item()
                num_samples += 1
                
                # Print progress
                if num_samples % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Traj {traj_idx+1}, Timestamp {t+1}/{num_timestamps}, "
                          f"Loss: {loss.item():.6f} (Accel: {loss_accel.item():.6f}, Stress: {loss_stress.item():.6f})")
        
        except AttributeError as e:
            print(f"Error in trajectory {traj_idx}: {e}")
            print("Please check the trajectory data structure.")
            break
        except Exception as e:
            print(f"Error processing trajectory {traj_idx}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if num_samples > 0:
        avg_loss = epoch_loss / num_samples
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Processed {num_samples} samples. Average Loss: {avg_loss:.6f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} - No samples processed")

print("\nTraining completed!")
print(f"Final average loss: {train_losses[-1] if train_losses else 'N/A'}")