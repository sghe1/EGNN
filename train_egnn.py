import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml
import os
import time
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

# Reuse your existing data infrastructure
from defplate_dataset import DefPlateDataset, collate_unet
from model_egnn import EGNN_DefPlate
from plots import make_final_plots  # Ensure plots.py is in the same folder

# Constants
BOUNDARY_NODE = 3
SPHERE_NODE = 1
NORMAL_NODE = 0
DIM_OUT_VEL = 3
DIM_OUT_STRESS = 1

def compute_loss(adj_A_list, feat_tp1_mat_list, node_types_list, preds_list, velocity_idxs, stress_idxs):
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    num_graphs = len(adj_A_list)
    
    for pred, target, nodetype in zip(preds_list, feat_tp1_mat_list, node_types_list):
        # Create masks
        vel_mask = (nodetype == NORMAL_NODE)
        stress_mask = (nodetype == NORMAL_NODE) | (nodetype == BOUNDARY_NODE)
        
        target_vel = target[:, velocity_idxs]
        target_stress = target[:, stress_idxs]
        
        pred_vel = pred[:, :3]
        pred_stress = pred[:, 3:4]
        
        loss_graph = 0.0
        
        if vel_mask.any():
            vel_loss = F.mse_loss(pred_vel[vel_mask], target_vel[vel_mask])
            loss_graph += vel_loss
            total_vel_loss += vel_loss
            
        if stress_mask.any():
            stress_loss = F.huber_loss(pred_stress[stress_mask], target_stress[stress_mask])
            loss_graph += stress_loss
            total_stress_loss += stress_loss
            
        total_loss += loss_graph

    return total_loss / num_graphs, total_vel_loss / num_graphs, total_stress_loss / num_graphs

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

# Helper wrapper for config to match Model expectation
class ArgsWrapper:
    pass

def train_egnn(device, num_workers, pin_memory):
    config_path = os.path.join(os.path.dirname(__file__), "config_egnn.yaml")
    config = load_config(config_path)
    train_cfg = config['training']
    model_cfg = config['model']
    
    # --- 1. SETUP DATA ---
    preprocessed_data_path = train_cfg['datapath']
    print(f"Loading data from {preprocessed_data_path}")
    list_of_trajs = torch.load(preprocessed_data_path)
    
    # --- DYNAMIC DIMENSION DETECTION ---
    sample_feat_dim = list_of_trajs[0]["X_seq_norm"].shape[2]
    # Input Dim = stored_dim + 3 (Kinematic Velocity injected in dataset)
    dim_in = sample_feat_dim + 3
    print(f"Detected Feature Dim: {sample_feat_dim}. Model Input Dim: {dim_in}")

    if dim_in == 15: # MeshPos included
        world_pos_idxs = slice(3, 6); velocity_idxs = slice(8, 11); stress_idxs = slice(11, 12)
    else: # No MeshPos
        world_pos_idxs = slice(0, 3); velocity_idxs = slice(5, 8); stress_idxs = slice(8, 9)

    # Initialize Dataset
    dataset = DefPlateDataset(
        list_of_trajs, 
        add_world_edges=train_cfg['add_world_edges'], 
        k_neighb=train_cfg['k_neighb'], 
        radius=train_cfg['radius_world_edge'], 
        world_pos_idxs=world_pos_idxs, 
        velocity_idxs=velocity_idxs
    )
    
    # --- 2. SPLIT / OVERFIT LOGIC ---
    if train_cfg.get('mode') == 'overfit':
        overfit_traj_id = train_cfg.get('overfit_traj_id', 0)
        overfit_time_idx = train_cfg.get('overfit_time_idx', [0, 1, 2])
        
        print(f"Mode: OVERFIT. Trajectory: {overfit_traj_id}, Time Steps: {overfit_time_idx}")
        indices = [i for i, s in enumerate(dataset.samples) 
                   if s['traj_id'] == overfit_traj_id and s['time_idx'] in overfit_time_idx]
        
        if not indices: raise ValueError("No matching samples found for overfit configuration!")
        train_set = Subset(dataset, indices)
        test_set = train_set # Validate on same data
        print(f"Overfitting on {len(train_set)} samples.")
    else:
        total = len(dataset)
        split = int(0.8 * total)
        train_set = Subset(dataset, range(split))
        test_set = Subset(dataset, range(split, total))
        print(f"Standard Mode. Train: {len(train_set)}, Val: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=train_cfg['batch_size'], shuffle=train_cfg['shuffle'], collate_fn=collate_unet, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collate_unet, num_workers=num_workers)

    # --- 3. SETUP MODEL (EGNN) ---
    model = EGNN_DefPlate(dim_in, DIM_OUT_VEL, DIM_OUT_STRESS, model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_lr_scheduler'])

    # History Tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_vel': [], 'train_str': [],
        'val_vel': [], 'val_str': []
    }
    grad_norms = []

    # --- 4. TRAINING LOOP ---
    print(f"Starting EGNN Training on {device}...")
    
    for epoch in range(train_cfg['epochs']):
        model.train()
        ep_loss = 0; ep_vel = 0; ep_str = 0
        
        for batch in train_loader:
            adj_list = [x.to(device) for x in batch[0]]
            xt_list = [x.to(device) for x in batch[1]]
            xtp1_list = [x.to(device) for x in batch[2]]
            nt_list = [x.to(device) for x in batch[6]]
            
            optimizer.zero_grad()
            preds = model(adj_list, xt_list, xtp1_list, nt_list)
            loss, v_loss, s_loss = compute_loss(adj_list, xtp1_list, nt_list, preds, velocity_idxs, stress_idxs)
            loss.backward()
            
            # Gradient Norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None: total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)
            
            optimizer.step()
            
            ep_loss += loss.item()
            ep_vel += v_loss.item()
            ep_str += s_loss.item()

        # Validation
        model.eval()
        val_loss = 0; val_vel = 0; val_str = 0
        with torch.no_grad():
            for batch in test_loader:
                adj_list = [x.to(device) for x in batch[0]]
                xt_list = [x.to(device) for x in batch[1]]
                xtp1_list = [x.to(device) for x in batch[2]]
                nt_list = [x.to(device) for x in batch[6]]
                
                preds = model(adj_list, xt_list, xtp1_list, nt_list)
                loss, v_l, s_l = compute_loss(adj_list, xtp1_list, nt_list, preds, velocity_idxs, stress_idxs)
                
                val_loss += loss.item()
                val_vel += v_l.item()
                val_str += s_l.item()
        
        scheduler.step()
        
        # Averages
        avg_train = ep_loss / len(train_loader)
        avg_train_v = ep_vel / len(train_loader)
        avg_train_s = ep_str / len(train_loader)
        
        avg_val = val_loss / len(test_loader) if len(test_loader) > 0 else 0
        avg_val_v = val_vel / len(test_loader) if len(test_loader) > 0 else 0
        avg_val_s = val_str / len(test_loader) if len(test_loader) > 0 else 0
        
        # Store
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['train_vel'].append(avg_train_v)
        history['train_str'].append(avg_train_s)
        history['val_vel'].append(avg_val_v)
        history['val_str'].append(avg_val_s)
        
        # --- PRINT EVERY EPOCH ---
        print(f"Epoch {epoch:03d} | Train: {avg_train:.6f} (V:{avg_train_v:.6f} S:{avg_train_s:.6f}) | Val: {avg_val:.6f} (V:{avg_val_v:.6f} S:{avg_val_s:.6f})")
        
    # --- 5. FINAL PLOTTING & SAVE ---
    os.makedirs(train_cfg['model_path'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(train_cfg['model_path'], "model_egnn.pt"))
    print("Saved model.")

    # Collect Final Predictions for Plots
    print("Collecting predictions for plots...")
    all_preds_vel = []
    all_preds_str = []
    all_targs_vel = []
    all_targs_str = []
    
    # We collect normalized predictions (as plots.py expects normalized usually, 
    # but check your plots.py if it denormalizes. Standard plots.py usually takes lists of arrays)
    with torch.no_grad():
        for batch in test_loader:
            adj_list = [x.to(device) for x in batch[0]]
            xt_list = [x.to(device) for x in batch[1]]
            xtp1_list = [x.to(device) for x in batch[2]]
            nt_list = [x.to(device) for x in batch[6]]
            
            preds = model(adj_list, xt_list, xtp1_list, nt_list)
            
            # Mask and Collect
            for p, t, nt in zip(preds, xtp1_list, nt_list):
                # Only Plot Normal Nodes for Velocity
                v_mask = (nt == NORMAL_NODE)
                if v_mask.any():
                    all_preds_vel.append(p[v_mask, :3].cpu().numpy())
                    all_targs_vel.append(t[v_mask, velocity_idxs].cpu().numpy())
                
                # Only Plot Normal/Boundary for Stress
                s_mask = (nt == NORMAL_NODE) | (nt == BOUNDARY_NODE)
                if s_mask.any():
                    all_preds_str.append(p[s_mask, 3:4].cpu().numpy())
                    all_targs_str.append(t[s_mask, stress_idxs].cpu().numpy())

    # Format for plots.py (List of arrays -> Arrays or List of Arrays)
    # plots.py expects [VelX, VelY, VelZ, Stress] structure usually
    # We will pass raw lists which plots.py handles via 'prepare_lists'
    
    # Concatenate for simpler passing if plots.py supports it, or keep as lists
    # Creating flat arrays for simple scatter logic
    flat_pred_vel = np.concatenate(all_preds_vel, axis=0) if all_preds_vel else np.array([])
    flat_targ_vel = np.concatenate(all_targs_vel, axis=0) if all_targs_vel else np.array([])
    flat_pred_str = np.concatenate(all_preds_str, axis=0) if all_preds_str else np.array([])
    flat_targ_str = np.concatenate(all_targs_str, axis=0) if all_targs_str else np.array([])

    # Prepare final lists: [VelX, VelY, VelZ, Stress]
    # We split columns
    final_preds = []
    final_targs = []
    
    if len(flat_pred_vel) > 0:
        final_preds = [flat_pred_vel[:,0], flat_pred_vel[:,1], flat_pred_vel[:,2], flat_pred_str[:,0]]
        final_targs = [flat_targ_vel[:,0], flat_targ_vel[:,1], flat_targ_vel[:,2], flat_targ_str[:,0]]
    
    # Generate Plots
    plots_dir = os.path.join(train_cfg['model_path'], "plots_egnn")
    
    # We pass None for norm/denorm distinction if we don't have means/stds loaded here easily
    # Or we just pass the normalized values to both arguments to see at least the scatter
    make_final_plots(
        save_dir=plots_dir,
        train_losses=history['train_loss'],
        val_losses=history['val_loss'],
        metric_name="Loss",
        train_metrics=None, val_metrics=None,
        grad_norms=grad_norms,
        model=model,
        activations={}, # Skip activations
        predictions=final_preds, targets=final_targs,
        predictions_norm=final_preds, targets_norm=final_targs, # Passing same for now
        train_vel_losses=history['train_vel'],
        train_stress_losses=history['train_str'],
        test_vel_losses=history['val_vel'],
        test_stress_losses=history['val_str'],
        velocity_idxs=velocity_idxs, stress_idxs=stress_idxs
    )
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_egnn(device, num_workers=0, pin_memory=False)