import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
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

def compute_loss(adj_A_list, feat_tp1_mat_list, node_types_list, preds_list, velocity_idxs, stress_idxs, 
                 std_vel=None, std_stress=None):
    """
    Optimized vectorized loss computation with normalization.
    
    IMPORTANT: All quantities are NORMALIZED.
    - feat_tp1_mat_list contains X_seq_norm[t+1] which is normalized: (X - mean) / std
    - preds_list contains normalized predictions from the model
    - Loss is computed on normalized quantities and then normalized by target std^2 for scale-invariance
    
    Args:
        std_vel: Standard deviation of velocity targets (for loss normalization). If None, uses 1.0
        std_stress: Standard deviation of stress targets (for loss normalization). If None, uses 1.0
    """
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    num_graphs = len(adj_A_list)
    
    # Normalization factors: divide by std^2 to make loss scale-invariant
    # Since data is already normalized (std ≈ 1), this is typically 1.0, but we keep it explicit
    norm_vel = (std_vel ** 2) if std_vel is not None else 1.0
    norm_stress = (std_stress ** 2) if std_stress is not None else 1.0
    
    # Process all graphs in batch for better GPU utilization
    for pred, target, nodetype in zip(preds_list, feat_tp1_mat_list, node_types_list):
        # Create masks once
        vel_mask = (nodetype == NORMAL_NODE)
        stress_mask = (nodetype == NORMAL_NODE) | (nodetype == BOUNDARY_NODE)
        
        # Extract targets and predictions (ALL NORMALIZED)
        # target is X_seq_norm[t+1] which contains normalized features
        target_vel = target[:, velocity_idxs]  # Normalized velocity
        target_stress = target[:, stress_idxs]  # Normalized stress
        pred_vel = pred[:, :3]  # Normalized velocity prediction
        pred_stress = pred[:, 3:4]  # Normalized stress prediction
        
        loss_graph = 0.0
        
        # Velocity loss (only on normal nodes) - normalized by std^2
        if vel_mask.any():
            vel_loss = F.mse_loss(pred_vel[vel_mask], target_vel[vel_mask], reduction='mean')
            vel_loss_norm = vel_loss / norm_vel  # Normalize by target variance
            loss_graph += vel_loss_norm
            total_vel_loss += vel_loss_norm.item() if isinstance(vel_loss_norm, torch.Tensor) else vel_loss_norm
            
        # Stress loss (on normal + boundary nodes) - normalized by std^2
        if stress_mask.any():
            stress_loss = F.huber_loss(pred_stress[stress_mask], target_stress[stress_mask], reduction='mean', delta=1.0)
            stress_loss_norm = stress_loss / norm_stress  # Normalize by target variance
            loss_graph += stress_loss_norm
            total_stress_loss += stress_loss_norm.item() if isinstance(stress_loss_norm, torch.Tensor) else stress_loss_norm
            
        total_loss += loss_graph

    return total_loss / num_graphs, total_vel_loss / num_graphs, total_stress_loss / num_graphs

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

# Helper wrapper for config to match Model expectation
class ArgsWrapper:
    pass

def train_egnn(device, num_workers, pin_memory):
    """
    Train EGNN model on normalized data.
    
    NORMALIZATION FLOW:
    1. Data preprocessing (preprocess_egnn.py) loads raw data and normalizes it:
       - X_seq_norm = (X - mean) / std (computed in data_loader.py)
       - All features (positions, velocities, stress) are normalized
    2. Dataset (DefPlateDataset) returns normalized features:
       - X_t: normalized input features at time t
       - X_tp1: normalized target features at time t+1
    3. Model processes normalized inputs and outputs normalized predictions
    4. Loss is computed on normalized quantities (normalized preds vs normalized targets)
    
    Denormalization is only done for visualization/plotting, not during training.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config_egnn.yaml")
    config = load_config(config_path)
    train_cfg = config['training']
    model_cfg = config['model']
    
    # --- 1. SETUP DATA ---
    # Resolve path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_path = train_cfg['datapath']
    if not os.path.isabs(preprocessed_data_path):
        preprocessed_data_path = os.path.normpath(os.path.join(script_dir, preprocessed_data_path))
    
    print(f"Loading data from {preprocessed_data_path}")
    
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(
            f"Preprocessed data file not found at: {preprocessed_data_path}\n"
            f"Please run preprocessing first using: python preprocess_egnn.py\n"
            f"Or check that the path in config_egnn.yaml is correct."
        )
    
    list_of_trajs = torch.load(preprocessed_data_path)
    
    # --- VERIFY NORMALIZATION ---
    # X_seq_norm contains NORMALIZED features: (X - mean) / std
    # All quantities (positions, velocities, stress) are normalized
    # Mean and std are stored per trajectory for denormalization later
    sample_traj = list_of_trajs[0]
    std_vel = None
    std_stress = None
    
    if "mean" in sample_traj and "std" in sample_traj:
        print(f"✓ Data is normalized. Mean shape: {sample_traj['mean'].shape}, Std shape: {sample_traj['std'].shape}")
        # Verify normalization: mean should be close to 0, std should be close to 1
        sample_X = sample_traj["X_seq_norm"]
        sample_mean = sample_X.mean().item()
        sample_std = sample_X.std().item()
        print(f"  Sample X_seq_norm stats: mean≈{sample_mean:.4f}, std≈{sample_std:.4f} (should be ≈0 and ≈1)")
    
    # --- DYNAMIC DIMENSION DETECTION ---
    sample_feat_dim = list_of_trajs[0]["X_seq_norm"].shape[2]
    # Input Dim = stored_dim + 3 (Kinematic Velocity injected in dataset)
    dim_in = sample_feat_dim + 3
    print(f"Detected Feature Dim: {sample_feat_dim}. Model Input Dim: {dim_in}")

    if dim_in == 15: # MeshPos included
        world_pos_idxs = slice(3, 6); velocity_idxs = slice(8, 11); stress_idxs = slice(11, 12)
    else: # No MeshPos
        world_pos_idxs = slice(0, 3); velocity_idxs = slice(5, 8); stress_idxs = slice(8, 9)
    
    # Extract std for velocity and stress for loss normalization (after dim_in is defined)
    if "std" in sample_traj:
        std_tensor = sample_traj["std"].squeeze()  # [F]
        if dim_in == 15:  # MeshPos included
            std_vel = std_tensor[8:11].mean().item()  # Average std across velocity components
            std_stress = std_tensor[11].item()  # Stress std
        else:  # No MeshPos
            std_vel = std_tensor[5:8].mean().item()  # Average std across velocity components
            std_stress = std_tensor[8].item()  # Stress std
        
        print(f"  Target std for loss normalization: vel={std_vel:.4f}, stress={std_stress:.4f}")

    # Check if A_seq is precomputed (world edges computed during preprocessing)
    use_precomputed_A = "A_seq" in list_of_trajs[0] if len(list_of_trajs) > 0 else False
    
    if use_precomputed_A:
        print("✓ Using precomputed adjacency matrices (A_seq) - world edges computed during preprocessing")
        print("  Training will be faster as adjacency computation is skipped")
    else:
        print("⚠ A_seq not found - will compute world edges on the fly (slower)")
        print("  Consider re-running preprocessing to compute A_seq")

    # Initialize Dataset
    # If A_seq is precomputed, world edge parameters are optional (only used as fallback)
    dataset = DefPlateDataset(
        list_of_trajs, 
        add_world_edges=train_cfg.get('add_world_edges', 'None') if not use_precomputed_A else None,
        k_neighb=train_cfg.get('k_neighb', 5) if not use_precomputed_A else None,
        radius=train_cfg.get('radius_world_edge', 0.03) if not use_precomputed_A else None,
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

    # Optimize data loading: use multiprocessing if available, pin memory for GPU
    train_loader = DataLoader(
        train_set, 
        batch_size=train_cfg['batch_size'], 
        shuffle=train_cfg['shuffle'], 
        collate_fn=collate_unet, 
        num_workers=num_workers if num_workers > 0 else (4 if device.type == 'cuda' else 0),
        pin_memory=pin_memory and device.type == 'cuda',
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=train_cfg['batch_size'], 
        shuffle=False, 
        collate_fn=collate_unet, 
        num_workers=num_workers if num_workers > 0 else (4 if device.type == 'cuda' else 0),
        pin_memory=pin_memory and device.type == 'cuda',
        persistent_workers=num_workers > 0
    )

    # --- 3. SETUP MODEL (EGNN) ---
    model = EGNN_DefPlate(dim_in, DIM_OUT_VEL, DIM_OUT_STRESS, model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_lr_scheduler'])
    
    # Mixed precision training for GPU acceleration
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    # History Tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_vel': [], 'train_str': [],
        'val_vel': [], 'val_str': []
    }
    grad_norms = []
    
    # Gradient norm computation frequency (every N steps to reduce overhead)
    grad_norm_freq = 10

    # --- 4. TRAINING LOOP ---
    print(f"Starting EGNN Training on {device}...")
    if use_amp:
        print("Using mixed precision training (AMP)")
    
    # Initialize prediction collection lists (for final epoch)
    all_preds_vel = []
    all_preds_str = []
    all_targs_vel = []
    all_targs_str = []
    
    for epoch in range(train_cfg['epochs']):
        model.train()
        ep_loss = 0; ep_vel = 0; ep_str = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Batch device transfer (more efficient than individual transfers)
            adj_list = [x.to(device, non_blocking=True) for x in batch[0]]
            xt_list = [x.to(device, non_blocking=True) for x in batch[1]]
            xtp1_list = [x.to(device, non_blocking=True) for x in batch[2]]
            nt_list = [x.to(device, non_blocking=True) for x in batch[6]]
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            # NOTE: xt_list contains normalized input features, xtp1_list contains normalized targets
            with autocast(enabled=use_amp):
                preds = model(adj_list, xt_list, xtp1_list, nt_list)  # Model outputs normalized predictions
                # Loss computed on normalized quantities and normalized by target std^2
                loss, v_loss, s_loss = compute_loss(adj_list, xtp1_list, nt_list, preds, velocity_idxs, stress_idxs,
                                                   std_vel=std_vel, std_stress=std_stress)
            
            # Mixed precision backward pass
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Gradient norm (computed less frequently to reduce overhead)
            if batch_idx % grad_norm_freq == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norms.append(total_norm ** 0.5)
            
            ep_loss += loss.item()
            ep_vel += v_loss if isinstance(v_loss, float) else v_loss.item()
            ep_str += s_loss if isinstance(s_loss, float) else s_loss.item()
            batch_count += 1

        # Validation (with prediction collection for final plots)
        model.eval()
        val_loss = 0; val_vel = 0; val_str = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Batch device transfer
                adj_list = [x.to(device, non_blocking=True) for x in batch[0]]
                xt_list = [x.to(device, non_blocking=True) for x in batch[1]]
                xtp1_list = [x.to(device, non_blocking=True) for x in batch[2]]
                nt_list = [x.to(device, non_blocking=True) for x in batch[6]]
                
                with autocast(enabled=use_amp):
                    # All quantities normalized: inputs, targets, and predictions
                    preds = model(adj_list, xt_list, xtp1_list, nt_list)  # Normalized predictions
                    # Loss normalized by target std^2 for scale-invariance
                    loss, v_l, s_l = compute_loss(adj_list, xtp1_list, nt_list, preds, velocity_idxs, stress_idxs,
                                                  std_vel=std_vel, std_stress=std_stress)
                
                val_loss += loss.item()
                val_vel += v_l if isinstance(v_l, float) else v_l.item()
                val_str += s_l if isinstance(s_l, float) else s_l.item()
                val_batch_count += 1
                
                # Collect predictions for final plots (only on last epoch to save time)
                if epoch == train_cfg['epochs'] - 1:
                    for p, t, nt in zip(preds, xtp1_list, nt_list):
                        v_mask = (nt == NORMAL_NODE)
                        if v_mask.any():
                            all_preds_vel.append(p[v_mask, :3].cpu().numpy())
                            all_targs_vel.append(t[v_mask, velocity_idxs].cpu().numpy())
                        
                        s_mask = (nt == NORMAL_NODE) | (nt == BOUNDARY_NODE)
                        if s_mask.any():
                            all_preds_str.append(p[s_mask, 3:4].cpu().numpy())
                            all_targs_str.append(t[s_mask, stress_idxs].cpu().numpy())
        
        scheduler.step()
        
        # Averages (use actual batch count to handle edge cases)
        avg_train = ep_loss / max(batch_count, 1)
        avg_train_v = ep_vel / max(batch_count, 1)
        avg_train_s = ep_str / max(batch_count, 1)
        
        avg_val = val_loss / max(val_batch_count, 1) if val_batch_count > 0 else 0
        avg_val_v = val_vel / max(val_batch_count, 1) if val_batch_count > 0 else 0
        avg_val_s = val_str / max(val_batch_count, 1) if val_batch_count > 0 else 0
        
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
    # Resolve model_path relative to script directory
    model_path = train_cfg['model_path']
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(script_dir, model_path))
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, "model_egnn.pt"))
    print("Saved model.")

    # Predictions already collected during last validation epoch (optimization)
    # If not collected (shouldn't happen), collect them now
    if not all_preds_vel:
        print("Collecting predictions for plots...")
        with torch.no_grad():
            for batch in test_loader:
                adj_list = [x.to(device, non_blocking=True) for x in batch[0]]
                xt_list = [x.to(device, non_blocking=True) for x in batch[1]]
                xtp1_list = [x.to(device, non_blocking=True) for x in batch[2]]
                nt_list = [x.to(device, non_blocking=True) for x in batch[6]]
                
                with autocast(enabled=use_amp):
                    preds = model(adj_list, xt_list, xtp1_list, nt_list)
                
                for p, t, nt in zip(preds, xtp1_list, nt_list):
                    v_mask = (nt == NORMAL_NODE)
                    if v_mask.any():
                        all_preds_vel.append(p[v_mask, :3].cpu().numpy())
                        all_targs_vel.append(t[v_mask, velocity_idxs].cpu().numpy())
                    
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
    plots_dir = os.path.join(model_path, "plots_egnn")
    
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
    # Fix OpenMP library conflict on macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable multiprocessing for data loading (4 workers for CPU, adjust based on system)
    num_workers = 4 if device.type == 'cpu' else 2
    pin_memory = device.type == 'cuda'
    train_egnn(device, num_workers=num_workers, pin_memory=pin_memory)