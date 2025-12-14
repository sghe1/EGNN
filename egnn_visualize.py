import os
# Fix OpenMP conflict on macOS (must be set before importing torch/numpy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from egnn_data import EGNNTFRecordDataset
from model_egnn import MeshEGNN
from egnn_transform import OverwriteKinematicVelocity, AddDynamicWorldEdges

# Constants
SPHERE_NODE = 1  # [1, 0] in One-Hot
BOUNDARY_NODE = 3 # [0, 1] in One-Hot
NORMAL_NODE = 0   # [0, 0] in One-Hot

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def make_wireframe(x, y, z, i, j, k):
    """Efficient wireframe generation for Plotly"""
    tri_points = np.vstack([i, j, k, i, np.full_like(i, -1)]).T.flatten()
    xe, ye, ze = x[tri_points], y[tri_points], z[tri_points]
    # Insert None to break lines
    xe[4::5] = ye[4::5] = ze[4::5] = None
    
    return go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(color='black', width=1.5),
        showlegend=False, hoverinfo='skip'
    )

def visualize_step(pos_true, pos_pred, cells, stress_true, stress_pred, title):
    # Triangulate cells [C, 4] -> [Triangles]
    # Quad splitting (0,1,2,3) -> (0,1,2) & (0,2,3)
    c0, c1, c2, c3 = cells[:,0], cells[:,1], cells[:,2], cells[:,3]
    tri_i = np.concatenate([c0, c0])
    tri_j = np.concatenate([c1, c2])
    tri_k = np.concatenate([c2, c3])

    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Ground Truth", "Prediction")
    )
    
    # Shared Color Range
    cmin, cmax = 0, max(stress_true.max(), stress_pred.max())
    
    # 1. Ground Truth
    fig.add_trace(go.Mesh3d(
        x=pos_true[:,0], y=pos_true[:,1], z=pos_true[:,2],
        i=tri_i, j=tri_j, k=tri_k,
        intensity=stress_true, colorscale='Viridis', 
        cmin=cmin, cmax=cmax, name='True', opacity=0.9
    ), row=1, col=1)
    
    fig.add_trace(make_wireframe(pos_true[:,0], pos_true[:,1], pos_true[:,2], tri_i, tri_j, tri_k), row=1, col=1)

    # 2. Prediction
    fig.add_trace(go.Mesh3d(
        x=pos_pred[:,0], y=pos_pred[:,1], z=pos_pred[:,2],
        i=tri_i, j=tri_j, k=tri_k,
        intensity=stress_pred, colorscale='Viridis', 
        cmin=cmin, cmax=cmax, name='Pred', opacity=0.9
    ), row=1, col=2)
    
    fig.add_trace(make_wireframe(pos_pred[:,0], pos_pred[:,1], pos_pred[:,2], tri_i, tri_j, tri_k), row=1, col=2)

    fig.update_layout(title=title, height=600, width=1200)
    
    # Save to HTML file for viewing
    os.makedirs("plots", exist_ok=True)
    filename = title.lower().replace(" ", "_") + ".html"
    filepath = os.path.join("plots", filename)
    fig.write_html(filepath)
    print(f"Plot saved to: {filepath}")
    
    # Also try to show in browser
    try:
        fig.show()
    except Exception as e:
        print(f"Could not display plot in browser: {e}")
        print(f"Please open {filepath} in your web browser to view the plot.")

def rollout(model, dataset, traj_idx, steps, device):
    # Load trajectory from RAM cache
    traj = dataset._load_traj_to_cache(traj_idx)
    
    # Stats for Normalization
    mean_feat = dataset.mean_feat.to(device) # [9]
    std_feat = dataset.std_feat.to(device)   # [9]
    mean_targ = dataset.mean_target.to(device) # [4]
    std_targ = dataset.std_target.to(device)   # [4]
    
    # Full Sequence (Raw Tensors)
    x_seq_raw = traj["x"].to(device) # [T, N, 9]
    y_seq_raw = traj["y"].to(device) # [T, N, 4]
    edge_index = traj["edge_index"].to(device)
    node_type = traj["node_type"].to(device)
    
    # Current State (t=0)
    curr_x_raw = x_seq_raw[0].clone()
    
    preds_pos, preds_stress = [], []
    targs_pos, targs_stress = [], []
    errors_mse = []
    
    # Identify kinematic nodes (Sphere)
    sphere_mask = (node_type == SPHERE_NODE)
    
    transform = dataset.transform # Collision Edges + Vel Injection
    
    print(f"Rolling out Trajectory {traj_idx} for {steps} steps...")
    
    for t in range(steps):
        # 1. Normalize Input
        # (x - mean) / std
        x_norm = (curr_x_raw - mean_feat) / std_feat
        
        # 2. Prepare Data Object (for Transforms)
        data = Data(x=x_norm, edge_index=edge_index, node_type=node_type, batch=None)
        
        # Inject Future Velocity from GT (Clairvoyant Physics)
        # We need y_norm for the transform to work
        # Transform expects y to be normalized? Or raw?
        # Check egnn_transforms.py: It reads data.y.
        # Check egnn_train.py: y is Raw until inside loop.
        # But transform runs BEFORE normalization usually. 
        # In egnn_data.py, we normalized x and y.
        # So here 'data.y' should be normalized target.
        
        if t + 1 < len(y_seq_raw):
            y_next_raw = y_seq_raw[t+1]
            y_next_norm = (y_next_raw - mean_targ) / std_targ
            data.y = y_next_norm
        else:
            break
            
        # Apply Transforms (Radius Search + Overwrite Sphere Vel)
        data = transform(data)
        
        # 3. Model Inference
        with torch.no_grad():
            # x is normalized features
            # pos needs to be normalized too (it is part of x[:, 0:3])
            pred_vel_norm, pred_stress_norm = model(data.x, data.x[:, 0:3], data.edge_index)
            
        # 4. Denormalize Predictions
        # Vel is indices 0:3 of Target, Stress is 3
        pred_vel = pred_vel_norm * std_targ[0:3] + mean_targ[0:3]
        pred_stress = pred_stress_norm * std_targ[3] + mean_targ[3]
        
        # 5. Physics Integration (Euler)
        # pos_next = pos_curr + v_pred
        pos_curr = curr_x_raw[:, 0:3]
        pos_next = pos_curr + pred_vel
        
        # 6. Enforce Kinematics (Sphere follows GT)
        gt_pos_next = x_seq_raw[t+1, :, 0:3]
        pos_next[sphere_mask] = gt_pos_next[sphere_mask]
        
        # 7. Update State for Next Step
        # x_raw structure: [Pos(3), Vel(3), Type(2), Stress(1)]
        curr_x_raw[:, 0:3] = pos_next
        curr_x_raw[:, 3:6] = pred_vel # Autoregressive Velocity
        # Type is constant
        curr_x_raw[:, 8:9] = pred_stress # Autoregressive Stress
        
        # 8. Store Results
        preds_pos.append(pos_next.cpu().numpy())
        preds_stress.append(pred_stress.cpu().numpy())
        
        targs_pos.append(gt_pos_next.cpu().numpy())
        targs_stress.append(y_seq_raw[t+1, :, 3:4].cpu().numpy())
        
        # Error
        mse = torch.mean((pos_next - gt_pos_next)**2).item()
        errors_mse.append(mse)

    return preds_pos, preds_stress, targs_pos, targs_stress, errors_mse
def main():
    cfg = load_config("config.yaml")
    device = torch.device(cfg['training']['device'])
    
    # --- CONFIG LOADING ---
    viz_cfg = cfg.get('visualization', {}) # Fallback to empty dict if missing
    start_step = viz_cfg.get('start_step', 0)
    rollout_len = viz_cfg.get('rollout_steps', 50)
    render_t = viz_cfg.get('render_step', 40)
    
    # 1. Load Model
    model = MeshEGNN(
        in_dim=6, 
        hidden_dim=cfg['model']['hidden_nf'],
        depth=cfg['model']['n_layers']
    ).to(device)
    
    ckpt_path = "egnn_final.pt"
    if not os.path.exists(ckpt_path): ckpt_path = "egnn_checkpoint.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Dataset Setup
    transform = Compose([
        OverwriteKinematicVelocity(),
        AddDynamicWorldEdges(radius=cfg['data']['radius'])
    ])
    
    # 3. Trajectory Selection
    train_mode = cfg['training'].get('mode', 'standard')
    
    if train_mode == 'overfit':
        traj_idx = cfg['training']['overfit_traj_ids'][0]
        print(f"Mode: Overfit. Visualizing Traj {traj_idx}")
        dataset = EGNNTFRecordDataset(
            data_dir=cfg['data']['data_dir'],
            preprocessed_dir=cfg['data']['preprocessed_dir'],
            allowed_traj_ids=[traj_idx],
            transform=transform
        )
    else:
        # If config doesn't specify which traj to visualize in standard mode, default to 0
        traj_idx = viz_cfg.get('traj_idx', 0)
        # Limit trajectories if max_trajs is set
        max_trajs = cfg['data'].get('max_trajs')
        allowed_traj_ids = None
        if max_trajs is not None:
            allowed_traj_ids = list(range(max_trajs))
        elif traj_idx is not None:
            # Ensure we load enough trajectories to get to traj_idx
            allowed_traj_ids = list(range(traj_idx + 5))
        
        print(f"Mode: Standard. Visualizing Traj {traj_idx}")
        dataset = EGNNTFRecordDataset(
            data_dir=cfg['data']['data_dir'],
            preprocessed_dir=cfg['data']['preprocessed_dir'],
            transform=transform,
            allowed_traj_ids=allowed_traj_ids
        )

    # 4. Load Cells
    import json
    from tfrecord.reader import tfrecord_loader
    from egnn_preprocess import decode
    
    with open(os.path.join(cfg['data']['data_dir'], "meta.json")) as f:
        meta = json.load(f)
    
    split_name = cfg['data']['split']
    loader = tfrecord_loader(os.path.join(cfg['data']['data_dir'], f"{split_name}.tfrecord"), None)
    cells = None
    for i, rec in enumerate(loader):
        if i == traj_idx:
            cells = decode(rec['cells'], meta["features"]["cells"]["shape"], np.int32)
            break
            
    if cells is None:
        raise ValueError(f"Cells not found for trajectory {traj_idx}")

    # 5. Rollout
    print(f"Running rollout for trajectory {traj_idx}, {rollout_len} steps...")
    try:
        p_pred, s_pred, p_true, s_true, errors = rollout(model, dataset, traj_idx, rollout_len, device)
        print(f"Rollout completed. Generated {len(p_pred)} prediction steps.")
        
        # 6. Visualize
        # Ensure we don't crash if rollout was shorter than requested render step
        viz_t = min(render_t, len(p_pred)-1)
        if viz_t < 0:
            print("Error: No prediction steps generated!")
            return
        
        print(f"Visualizing step {viz_t}. MSE: {errors[viz_t]:.6f}")
        
        visualize_step(
            p_true[viz_t], p_pred[viz_t], cells, 
            s_true[viz_t].flatten(), s_pred[viz_t].flatten(), 
            title=f"EGNN Rollout Step {viz_t} (Traj {traj_idx})"
        )
        
        # Plot error curve
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(y=errors, mode='lines+markers', name='MSE'))
        fig_err.update_layout(title=f"Rollout Error - Traj {traj_idx}")
        os.makedirs("plots", exist_ok=True)
        fig_err.write_html(os.path.join("plots", f"rollout_error_traj_{traj_idx}.html"))
        fig_err.show()
        
        print("Visualization complete!")
    except Exception as e:
        print(f"Error during rollout or visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()