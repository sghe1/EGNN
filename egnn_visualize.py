import os
# Fix OpenMP conflict on macOS
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
SPHERE_NODE = 1  
BOUNDARY_NODE = 3
NORMAL_NODE = 0 

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def make_wireframe(x, y, z, i, j, k):
    """Efficient wireframe generation for Plotly"""
    tri_points = np.vstack([i, j, k, i, np.full_like(i, -1)]).T.flatten()
    xe, ye, ze = x[tri_points], y[tri_points], z[tri_points]
    xe[4::5] = ye[4::5] = ze[4::5] = None
    
    return go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(color='black', width=1.5),
        showlegend=False, hoverinfo='skip'
    )

def visualize_rollout_animation(pos_true_seq, pos_pred_seq, cells, stress_true_seq, stress_pred_seq, title, output_path):
    """
    Creates a single HTML file with a time slider to play the rollout animation.
    """
    print(f"Generating animation for {len(pos_true_seq)} steps...")
    
    # Triangulate cells
    c0, c1, c2, c3 = cells[:,0], cells[:,1], cells[:,2], cells[:,3]
    tri_i = np.concatenate([c0, c0])
    tri_j = np.concatenate([c1, c2])
    tri_k = np.concatenate([c2, c3])

    # Global Color Range (Fixed across time for consistency)
    # Convert lists to single numpy array for min/max
    all_s_true = np.concatenate(stress_true_seq)
    all_s_pred = np.concatenate(stress_pred_seq)
    cmin, cmax = 0, max(all_s_true.max(), all_s_pred.max())

    # Create Initial Figure
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Ground Truth", "Prediction")
    )

    # --- Helper to create traces for a specific step ---
    def get_traces(t):
        p_t = pos_true_seq[t]
        p_p = pos_pred_seq[t]
        s_t = stress_true_seq[t]
        s_p = stress_pred_seq[t]
        
        # 1. True Surface
        t1 = go.Mesh3d(x=p_t[:,0], y=p_t[:,1], z=p_t[:,2], i=tri_i, j=tri_j, k=tri_k,
                       intensity=s_t, colorscale='Viridis', cmin=cmin, cmax=cmax, name='True')
        # 2. True Wireframe
        t2 = make_wireframe(p_t[:,0], p_t[:,1], p_t[:,2], tri_i, tri_j, tri_k)
        
        # 3. Pred Surface
        t3 = go.Mesh3d(x=p_p[:,0], y=p_p[:,1], z=p_p[:,2], i=tri_i, j=tri_j, k=tri_k,
                       intensity=s_p, colorscale='Viridis', cmin=cmin, cmax=cmax, name='Pred')
        # 4. Pred Wireframe
        t4 = make_wireframe(p_p[:,0], p_p[:,1], p_p[:,2], tri_i, tri_j, tri_k)
        
        return [t1, t2, t3, t4]

    # Add Initial Data (Step 0)
    for trace in get_traces(0):
        fig.add_trace(trace)

    # --- Create Frames for Animation ---
    frames = []
    steps = len(pos_true_seq)
    for t in range(steps):
        # Only update the data arrays, reuse layout properties
        # Order must match the add_trace order: [MeshT, WireT, MeshP, WireP]
        
        # Pre-calculate wireframe arrays
        p_t = pos_true_seq[t]
        wire_t = make_wireframe(p_t[:,0], p_t[:,1], p_t[:,2], tri_i, tri_j, tri_k)
        
        p_p = pos_pred_seq[t]
        wire_p = make_wireframe(p_p[:,0], p_p[:,1], p_p[:,2], tri_i, tri_j, tri_k)
        
        frames.append(go.Frame(
            data=[
                # Update Trace 0 (True Mesh)
                go.Mesh3d(x=p_t[:,0], y=p_t[:,1], z=p_t[:,2], intensity=stress_true_seq[t]),
                # Update Trace 1 (True Wire)
                go.Scatter3d(x=wire_t.x, y=wire_t.y, z=wire_t.z),
                # Update Trace 2 (Pred Mesh)
                go.Mesh3d(x=p_p[:,0], y=p_p[:,1], z=p_p[:,2], intensity=stress_pred_seq[t]),
                # Update Trace 3 (Pred Wire)
                go.Scatter3d(x=wire_p.x, y=wire_p.y, z=wire_p.z)
            ],
            name=str(t)
        ))

    fig.frames = frames

    # --- Add Slider and Buttons ---
    fig.update_layout(
        title=title, height=600, width=1200,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top", "xanchor": "left",
            "currentvalue": {"font": {"size": 20}, "prefix": "Step:", "visible": True, "xanchor": "right"},
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9, "x": 0.1, "y": 0,
            "steps": [{"args": [[str(k)], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}], "label": str(k), "method": "animate"} for k in range(steps)]
        }]
    )

    # Fix camera aspect ratio
    fig.update_layout(scene_aspectmode='data', scene2_aspectmode='data')
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"Animation saved to: {output_path}")
    
    # Try to open
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(output_path))
    except:
        pass

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
        x_norm = (curr_x_raw - mean_feat) / std_feat
        
        # 2. Prepare Data Object (for Transforms)
        data = Data(x=x_norm, edge_index=edge_index, node_type=node_type, batch=None)
        
        # Inject Future Velocity from GT (Clairvoyant Physics)
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
            pred_vel_norm, pred_stress_norm = model(data.x, data.x[:, 0:3], data.edge_index)
            
        # 4. Denormalize Predictions
        pred_vel = pred_vel_norm * std_targ[0:3] + mean_targ[0:3]
        pred_stress = pred_stress_norm * std_targ[3] + mean_targ[3]
        
        # 5. Physics Integration (Euler)
        pos_curr = curr_x_raw[:, 0:3]
        pos_next = pos_curr + pred_vel
        
        # 6. Enforce Kinematics (Sphere follows GT)
        gt_pos_next = x_seq_raw[t+1, :, 0:3]
        pos_next[sphere_mask] = gt_pos_next[sphere_mask]
        
        # 7. Update State for Next Step
        curr_x_raw[:, 0:3] = pos_next
        curr_x_raw[:, 3:6] = pred_vel 
        curr_x_raw[:, 8:9] = pred_stress
        
        # 8. Store Results (Numpy)
        preds_pos.append(pos_next.cpu().numpy())
        preds_stress.append(pred_stress.cpu().numpy().flatten()) # Flatten for coloring
        
        targs_pos.append(gt_pos_next.cpu().numpy())
        targs_stress.append(y_seq_raw[t+1, :, 3:4].cpu().numpy().flatten())
        
        # Error
        mse = torch.mean((pos_next - gt_pos_next)**2).item()
        errors_mse.append(mse)

    return preds_pos, preds_stress, targs_pos, targs_stress, errors_mse

def main():
    cfg = load_config("config.yaml")
    device = torch.device(cfg['training']['device'])
    
    # --- CONFIG LOADING ---
    viz_cfg = cfg.get('visualization', {})
    rollout_len = viz_cfg.get('rollout_steps', 50)
    
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
        allowed_traj_ids = [traj_idx]
    else:
        traj_idx = viz_cfg.get('traj_idx', 0)
        allowed_traj_ids = list(range(traj_idx + 5))
        
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
            
    if cells is None: raise ValueError(f"Cells not found for trajectory {traj_idx}")

    # 5. Rollout
    print(f"Running rollout for trajectory {traj_idx}, {rollout_len} steps...")
    p_pred, s_pred, p_true, s_true, errors = rollout(model, dataset, traj_idx, rollout_len, device)
    
    # 6. Visualize Animation (ALL STEPS)
    output_path = os.path.join("plots", f"animation_traj_{traj_idx}.html")
    
    visualize_rollout_animation(
        p_true, p_pred, cells, s_true, s_pred, 
        title=f"EGNN Physics Rollout (Traj {traj_idx})",
        output_path=output_path
    )
    
    # Plot error curve
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(y=errors, mode='lines+markers', name='MSE'))
    fig_err.update_layout(title=f"Rollout Error - Traj {traj_idx}", xaxis_title="Step", yaxis_title="MSE")
    fig_err.write_html(os.path.join("plots", f"error_traj_{traj_idx}.html"))
    fig_err.show()

if __name__ == "__main__":
    main()