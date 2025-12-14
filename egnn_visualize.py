import os
# Fix OpenMP conflict on macOS (must be set before importing torch/numpy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from egnn_data import EGNNTFRecordDataset
from model_egnn import MeshEGNN
from egnn_transform import OverwriteKinematicVelocity, AddDynamicWorldEdges

# Constants
SPHERE_NODE = 1
NORMAL_NODE = 0
BOUNDARY_NODE = 3 # Or whatever your 'type' map is. In preprocess: 0=Plate, 1=Sphere, 3=Handle

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def make_wireframe(x, y, z, i, j, k, color='black', width=1.5):
    tri_points = np.vstack([i, j, k, i, np.full_like(i, -1)]).T.flatten()
    xe, ye, ze = x[tri_points], y[tri_points], z[tri_points]
    xe[4::5] = ye[4::5] = ze[4::5] = None
    return go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color=color, width=width), showlegend=False, hoverinfo='skip')

def visualize_mesh_pair(pos_true, pos_pred, cells, stress_true, stress_pred, title):
    # Triangulate cells [C, 4] -> [Triangles]
    # Standard splitting of quad (0,1,2,3) -> (0,1,2) & (0,2,3)
    tri_i, tri_j, tri_k = [], [], []
    for c in cells:
        tri_i.extend([c[0], c[0]]); tri_j.extend([c[1], c[2]]); tri_k.extend([c[2], c[3]])
    
    tri_i, tri_j, tri_k = np.array(tri_i), np.array(tri_j), np.array(tri_k)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]], subplot_titles=("Ground Truth", "Prediction"))
    
    # Common settings
    cmin, cmax = 0, max(stress_true.max(), stress_pred.max())
    
    # 1. True
    fig.add_trace(go.Mesh3d(x=pos_true[:,0], y=pos_true[:,1], z=pos_true[:,2], i=tri_i, j=tri_j, k=tri_k, 
                            intensity=stress_true, colorscale='Viridis', cmin=cmin, cmax=cmax, name='True'), row=1, col=1)
    fig.add_trace(make_wireframe(pos_true[:,0], pos_true[:,1], pos_true[:,2], tri_i, tri_j, tri_k), row=1, col=1)

    # 2. Pred
    fig.add_trace(go.Mesh3d(x=pos_pred[:,0], y=pos_pred[:,1], z=pos_pred[:,2], i=tri_i, j=tri_j, k=tri_k, 
                            intensity=stress_pred, colorscale='Viridis', cmin=cmin, cmax=cmax, name='Pred'), row=1, col=2)
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
    # Load single trajectory
    # We cheat and use the dataset internal method to get full sequence
    traj = dataset._load_traj_to_cache(traj_idx)
    x_seq = traj["x"].to(device) # [T, N, 9]
    y_seq = traj["y"].to(device) # [T, N, 4]
    edge_index = traj["edge_index"].to(device)
    node_type = traj["node_type"].to(device)
    
    # Norm stats
    mean_t = dataset.mean_target.to(device)
    std_t = dataset.std_target.to(device)
    
    # Initial State (t=0)
    curr_x = x_seq[0].clone() # [N, 9] -> [Pos(3), Vel(3), Type(2), Stress(1)]
    
    # Transform logic (Radius Graph) needs to be applied dynamically
    # We must construct a 'Data' object for the transforms
    from torch_geometric.data import Data
    
    transform = dataset.transform # Collision + Vel Overwrite
    
    preds_pos, preds_stress = [], []
    targs_pos, targs_stress = [], []
    
    # Track rigid body path (Ground Truth Position)
    sphere_mask = (node_type == SPHERE_NODE)
    
    print(f"Rolling out Trajectory {traj_idx} for {steps} steps...")
    
    for t in range(steps):
        # 1. Prepare Input Data
        # We need to construct a Data object so transforms work
        data = Data(x=curr_x, edge_index=edge_index, node_type=node_type, batch=None)
        
        # Inject Future Velocity from GT (Clairvoyant Physics)
        # transform needs 'y' to steal velocity. GT y is at t+1
        if t + 1 < len(y_seq):
            data.y = y_seq[t+1] # Next step target
        else:
            break # End of sequence
            
        # Apply Transforms (Radius Graph + Vel Injection)
        data = transform(data)
        
        # 2. Forward
        with torch.no_grad():
            # EGNN needs: x, pos, edge_index
            pred_vel, pred_stress = model(data.x, data.x[:, 0:3], data.edge_index)
        
        # 3. Integrate Physics
        # Denormalize Velocity first? No, model output matches dataset y scale.
        # But for position integration, we need physical scale if normalized?
        # WAIT: In our egnn_preprocess, 'x' contains RAW positions (not normalized yet?)
        # Let's check egnn_preprocess... 
        # Ah, egnn_preprocess saves raw values, BUT we usually normalize in the dataset logic or model?
        # NO. Our egnn_preprocess calculates stats but SAVES RAW TENSORS.
        # The normalization usually happens inside __getitem__ or using a transform.
        # CHECK: Did we apply normalization in egnn_data.py? NO.
        # This means our model currently trains on RAW VALUES.
        # EGNN handles this usually fine, but for stability usually we normalize.
        # Assuming RAW for now as per your request to keep it simple.
        
        # Euler Integration: p_next = p_curr + v_pred
        pos_curr = curr_x[:, 0:3]
        pos_next = pos_curr + pred_vel
        
        # Enforce Boundary Conditions
        # Sphere: Follow GT path exactly
        gt_pos_next = x_seq[t+1, :, 0:3]
        pos_next[sphere_mask] = gt_pos_next[sphere_mask]
        
        # Update State for next step
        curr_x[:, 0:3] = pos_next # Update Pos
        curr_x[:, 3:6] = pred_vel # Update Vel (Autoregressive)
        curr_x[:, 8:9] = pred_stress # Update Stress (Autoregressive)
        
        # Store
        preds_pos.append(pos_next.cpu().numpy())
        preds_stress.append(pred_stress.cpu().numpy())
        
        targs_pos.append(gt_pos_next.cpu().numpy())
        targs_stress.append(y_seq[t+1, :, 3:4].cpu().numpy())

    return preds_pos, preds_stress, targs_pos, targs_stress

def main():
    cfg = load_config("config.yaml")
    device = torch.device(cfg['training']['device'])
    
    # Load Model
    model = MeshEGNN(
        in_dim=cfg['model']['in_node_nf'],
        hidden_dim=cfg['model']['hidden_nf'],
        depth=cfg['model']['n_layers']
    ).to(device)
    model.load_state_dict(torch.load("egnn_final.pt", map_location=device))
    model.eval()
    
    # Load Dataset (for transforms & stats)
    transform = Compose([
        OverwriteKinematicVelocity(),
        AddDynamicWorldEdges(radius=cfg['data']['radius'])
    ])
    # Limit trajectories if max_trajs is set
    max_trajs = cfg['data'].get('max_trajs')
    allowed_traj_ids = None
    if max_trajs is not None:
        allowed_traj_ids = list(range(max_trajs))
    
    dataset = EGNNTFRecordDataset(
        data_dir=cfg['data']['data_dir'],
        preprocessed_dir=cfg['data']['preprocessed_dir'],
        transform=transform,
        allowed_traj_ids=allowed_traj_ids
    )
    
    # Load cells for viz (from meta)
    import json
    with open(os.path.join(cfg['data']['data_dir'], "meta.json")) as f:
        meta = json.load(f)
    # We assume cells are constant, load from first record logic or raw
    # Hack: just load traj 0 cells from raw if possible, or store in preprocess
    # Better: cells are static, let's load from traj 0 pt file
    # We didn't save cells in .pt (only edge_index). We need cells for plotly surfaces.
    # Quick fix: Load raw tfrecord 0 just to get cells
    from tfrecord.reader import tfrecord_loader
    loader = tfrecord_loader(os.path.join(cfg['data']['data_dir'], "train.tfrecord"), None)
    rec = next(loader)
    # Decode cells using the same method as preprocess
    from egnn_preprocess import decode
    cells = decode(rec['cells'], meta["features"]["cells"]["shape"], np.int32)

    # Run Rollout
    idx = 0 # Trajectory to visualize
    steps = 50
    print(f"Running rollout for trajectory {idx}, {steps} steps...")
    try:
        p_pred, s_pred, p_true, s_true = rollout(model, dataset, idx, steps, device)
        print(f"Rollout completed. Generated {len(p_pred)} prediction steps.")
        
        # Visualize Step 40 (or last available step)
        t = min(40, len(p_pred) - 1)
        if t < 0:
            print("Error: No prediction steps generated!")
            return
        
        print(f"Visualizing step {t}...")
        visualize_mesh_pair(
            p_true[t], p_pred[t], cells, 
            s_true[t].flatten(), s_pred[t].flatten(), 
            title=f"EGNN Rollout Step {t}"
        )
        print("Visualization complete!")
    except Exception as e:
        print(f"Error during rollout or visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()