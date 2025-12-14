import torch
import os
import yaml
# Reuse your existing visualization logic to ensure exact comparison
from visualize_simulation import rollout, apply_render_mode, visualize_mesh_pair, load_config, ArgsWrapper
from model_egnn_dense import EGNN_DefPlate 

def main():
    # 1. Config & Setup 
    config_path = "config_egnn.yaml" # Load EGNN config
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    
    # Path logic
    preprocessed_path = train_cfg['datapath']
    add_world_edges = train_cfg['add_world_edges']
    
    # Indices logic (Must match preprocessing)
    if "False" in preprocessed_path:
        world_pos_idxs = slice(0, 3)
        velocity_idxs = slice(5, 8)
        stress_idxs = slice(8, 9)
        dim_in = 12
    else:
        # Default True case
        world_pos_idxs = slice(3, 6)
        velocity_idxs = slice(8, 11)
        stress_idxs = slice(11, 12)
        dim_in = 15

    # 2. Load Data
    print(f"Loading data from {preprocessed_path}...")
    list_of_trajs = torch.load(preprocessed_path)
    traj = list_of_trajs[0] # Visualize first trajectory
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Load Model (EGNN)
    model_args = ArgsWrapper()
    model_args.hid_gnn_layer_dim = model_cfg['hid_gnn_layer_dim']
    model_args.depth = model_cfg['depth']

    model = EGNN_DefPlate(dim_in, 3, 1, model_cfg).to(device)
    
    # Load weights if available
    weights_path = os.path.join(train_cfg['model_path'], "model_egnn.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded EGNN weights.")
    else:
        print(f"Warning: No weights found at {weights_path}, visualizing untrained model.")
    
    model.eval()

    # 4. Alias rollout_step
    # The rollout function expects model.rollout_step(A, X)
    model.rollout_step = model.embed_one

    # 5. Run Rollout
    print("Starting rollout...")
    # Capture dynamic_edges_list (the 5th return value)
    coords, stress, _, _, dynamic_edges_list = rollout(
        model, 
        traj["A"].to(device), 
        traj["X_seq_norm"].to(device), 
        traj["mean"].to(device), 
        traj["std"].to(device),
        t0=0, steps=50, 
        node_type=traj["node_type"].to(device),
        vel_idxs=velocity_idxs, stress_idxs=stress_idxs,
        node_type_idxs=slice(6, 8), 
        world_pos_idxs=world_pos_idxs,
        # Ensure we use the radius logic if config says so
        add_world_edges=(add_world_edges == "radius")
    )
    
    # 6. Visualize Last Step
    step_idx = 49
    print(f"Visualizing step {step_idx+1}...")
    
    # Ground Truth Physics state
    gt_phys = traj["X_seq_norm"][step_idx+1].to(device) * traj["std"].to(device) + traj["mean"].to(device)
    
    # Filter & Plot
    cells = traj["cells"].cpu().numpy()
    nt = traj["node_type"].cpu().numpy()
    
    p_t, p_p, s_t, s_p, nt_t, nt_p, c_f = apply_render_mode(
        gt_phys[:, world_pos_idxs].cpu().numpy(), coords[step_idx], 
        gt_phys[:, stress_idxs].squeeze().cpu().numpy(), stress[step_idx].squeeze(),
        nt, nt, cells
    )
    
    # Pass the captured dynamic edges to the visualizer
    visualize_mesh_pair(
        pos_true=p_t, 
        pos_pred=p_p, 
        cells=c_f, 
        stress_true=s_t, 
        stress_pred=s_p, 
        node_type_true=nt_t, 
        node_type_pred=nt_p, 
        title_true="GT", 
        title_pred="EGNN Pred", 
        color_mode="stress", 
        dynamic_edges=dynamic_edges_list[step_idx] # <--- PLOTTING WORLD EDGES
    )

if __name__ == "__main__":
    main()