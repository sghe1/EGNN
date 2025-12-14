import torch
import os
import yaml
import numpy as np
from data_loader import load_all_trajectories, load_config

# Constants for node types
BOUNDARY_NODE = 3
SPHERE_NODE = 1
NORMAL_NODE = 0

def add_w_edges_neigh(base_A, node_types, pos_t, k):
    """
    For each Sphere node, it looks at its k-nearest neighbors.
    If a neighbor is a Plate node (Normal or Boundary), it adds an edge.
    """
    A_t = base_A.clone()
    
    # Identify Sphere indices
    sphere_indices = torch.nonzero(node_types == SPHERE_NODE, as_tuple=True)[0]
    
    if len(sphere_indices) > 0:
        sphere_pos = pos_t[sphere_indices]
        dists = torch.cdist(sphere_pos, pos_t)
        k_val = min(k + 1, len(pos_t))
        _, neighbor_indices = torch.topk(dists, k=k_val, dim=1, largest=False)
        
        # Get type of neighbors and create a boolean mask for valid connections
        nb_types = node_types[neighbor_indices]
        type_mask = (nb_types == NORMAL_NODE) | (nb_types == BOUNDARY_NODE)
        self_mask = neighbor_indices != sphere_indices.unsqueeze(1)
        valid_mask = type_mask & self_mask
        source_idxs = sphere_indices.unsqueeze(1).expand_as(neighbor_indices)[valid_mask]
        target_idxs = neighbor_indices[valid_mask]
        
        # Update the adjacency matrix
        if len(source_idxs) > 0:
            A_t.index_put_((source_idxs, target_idxs), torch.tensor(1.0, device=A_t.device))
            A_t.index_put_((target_idxs, source_idxs), torch.tensor(1.0, device=A_t.device))
    
    # Normalize adjacency matrix
    row_sums = A_t.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A_t / row_sums
    
    return A_norm

def add_w_edges_radius(base_A, node_types, pos_t, radius):
    """
    Computes A_t dynamically using radius search (OPTIMIZED with sparse operations).
    Excludes existing mesh edges (base_A) and self-loops.
    """
    # Ensure devices match
    if base_A.device != pos_t.device:
        base_A = base_A.to(pos_t.device)
    
    N = pos_t.shape[0]
    
    # OPTIMIZATION: Use sparse operations for radius search
    # Instead of computing all pairwise distances, use a more efficient approach
    # For small graphs, cdist is still reasonable, but we optimize the mask operations
    
    # Compute pairwise distances (unavoidable for radius search, but we'll optimize masking)
    dists = torch.cdist(pos_t, pos_t)
    radius_mask = dists < radius
    radius_mask.fill_diagonal_(False)  # Remove self-loops
    
    # Get existing mesh edges as sparse indices for faster operations
    mesh_edge_mask = base_A > 0
    
    # Exclude existing mesh edges (sparse operation)
    valid_world_mask = radius_mask & (~mesh_edge_mask)
    
    # Build combined adjacency efficiently
    # Use sparse addition: mesh edges + world edges
    A_combined = base_A.clone()
    A_combined[valid_world_mask] = 1.0
    
    # Normalize (row-wise)
    row_sums = A_combined.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A_combined / row_sums
    
    return A_norm

def preprocess_and_save(config):
    data_cfg = config['data']
    train_cfg = config.get('training', {})
    tfrecord_path = data_cfg['tfrecord_path']
    meta_path = data_cfg['meta_path']
    max_trajs = data_cfg['max_trajs']
    norm_method = data_cfg['normalization_method']
    include_mesh_pos = data_cfg['include_mesh_pos']
    
    # World edge parameters from training config (with defaults)
    # NOTE: If you change these parameters, you must re-run preprocessing!
    add_world_edges = train_cfg.get('add_world_edges', 'None')
    radius_world_edge = train_cfg.get('radius_world_edge', 0.03)
    k_neighb = train_cfg.get('k_neighb', 5)
    
    # Define Indices based on config
    if include_mesh_pos:
        mesh_pos_idxs = slice(0, 3)
        world_pos_idxs = slice(3, 6)
        node_type_idxs = slice(6, 8)
        velocity_idxs = slice(8, 11)
        stress_idxs = slice(11, 12)
    else:
        mesh_pos_idxs = None
        world_pos_idxs = slice(0, 3)
        node_type_idxs = slice(3, 5)
        velocity_idxs = slice(5, 8)
        stress_idxs = slice(8, 9)

    # Output Directory Logic
    base_dir = data_cfg['output_dir']
    out_dir_name = f"{base_dir}_{norm_method}_{include_mesh_pos}"
    os.makedirs(out_dir_name, exist_ok=True)
    
    print(f"Preprocessing {max_trajs} trajectories...")
    print(f"World edges: {add_world_edges} (radius={radius_world_edge}, k={k_neighb})")
    
    # Load raw data
    trajs = load_all_trajectories(
        tfrecord_path, meta_path, max_trajs, 
        mesh_pos_idxs, world_pos_idxs, node_type_idxs, 
        velocity_idxs, stress_idxs, 
        include_mesh_pos, norm_method
    )
    
    # Compute world edges for each time step (after normalization)
    print("Computing world edges for each time step...")
    for traj_idx, traj in enumerate(trajs):
        base_A = traj["A"]  # Base mesh adjacency (static)
        X_seq_norm = traj["X_seq_norm"]  # [T, N, F] normalized features
        node_types = traj["node_type"]  # [N] node types
        T, N, F = X_seq_norm.shape
        
        # Extract world positions from normalized features
        pos_seq = X_seq_norm[:, :, world_pos_idxs]  # [T, N, 3]
        
        # Compute adjacency matrix for each time step
        # OPTIMIZATION: Store as sparse COO format for memory efficiency
        A_seq = []
        A_seq_sparse = []  # Store sparse indices and values for faster operations
        
        for t in range(T):
            pos_t = pos_seq[t]  # [N, 3]
            
            if add_world_edges == "radius":
                A_t = add_w_edges_radius(base_A, node_types, pos_t, radius_world_edge)
            elif add_world_edges == "neighbours":
                A_t = add_w_edges_neigh(base_A, node_types, pos_t, k_neighb)
            elif add_world_edges == "None":
                A_t = base_A.clone()
            else:
                raise ValueError(f"Unknown add_world_edges method: {add_world_edges}")
            
            A_seq.append(A_t)
            
            # Also store sparse representation for potential future use
            # Convert to COO format: indices and values
            edge_mask = A_t > 0
            edge_indices = edge_mask.nonzero(as_tuple=False).t()  # [2, E]
            edge_values = A_t[edge_mask]  # [E]
            A_seq_sparse.append({
                'indices': edge_indices,
                'values': edge_values,
                'size': A_t.shape
            })
        
        # Store both dense (for compatibility) and sparse (for future optimization)
        traj["A_seq"] = torch.stack(A_seq, dim=0)  # [T, N, N] - dense for compatibility
        traj["A_seq_sparse"] = A_seq_sparse  # Sparse format for potential future use
        
        if (traj_idx + 1) % 10 == 0:
            print(f"  Processed {traj_idx + 1}/{len(trajs)} trajectories")
    
    print(f"✓ Computed world edges for all {len(trajs)} trajectories")
    
    # Save
    out_path = os.path.join(out_dir_name, "preprocessed_train.pt")
    torch.save(trajs, out_path)
    print(f"Saved to {out_path}")
    
    # Save Metadata
    if len(trajs) > 0:
        meta = {
            "num_trajectories": len(trajs),
            "feature_dim": trajs[0]["X_seq_norm"].shape[2],
            "mean": trajs[0]["mean"],
            "std": trajs[0]["std"],
            "add_world_edges": add_world_edges,
            "radius_world_edge": radius_world_edge if add_world_edges == "radius" else None,
            "k_neighb": k_neighb if add_world_edges == "neighbours" else None
        }
        torch.save(meta, os.path.join(out_dir_name, "preprocessed_metadata.pt"))

if __name__ == "__main__":
    with open("config_egnn.yaml", "r") as f:
        config = yaml.safe_load(f)
    preprocess_and_save(config)