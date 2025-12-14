import torch
import os
import yaml
from data_loader import load_all_trajectories, load_config

def preprocess_and_save(config):
    data_cfg = config['data']
    tfrecord_path = data_cfg['tfrecord_path']
    meta_path = data_cfg['meta_path']
    max_trajs = data_cfg['max_trajs']
    norm_method = data_cfg['normalization_method']
    include_mesh_pos = data_cfg['include_mesh_pos']
    
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
    
    # Load raw data
    trajs = load_all_trajectories(
        tfrecord_path, meta_path, max_trajs, 
        mesh_pos_idxs, world_pos_idxs, node_type_idxs, 
        velocity_idxs, stress_idxs, 
        include_mesh_pos, norm_method
    )
    
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
            "std": trajs[0]["std"]
        }
        torch.save(meta, os.path.join(out_dir_name, "preprocessed_metadata.pt"))

if __name__ == "__main__":
    with open("config_egnn.yaml", "r") as f:
        config = yaml.safe_load(f)
    preprocess_and_save(config)