"""
DataLoader utilities for EGNN training.

This module contains the custom collate function for batching variable-sized graphs.
"""

import torch


def custom_collate_fn(batch):
    """
    Custom collate function for variable-sized graphs.
    
    Implements graph batching by concatenating all graphs into one large graph
    with a block-diagonal adjacency matrix. This allows batching graphs with
    different numbers of nodes.
    
    Strategy:
    - If batch_size=1, return the single item directly
    - If batch_size > 1, always use graph batching (works for both same-size and different-size graphs):
      * Concatenate all node features, coordinates, targets along the node dimension
      * Create block-diagonal adjacency matrix (ensures nodes from different graphs don't interact)
      * Create batch_index to track which nodes belong to which graph
      * Process as single batch item (B=1) with total_N nodes
    """
    # CRITICAL: This function MUST be called by DataLoader
    # If you see errors about tensor size mismatches, it means default_collate was called instead
    # Validate batch format - ensure all items are dictionaries
    if not isinstance(batch, list):
        raise TypeError(f"Expected batch to be a list, got {type(batch)}")
    
    if len(batch) == 0:
        raise ValueError("Batch is empty")
    
    # Check that all items are dictionaries
    # If any item is not a dict, it means default_collate was called first (shouldn't happen)
    for i, item in enumerate(batch):
        if not isinstance(item, dict):
            raise TypeError(
                f"Batch item {i} is not a dictionary, got {type(item)}. "
                f"This suggests default_collate was called before custom_collate_fn. "
                f"Item: {item if not isinstance(item, torch.Tensor) else f'Tensor of shape {item.shape}'}"
            )
    
    # If batch_size=1, just return the single item
    if len(batch) == 1:
        return batch[0]
    
    # For batch_size > 1, always use graph batching
    # This is more robust and handles both same-size and different-size graphs
    # Graph batching concatenates all graphs into one large graph with block-diagonal adjacency
    # Standard approach: concatenate all graphs and process as single batch item (B=1)
    # Get dimensions
    num_graphs = len(batch)
    
    # Validate that all items have required keys
    required_keys = ['coors', 'feats', 'target_vel', 'target_stress', 'world_pos', 'adj_mat']
    for i, item in enumerate(batch):
        for key in required_keys:
            if key not in item:
                raise KeyError(f"Batch item {i} missing required key '{key}'. Available keys: {list(item.keys())}")
    
    T = batch[0]['coors'].shape[0]  # All should have same T
    node_counts = [item['coors'].shape[1] for item in batch]  # Different N per graph
    total_nodes = sum(node_counts)
    
    # Verify all have same T
    assert all(item['coors'].shape[0] == T for item in batch), \
        f"All trajectories must have same number of timesteps T, got {[item['coors'].shape[0] for item in batch]}"
    
    # Initialize concatenated tensors (B=1 for concatenated graph)
    device = batch[0]['coors'].device
    dtype = batch[0]['coors'].dtype
    
    coors_batched = torch.zeros(1, T, total_nodes, 3, dtype=dtype, device=device)
    feats_batched = torch.zeros(1, T, total_nodes, batch[0]['feats'].shape[2], dtype=dtype, device=device)
    target_vel_batched = torch.zeros(1, T, total_nodes, 3, dtype=dtype, device=device)
    target_stress_batched = torch.zeros(1, T, total_nodes, 1, dtype=dtype, device=device)
    world_pos_batched = torch.zeros(1, T, total_nodes, 3, dtype=dtype, device=device)
    
    # Create block-diagonal adjacency matrix (single batch item)
    adj_mat_batched = torch.zeros(1, total_nodes, total_nodes, dtype=torch.bool, device=device)
    
    # Create batch index: which graph does each node belong to?
    batch_index = torch.zeros(total_nodes, dtype=torch.long, device=device)
    
    # Concatenate each graph along the node dimension
    node_offset = 0
    for graph_idx, item in enumerate(batch):
        n_nodes = node_counts[graph_idx]
        
        # Handle input format: item might be (T, N, ...) or already have batch dimension
        coors_item = item['coors']
        feats_item = item['feats']
        target_vel_item = item['target_vel']
        target_stress_item = item['target_stress']
        world_pos_item = item['world_pos']
        adj_mat_item = item['adj_mat']
        
        # Remove batch dimension if present (should be single trajectory)
        if len(coors_item.shape) == 4:  # (1, T, N, 3)
            coors_item = coors_item[0]  # (T, N, 3)
            feats_item = feats_item[0]
            target_vel_item = target_vel_item[0]
            target_stress_item = target_stress_item[0]
            world_pos_item = world_pos_item[0]
        
        # Copy data into concatenated tensor (all graphs in single batch item)
        coors_batched[0, :, node_offset:node_offset+n_nodes, :] = coors_item
        feats_batched[0, :, node_offset:node_offset+n_nodes, :] = feats_item
        target_vel_batched[0, :, node_offset:node_offset+n_nodes, :] = target_vel_item
        target_stress_batched[0, :, node_offset:node_offset+n_nodes, :] = target_stress_item
        world_pos_batched[0, :, node_offset:node_offset+n_nodes, :] = world_pos_item
        
        # Add adjacency matrix as block (handle both 2D and 3D adj_mat)
        if len(adj_mat_item.shape) == 2:
            # (N, N) - direct assignment
            adj_mat_batched[0, node_offset:node_offset+n_nodes, node_offset:node_offset+n_nodes] = adj_mat_item
        elif len(adj_mat_item.shape) == 3:
            # (1, N, N) or (B_item, N, N) - take first
            adj_mat_batched[0, node_offset:node_offset+n_nodes, node_offset:node_offset+n_nodes] = adj_mat_item[0]
        else:
            raise ValueError(f"Unexpected adj_mat shape: {adj_mat_item.shape}")
        
        # Set batch index
        batch_index[node_offset:node_offset+n_nodes] = graph_idx
        
        node_offset += n_nodes
    
    return {
        'coors': coors_batched,  # (1, T, total_N, 3)
        'feats': feats_batched,  # (1, T, total_N, feat_dim)
        'adj_mat': adj_mat_batched,  # (1, total_N, total_N) - block diagonal
        'target_vel': target_vel_batched,  # (1, T, total_N, 3)
        'target_stress': target_stress_batched,  # (1, T, total_N, 1)
        'world_pos': world_pos_batched,  # (1, T, total_N, 3)
        'batch_index': batch_index,  # (total_N,) - which graph each node belongs to
        'node_counts': node_counts,  # List of node counts per graph (for loss computation)
    }
