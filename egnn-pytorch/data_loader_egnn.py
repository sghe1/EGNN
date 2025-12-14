"""
Data loading utilities for deforming_plate dataset.

Dataset Structure (per trajectory):
- 400 time steps
- Variable number of nodes (~1271 on average)
- Each TFRecord example = one full trajectory

Input Node Features (8 dimensions):
- position(3): Current 3D world position (x, y, z) at time t
- actuation(3): Boundary displacement / input signal for the node
- node_type_one_hot(2): One-hot encoding [is_plate_node, is_actuated_node]

Targets (separate from inputs):
- velocity: (T, N, 3) - Lagrangian velocity computed as pos[t] - pos[t-1]
- stress: (T, N, 1) - Scalar von-Mises stress per node per timestep

Mesh Connectivity:
- cells: (C, 4) - Tetrahedral cell connectivity (4 node indices per cell)
- Used to derive graph edges (element adjacency)

Important:
- Velocity and stress are TARGETS only, NOT part of input features
- Position in features is the same as coordinates (world_pos)
- Actuation may not be in TFRecord - if missing, set to zeros
"""

import json
import numpy as np
import torch
from tfrecord.reader import tfrecord_loader


# ============================================================
# 1) DECODIFICA RECORD RAW (NO TENSORFLOW)
# ============================================================

def decode_trajectory_from_record(record, meta):
    """
    Decode a trajectory record from TFRecord with robust error handling.
    
    Handles:
    - bytes
    - numeric np.ndarray
    - np.ndarray(object) containing bytes
    
    Uses np.frombuffer + reshape with shapes from meta.json.
    
    Features:
    - Validates shape_spec contains at most one -1
    - Validates total elements match inferred shape
    - Safely squeezes leading singleton dimensions (only if rank >= 2)
    - Ensures actuation is always present (zeros if missing)
    - Validates shapes: world_pos (T, N, 3), stress (T, N, 1), mesh_pos (N, 3), cells (C, 4)
    
    Args:
        record: Dictionary from tfrecord_loader
        meta: Metadata dictionary with "features" key containing shape specifications
    
    Returns:
        Dictionary with keys:
        - world_pos: (T, N, 3) np.float32
        - stress: (T, N, 1) np.float32
        - node_type: (N, 1) or (N,) np.int32
        - mesh_pos: (N, 3) np.float32
        - cells: (C, 4) np.int32
        - actuation: (T, N, 3) np.float32 (always present)
    
    Raises:
        ValueError: If shape validation fails or reshaping is impossible
        KeyError: If required keys are missing from record or meta
    """

    def normalize_to_bytes(value):
        """
        Estrae sempre dei bytes, ma SOLO se value NON è già un ndarray numerico.
        """
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return bytes(value.flat[0])
            # ndarray numerico → NON gestito qui
            raise TypeError("normalize_to_bytes chiamato su ndarray numerico")

        raise TypeError(f"Tipo inatteso: {type(value)}")

    def decode_raw_array(value, dtype, shape_spec):
        """
        Decode raw array from TFRecord with robust shape handling.
        
        Args:
            value: Can be:
                - np.ndarray (numeric) → cast and reshape
                - bytes / np.ndarray(object) → np.frombuffer + reshape
            dtype: Target numpy dtype
            shape_spec: Target shape tuple/list. Must contain exactly one -1 for inference.
        
        Returns:
            np.ndarray with shape matching shape_spec (with -1 resolved)
        
        Raises:
            ValueError: If shape_spec contains more than one -1, or if reshaping is impossible
        """
        # Validate shape_spec: must contain exactly one -1
        if isinstance(shape_spec, (int, np.integer)):
            shape_spec = [shape_spec]
        shape_spec = list(shape_spec)
        
        num_wildcards = sum(1 for d in shape_spec if d == -1)
        if num_wildcards > 1:
            raise ValueError(
                f"shape_spec must contain at most one -1, got {num_wildcards} wildcards: {shape_spec}"
            )
        
        # Case 1: Already numeric ndarray
        if isinstance(value, np.ndarray) and value.dtype != object:
            arr = value.astype(dtype)
            total_elements = arr.size
            
            # Build target shape
            tgt = list(shape_spec)
            if -1 in tgt:
                # Compute known dimensions product
                known_product = 1
                for d in tgt:
                    if d != -1:
                        if d <= 0:
                            raise ValueError(f"Invalid shape dimension {d} in shape_spec {shape_spec}")
                        known_product *= d
                
                # Infer missing dimension
                if known_product == 0:
                    raise ValueError(f"Cannot infer dimension: known_product is 0 in shape_spec {shape_spec}")
                
                if total_elements % known_product != 0:
                    raise ValueError(
                        f"Cannot reshape array of size {total_elements} to shape {shape_spec}: "
                        f"not divisible by known dimensions product {known_product}"
                    )
                
                missing = total_elements // known_product
                tgt[tgt.index(-1)] = missing
            else:
                # No wildcard: validate total size matches
                expected_size = 1
                for d in tgt:
                    if d <= 0:
                        raise ValueError(f"Invalid shape dimension {d} in shape_spec {shape_spec}")
                    expected_size *= d
                
                if total_elements != expected_size:
                    raise ValueError(
                        f"Cannot reshape array of size {total_elements} to shape {shape_spec}: "
                        f"expected {expected_size} elements"
                    )
            
            # Reshape if needed
            if tuple(arr.shape) != tuple(tgt):
                arr = arr.reshape(tgt)
            return arr

        # Case 2: bytes or object array → frombuffer
        raw = normalize_to_bytes(value)
        arr = np.frombuffer(raw, dtype=dtype)
        total_elements = arr.size
        
        # Build target shape
        tgt = list(shape_spec)
        if -1 in tgt:
            # Compute known dimensions product
            known_product = 1
            for d in tgt:
                if d != -1:
                    if d <= 0:
                        raise ValueError(f"Invalid shape dimension {d} in shape_spec {shape_spec}")
                    known_product *= d
            
            # Infer missing dimension
            if known_product == 0:
                raise ValueError(f"Cannot infer dimension: known_product is 0 in shape_spec {shape_spec}")
            
            if total_elements % known_product != 0:
                raise ValueError(
                    f"Cannot reshape array of size {total_elements} to shape {shape_spec}: "
                    f"not divisible by known dimensions product {known_product}"
                )
            
            missing = total_elements // known_product
            tgt[tgt.index(-1)] = missing
        else:
            # No wildcard: validate total size matches
            expected_size = 1
            for d in tgt:
                if d <= 0:
                    raise ValueError(f"Invalid shape dimension {d} in shape_spec {shape_spec}")
                expected_size *= d
            
            if total_elements != expected_size:
                raise ValueError(
                    f"Cannot reshape array of size {total_elements} to shape {shape_spec}: "
                    f"expected {expected_size} elements"
                )

        return arr.reshape(tgt)

    shapes = meta["features"]

    # Decode arrays
    world_pos = decode_raw_array(record["world_pos"], np.float32,
                                 shapes["world_pos"]["shape"])
    stress = decode_raw_array(record["stress"], np.float32,
                              shapes["stress"]["shape"])
    node_type = decode_raw_array(record["node_type"], np.int32,
                                 shapes["node_type"]["shape"])
    mesh_pos = decode_raw_array(record["mesh_pos"], np.float32,
                                shapes["mesh_pos"]["shape"])
    cells = decode_raw_array(record["cells"], np.int32,
                             shapes["cells"]["shape"])

    # Safely squeeze leading singleton dimensions (only if rank >= 2)
    # This avoids accidentally dropping nodes for 1D arrays
    def safe_squeeze_leading(arr, min_rank=2):
        """Squeeze leading dimension only if array has rank >= min_rank and leading dim is 1"""
        if arr.ndim >= min_rank and arr.shape[0] == 1:
            return arr[0]  # Remove leading dimension
        return arr

    node_type = safe_squeeze_leading(node_type, min_rank=2)
    mesh_pos = safe_squeeze_leading(mesh_pos, min_rank=2)
    cells = safe_squeeze_leading(cells, min_rank=2)

    # Ensure actuation is always present
    # Try "actuation" first, then "boundary_displacement", else use zeros
    actuation = None
    if "actuation" in record and "actuation" in shapes:
        actuation = decode_raw_array(record["actuation"], np.float32,
                                     shapes["actuation"]["shape"])
        actuation = actuation.astype(np.float32)
    elif "boundary_displacement" in record and "boundary_displacement" in shapes:
        actuation = decode_raw_array(record["boundary_displacement"], np.float32,
                                     shapes["boundary_displacement"]["shape"])
        actuation = actuation.astype(np.float32)
    
    # If actuation is still None, create zeros array matching world_pos shape
    if actuation is None:
        actuation = np.zeros_like(world_pos, dtype=np.float32)
    
    # Validate actuation shape matches world_pos
    if actuation.shape != world_pos.shape:
        # If actuation is static (N, 3), tile it over time
        T, N, _ = world_pos.shape
        if actuation.shape == (N, 3):
            actuation = np.tile(actuation[np.newaxis, :, :], (T, 1, 1))
        else:
            raise ValueError(
                f"Actuation shape {actuation.shape} does not match world_pos shape {world_pos.shape}. "
                f"Expected (T, N, 3) or (N, 3), got {actuation.shape}"
            )

    # Shape assertions
    assert world_pos.ndim == 3 and world_pos.shape[2] == 3, \
        f"world_pos must be (T, N, 3), got {world_pos.shape}"
    
    assert stress.ndim == 3 and stress.shape[2] == 1, \
        f"stress must be (T, N, 1), got {stress.shape}"
    
    assert mesh_pos.ndim == 2 and mesh_pos.shape[1] == 3, \
        f"mesh_pos must be (N, 3), got {mesh_pos.shape}"
    
    assert cells.ndim == 2 and cells.shape[1] == 4, \
        f"cells must be (C, 4), got {cells.shape}"
    
    # Validate that world_pos and stress have matching T and N
    T_wp, N_wp, _ = world_pos.shape
    T_st, N_st, _ = stress.shape
    assert T_wp == T_st, \
        f"world_pos and stress must have same T: {T_wp} != {T_st}"
    assert N_wp == N_st, \
        f"world_pos and stress must have same N: {N_wp} != {N_st}"
    
    # Validate that mesh_pos has same N as world_pos
    N_mp, _ = mesh_pos.shape
    assert N_mp == N_wp, \
        f"mesh_pos and world_pos must have same N: {N_mp} != {N_wp}"

    return {
        "world_pos": world_pos.astype(np.float32),  # (T, N, 3)
        "stress": stress.astype(np.float32),        # (T, N, 1) - TARGET, not input
        "node_type": node_type.astype(np.int32),    # (N, 1) or (N,)
        "mesh_pos": mesh_pos.astype(np.float32),    # (N, 3)
        "cells": cells.astype(np.int32),            # (C, 4) - for connectivity only
        "actuation": actuation.astype(np.float32),  # (T, N, 3) - always present
    }


def load_raw_trajectory_from_tfrecord(tfrecord_path, meta, traj_index):
    loader = tfrecord_loader(tfrecord_path, index_path=None)

    for i, record in enumerate(loader):
        if i == traj_index:
            return decode_trajectory_from_record(record, meta)

    raise IndexError(f"trajectory index {traj_index} out of range")


# ============================================================
# 2) COSTRUZIONE EDGES
# ============================================================

def build_edges_from_cells(cells, num_nodes=None):
    """
    MeshGraphNets (DeformingPlate) uses a tetrahedral mesh.
    Each cell has 4 vertices (a tetra), and the mesh edges are the 6 edges of the tetra.
    We add both directions to get a bidirectional edge list.

    Args:
        cells: iterable/torch.Tensor of shape (num_cells, 4) with vertex indices
        num_nodes: unused (kept for API compatibility)

    Returns:
        edge_index: torch.LongTensor of shape (#edges, 2)
    """
    edge_set = set()

    # All 6 edges of a tetra (complete graph on 4 vertices)
    tetra_edge_pairs = [(0, 1), (0, 2), (0, 3),
                        (1, 2), (1, 3),
                        (2, 3)]

    for c in cells:
        i0, i1, i2, i3 = map(int, c.tolist())
        verts = (i0, i1, i2, i3)

        for a, b in tetra_edge_pairs:
            u, v = verts[a], verts[b]
            if u != v:
                edge_set.add((u, v))
                edge_set.add((v, u))  # bidirectional

    edge_list = sorted(edge_set)
    return torch.tensor(edge_list, dtype=torch.long)


# ============================================================
# 3) ASSEMBLAGGIO INPUT EGNN
# ============================================================

def trajectory_to_egnn_inputs(traj):
    """
    Construct EGNN inputs from trajectory data.
    
    According to the deforming_plate dataset specification:
    - Input node features: [position(3), actuation(3), node_type_one_hot(2)] = 8 dims
    - Targets (separate): velocity (T, N, 3), stress (T, N, 1)
    - Coordinates: world_pos (T, N, 3) - used as node positions in graph
    
    Args:
        traj: Dictionary with keys:
            - world_pos: (T, N, 3) - 3D positions over time
            - stress: (T, N, 1) - stress values (TARGET, not input)
            - node_type: (N, 1) or (N,) - node type integer codes
            - cells: (C, 4) - tetrahedral cell connectivity
            - actuation: (T, N, 3) - actuation/boundary displacement (always present)
    
    Returns:
        coors_seq: (T, N, 3) - node coordinates (same as world_pos)
        feats_seq: (T, N, 8) - node features [pos(3), actuation(3), node_type_one_hot(2)]
        edge_index: (E, 2) - edge connectivity from cells
    """
    world_pos = traj["world_pos"]   # (T, N, 3) - positions over time
    stress = traj["stress"]         # (T, N, 1) - TARGET, not input
    node_type = traj["node_type"]   # (N, 1) or (N,) - node type codes
    cells = traj["cells"]           # (C, 4) - tetrahedral connectivity
    actuation = traj["actuation"]   # (T, N, 3) - always present (guaranteed by decode_trajectory_from_record)

    T, N, _ = world_pos.shape

    # Validate actuation shape (should already be validated in decode_trajectory_from_record)
    assert actuation.shape == (T, N, 3), \
        f"actuation shape {actuation.shape} does not match world_pos shape {world_pos.shape}"

    # Coordinates are the world positions
    coors_seq = torch.tensor(world_pos, dtype=torch.float32)  # (T, N, 3)

    # Construct 8D input features: [position(3), actuation(3), node_type_one_hot(2)]
    # Position: use current world position at each timestep
    # Actuation: boundary displacement / input signal (always present, guaranteed to be (T, N, 3))
    # Node type: convert to 2-class one-hot encoding

    # Convert node_type to one-hot encoding
    # According to meshgraphnets/common.py NodeType enum:
    # - NORMAL = 0 (plate nodes, where we compute loss)
    # - OBSTACLE = 1 (boundary nodes, no loss)
    # - HANDLE = 3 (actuator nodes, no loss)
    # Encoding: 0 -> [0, 0] (NORMAL/plate), 1 -> [1, 0] (OBSTACLE), 3 -> [0, 1] (HANDLE)
    # node_type == 0 (NORMAL) indicates plate nodes (where we compute loss)
    # node_type == 1 (OBSTACLE) or 3 (HANDLE) are boundary/actuator nodes (fixed, no loss)
    
    # Handle node_type shape: could be (N, 1) or (N,) after safe_squeeze_leading
    if node_type.ndim == 2 and node_type.shape[1] == 1:
        node_type_flat = node_type[:, 0]  # (N,)
    else:
        node_type_flat = node_type.flatten()  # (N,)
    
    # Validate node_type has correct number of nodes
    assert len(node_type_flat) == N, \
        f"node_type length {len(node_type_flat)} does not match number of nodes {N}"
    node_type_one_hot = np.zeros((N, 2), dtype=np.float32)
    node_type_one_hot[node_type_flat == 0, :] = [0.0, 0.0]  # Type 0 -> [0, 0]
    node_type_one_hot[node_type_flat == 1, :] = [1.0, 0.0]  # Type 1 (plate) -> [1, 0]
    node_type_one_hot[node_type_flat == 3, :] = [0.0, 1.0]  # Type 3 -> [0, 1]

    # Construct features for each timestep
    feats = []
    for t in range(T):
        # Concatenate: [position(3), actuation(3), node_type_one_hot(2)] = 8 dims
        feats_t = np.concatenate([
            world_pos[t],              # (N, 3) - current position
            actuation[t],               # (N, 3) - actuation/boundary displacement
            node_type_one_hot,         # (N, 2) - one-hot node type
        ], axis=-1)  # -> (N, 8)
        feats.append(feats_t)

    feats_seq = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)  # (T, N, 8)

    # Build edge connectivity from tetrahedral cells
    edge_index = build_edges_from_cells(cells, num_nodes=N)

    return coors_seq, feats_seq, edge_index


# ============================================================
# 4) WRAPPER FINALE
# ============================================================

def data_loader_egnn(tfrecord_path, meta_path, traj_index):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    traj_dict = load_raw_trajectory_from_tfrecord(tfrecord_path, meta, traj_index)
    return trajectory_to_egnn_inputs(traj_dict)


# ============================================================
# 5) TEST
# ============================================================

if __name__ == "__main__":
    tfrecord_path = "data/deforming_plate/train.tfrecord"
    meta_path = "data/deforming_plate/meta.json"

    coors, feats, edge_index = data_loader_egnn(
        tfrecord_path, meta_path, traj_index=0
    )

    print("coors_seq:", coors.shape)
    print("feats_seq:", feats.shape)
    print("edge_index:", edge_index.shape)