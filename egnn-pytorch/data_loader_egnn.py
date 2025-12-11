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
    Decodifica un record da tfrecord_loader gestendo:
    - bytes
    - np.ndarray numerici
    - np.ndarray(object) contenenti bytes

    Usa np.frombuffer + reshape usando le shape nel meta.json.
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
        value può essere:
        - np.ndarray numerico → restituito (cast + reshape)
        - bytes / np.ndarray(object) → np.frombuffer + reshape
        """

        # Caso 1: già ndarray numerico
        if isinstance(value, np.ndarray) and value.dtype != object:
            arr = value.astype(dtype)
            # costruiamo la shape corretta
            tgt = list(shape_spec)
            if -1 in tgt:
                known = 1
                for d in tgt:
                    if d != -1:
                        known *= d
                missing = arr.size // known
                tgt[tgt.index(-1)] = missing
            if tuple(arr.shape) != tuple(tgt):
                arr = arr.reshape(tgt)
            return arr

        # Caso 2: è bytes o array di object → frombuffer
        raw = normalize_to_bytes(value)
        arr = np.frombuffer(raw, dtype=dtype)

        tgt = list(shape_spec)
        if -1 in tgt:
            known = 1
            for d in tgt:
                if d != -1:
                    known *= d
            missing = arr.size // known
            tgt[tgt.index(-1)] = missing

        return arr.reshape(tgt)

    shapes = meta["features"]

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

    # Rimuovi la prima dimensione =1 per statiche
    if node_type.shape[0] == 1:
        node_type = node_type[0]
    if mesh_pos.shape[0] == 1:
        mesh_pos = mesh_pos[0]
    if cells.shape[0] == 1:
        cells = cells[0]

    # Try to load actuation if available, otherwise set to zeros
    # Actuation represents boundary displacement / input signal per node
    actuation = None
    if "actuation" in record and "actuation" in shapes:
        actuation = decode_raw_array(record["actuation"], np.float32,
                                     shapes["actuation"]["shape"])
        actuation = actuation.astype(np.float32)
    elif "boundary_displacement" in record and "boundary_displacement" in shapes:
        actuation = decode_raw_array(record["boundary_displacement"], np.float32,
                                     shapes["boundary_displacement"]["shape"])
        actuation = actuation.astype(np.float32)

    return {
        "world_pos": world_pos.astype(np.float32),  # (T, N, 3)
        "stress": stress.astype(np.float32),        # (T, N, 1) - TARGET, not input
        "node_type": node_type.astype(np.int32),    # (N, 1)
        "mesh_pos": mesh_pos.astype(np.float32),    # (N, 3)
        "cells": cells.astype(np.int32),            # (C, 4) - for connectivity only
        "actuation": actuation,  # (T, N, 3) or None - input feature
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

def build_edges_from_cells(cells, num_nodes):
    edge_set = set()

    for c in cells:
        i0, i1, i2, i3 = map(int, c.tolist())
        quad = [i0, i1, i2, i3]

        for u, v in zip(quad, quad[1:] + quad[:1]):
            if u != v:
                edge_set.add((u, v))
                edge_set.add((v, u))

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
            - node_type: (N, 1) - node type integer codes
            - cells: (C, 4) - tetrahedral cell connectivity
            - actuation: (T, N, 3) or None - actuation/boundary displacement
    
    Returns:
        coors_seq: (T, N, 3) - node coordinates (same as world_pos)
        feats_seq: (T, N, 8) - node features [pos(3), actuation(3), node_type_one_hot(2)]
        edge_index: (E, 2) - edge connectivity from cells
    """
    world_pos = traj["world_pos"]   # (T, N, 3) - positions over time
    stress = traj["stress"]         # (T, N, 1) - TARGET, not input
    node_type = traj["node_type"]   # (N, 1) - node type codes
    cells = traj["cells"]           # (C, 4) - tetrahedral connectivity
    actuation = traj.get("actuation", None)  # (T, N, 3) or None

    T, N, _ = world_pos.shape

    # Coordinates are the world positions
    coors_seq = torch.tensor(world_pos, dtype=torch.float32)  # (T, N, 3)

    # Construct 8D input features: [position(3), actuation(3), node_type_one_hot(2)]
    # Position: use current world position at each timestep
    # Actuation: boundary displacement / input signal (if available, else zeros)
    # Node type: convert to 2-class one-hot encoding
    
    # Handle actuation: if not available, set to zeros
    if actuation is None:
        # Actuation not in TFRecord - set to zeros
        # TODO: If actuation field exists in TFRecord, load it here
        actuation = np.zeros((T, N, 3), dtype=np.float32)
    else:
        # Ensure actuation has correct shape
        if actuation.shape != (T, N, 3):
            # If actuation is static (N, 3), tile it over time
            if len(actuation.shape) == 2 and actuation.shape == (N, 3):
                actuation = np.tile(actuation[np.newaxis, :, :], (T, 1, 1))
            else:
                raise ValueError(f"Unexpected actuation shape: {actuation.shape}, expected (T, N, 3) or (N, 3)")

    # Convert node_type to one-hot encoding
    # According to meshgraphnets/common.py NodeType enum:
    # - NORMAL = 0 (plate nodes, where we compute loss)
    # - OBSTACLE = 1 (boundary nodes, no loss)
    # - HANDLE = 3 (actuator nodes, no loss)
    # Encoding: 0 -> [0, 0] (NORMAL/plate), 1 -> [1, 0] (OBSTACLE), 3 -> [0, 1] (HANDLE)
    # node_type == 0 (NORMAL) indicates plate nodes (where we compute loss)
    # node_type == 1 (OBSTACLE) or 3 (HANDLE) are boundary/actuator nodes (fixed, no loss)
    node_type_flat = node_type.flatten()  # (N,)
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