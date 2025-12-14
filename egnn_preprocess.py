import os, json, yaml, torch, numpy as np
from tfrecord.reader import tfrecord_loader
from tqdm import tqdm

FEATURE_DIM = 8  # Pos(3) + Act(3) + Type(2)
TARGET_DIM = 4   # Vel(3) + Stress(1)

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def decode(val, shape, dtype):
    # Handle bytes vs numpy types and reshape
    if isinstance(val, (bytes, bytearray)): arr = np.frombuffer(val, dtype=dtype)
    elif isinstance(val, np.ndarray) and val.dtype == object: arr = np.frombuffer(val.flat[0], dtype=dtype)
    else: arr = val.astype(dtype)
    
    # Resolve -1 dimension
    if -1 in shape:
        known = np.prod([s for s in shape if s != -1])
        shape = list(shape)
        shape[shape.index(-1)] = arr.size // known
    return arr.reshape(shape)

def build_edges(cells):
    # Create bidirectional edges from tetras
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    edges = set()
    for c in cells:
        for i, j in pairs:
            u, v = c[i], c[j]
            if u != v: edges.update([(u, v), (v, u)])
    return torch.tensor(sorted(list(edges)), dtype=torch.long).t()

def preprocess(cfg):
    d_cfg = cfg['data']
    raw_dir, out_dir = d_cfg['data_dir'], d_cfg['preprocessed_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(raw_dir, "meta.json"), "r") as f: 
        meta = json.load(f)
        
    loader = tfrecord_loader(os.path.join(raw_dir, f"{d_cfg['split']}.tfrecord"), None)
    
    sum_f = torch.zeros(FEATURE_DIM).double()
    sq_f = torch.zeros(FEATURE_DIM).double()
    sum_t = torch.zeros(TARGET_DIM).double()
    sq_t = torch.zeros(TARGET_DIM).double()
    count = 0
    index_list = []

    print(f"Processing max {d_cfg['max_trajs']} trajectories...")

    for idx, rec in tqdm(enumerate(loader)):
        if d_cfg['max_trajs'] and idx >= d_cfg['max_trajs']: break

        # Decode raw fields
        pos = decode(rec["world_pos"], meta["features"]["world_pos"]["shape"], np.float32)
        stress = decode(rec["stress"], meta["features"]["stress"]["shape"], np.float32)
        cells = decode(rec["cells"], meta["features"]["cells"]["shape"], np.int32)
        type_raw = decode(rec["node_type"], meta["features"]["node_type"]["shape"], np.int32).squeeze()
        
        # Handle optional actuation
        if "actuation" in rec: act = decode(rec["actuation"], meta["features"]["actuation"]["shape"], np.float32)
        else: act = np.zeros_like(pos)
        if act.shape[0] != pos.shape[0]: act = np.tile(act, (pos.shape[0], 1, 1))

        # Computed fields
        vel = np.zeros_like(pos); vel[1:] = pos[1:] - pos[:-1]
        edge_index = build_edges(cells)
        
        # One-hot node types (0->[0,0], 1->[1,0], 3->[0,1])
        nt = np.zeros((len(type_raw), 2), dtype=np.float32)
        nt[type_raw == 1] = [1, 0]; nt[type_raw == 3] = [0, 1]

        x_seq, y_seq = [], []
        
        for t in range(pos.shape[0] - 1):
            # Input: [Pos, Act, Type] | Target: [Vel, Stress]
            xt = np.concatenate([pos[t], act[t], nt], axis=-1)
            yt = np.concatenate([vel[t+1], stress[t+1]], axis=-1)
            
            x_seq.append(xt); y_seq.append(yt)
            
            # Stats accumulation
            xt_t, yt_t = torch.from_numpy(xt), torch.from_numpy(yt)
            sum_f += xt_t.sum(0); sq_f += (xt_t**2).sum(0)
            sum_t += yt_t.sum(0); sq_t += (yt_t**2).sum(0)
            index_list.append((idx, t))

        count += (pos.shape[0] - 1) * pos.shape[1]

        torch.save({
            "x": torch.tensor(np.stack(x_seq)).float(),
            "y": torch.tensor(np.stack(y_seq)).float(),
            "edge_index": edge_index,
            "node_type": torch.tensor(type_raw).long()
        }, os.path.join(out_dir, f"egnn_traj_{idx:05d}.pt"))

    # Stats calculation
    mean_f = sum_f / count
    std_f = torch.sqrt((sq_f / count) - mean_f**2).clamp(min=1e-8)
    mean_t = sum_t / count
    std_t = torch.sqrt((sq_t / count) - mean_t**2).clamp(min=1e-8)

    # Do not normalize node types (indices 6,7)
    mean_f[6:] = 0.0; std_f[6:] = 1.0

    torch.save({
        "mean_feat": mean_f.float(), "std_feat": std_f.float(),
        "mean_target": mean_t.float(), "std_target": std_t.float()
    }, os.path.join(out_dir, "egnn_stats.pt"))
    
    torch.save({"index": index_list}, os.path.join(out_dir, "egnn_index.pt"))

if __name__ == "__main__":
    preprocess(load_config("config.yaml"))