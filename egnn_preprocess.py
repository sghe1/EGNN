import os, json, yaml, torch, numpy as np
from tfrecord.reader import tfrecord_loader
from tqdm import tqdm

# Layout: [Pos(3), Vel(3), Type(2), Stress(1)] = 9
FEATURE_DIM = 9  
TARGET_DIM = 4   

def decode(val, shape, dtype):
    if isinstance(val, (bytes, bytearray)): arr = np.frombuffer(val, dtype=dtype)
    elif isinstance(val, np.ndarray) and val.dtype == object: arr = np.frombuffer(val.flat[0], dtype=dtype)
    else: arr = val.astype(dtype)
    if -1 in shape:
        shape = list(shape); shape[shape.index(-1)] = arr.size // np.prod([s for s in shape if s != -1])
    return arr.reshape(shape)

def build_static_edges(cells):
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    edges = set()
    # Ensure cells is a numpy array and iterate properly
    cells = np.asarray(cells)
    for c in cells:
        # Ensure c is 1D and extract indices
        c_flat = np.asarray(c).flatten()
        for i, j in pairs:
            # Extract scalar values safely
            u = int(c_flat[i])
            v = int(c_flat[j])
            if u != v: 
                edges.update([(u, v), (v, u)])
    return torch.tensor(sorted(list(edges)), dtype=torch.long).t()

def preprocess(cfg):
    d_cfg = cfg['data']
    raw, out = d_cfg['data_dir'], d_cfg['preprocessed_dir']
    
    # Resolve paths relative to script location, not current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(raw):
        raw = os.path.normpath(os.path.join(script_dir, raw))
    if not os.path.isabs(out):
        out = os.path.normpath(os.path.join(script_dir, out))
    
    os.makedirs(out, exist_ok=True)
    
    meta_path = os.path.join(raw, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.json not found at: {meta_path}\n"
            f"Config data_dir: {d_cfg['data_dir']}\n"
            f"Script directory: {script_dir}\n"
            f"Resolved path: {raw}"
        )
    
    with open(meta_path) as f: meta = json.load(f)
    
    tfrecord_path = os.path.join(raw, f"{d_cfg['split']}.tfrecord")
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(
            f"TFRecord file not found at: {tfrecord_path}\n"
            f"Config split: {d_cfg['split']}\n"
            f"Resolved data_dir: {raw}"
        )
    
    loader = tfrecord_loader(tfrecord_path, None)
    
    # Stats Accumulators
    sum_f, sq_f = torch.zeros(FEATURE_DIM).double(), torch.zeros(FEATURE_DIM).double()
    sum_t, sq_t = torch.zeros(TARGET_DIM).double(), torch.zeros(TARGET_DIM).double()
    count = 0
    index_list = []

    print(f"Preprocessing {d_cfg['max_trajs'] or 'ALL'} trajectories...")

    for idx, rec in tqdm(enumerate(loader)):
        if d_cfg['max_trajs'] and idx >= d_cfg['max_trajs']: break

        pos = decode(rec["world_pos"], meta["features"]["world_pos"]["shape"], np.float32)
        stress = decode(rec["stress"], meta["features"]["stress"]["shape"], np.float32)
        cells = decode(rec["cells"], meta["features"]["cells"]["shape"], np.int32)
        type_raw = decode(rec["node_type"], meta["features"]["node_type"]["shape"], np.int32).squeeze()

        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        edge_index = build_static_edges(cells)
        
        # One-Hot Type: Sphere(1)->[1,0], Other->[0,1]
        nt = np.zeros((len(type_raw), 2), dtype=np.float32)
        nt[type_raw == 1] = [1, 0] 
        nt[type_raw != 1] = [0, 1] 

        x_seq, y_seq = [], []
        
        for t in range(pos.shape[0] - 1):
            xt = np.concatenate([pos[t], vel[t], nt, stress[t]], axis=-1)
            yt = np.concatenate([vel[t+1], stress[t+1]], axis=-1)
            x_seq.append(xt); y_seq.append(yt)
            
            # Stats
            xt_t, yt_t = torch.from_numpy(xt), torch.from_numpy(yt)
            sum_f += xt_t.sum(0); sq_f += (xt_t**2).sum(0)
            sum_t += yt_t.sum(0); sq_t += (yt_t**2).sum(0)
            index_list.append((idx, t))

        count += (pos.shape[0]-1) * pos.shape[1]
        
        torch.save({
            "x": torch.tensor(np.stack(x_seq)).float(),
            "y": torch.tensor(np.stack(y_seq)).float(),
            "edge_index": edge_index,
            "node_type": torch.tensor(type_raw).long()
        }, os.path.join(out, f"egnn_traj_{idx:05d}.pt"))

    # --- ISOTROPIC NORMALIZATION LOGIC ---
    # 1. Compute raw variance per channel
    mean_f = sum_f / count
    var_f = (sq_f / count) - mean_f**2
    mean_t = sum_t / count
    var_t = (sq_t / count) - mean_t**2

    # 2. Average variance across spatial dimensions (X, Y, Z)
    # Indices: Pos[0:3], Vel[3:6]
    pos_std_iso = torch.sqrt(var_f[0:3].mean())
    vel_std_iso = torch.sqrt(var_f[3:6].mean())
    targ_vel_std_iso = torch.sqrt(var_t[0:3].mean())

    # 3. Create Final Std Vectors
    std_f = torch.sqrt(var_f)
    std_f[0:3] = pos_std_iso  # Force Pos X,Y,Z to use same scale
    std_f[3:6] = vel_std_iso  # Force Vel X,Y,Z to use same scale
    
    std_t = torch.sqrt(var_t)
    std_t[0:3] = targ_vel_std_iso

    # 4. Don't normalize Type(6,7) or mask it? 
    # Usually we leave One-Hot as is (Mean=0, Std=1 prevents div by zero)
    mean_f[6:8] = 0.0; std_f[6:8] = 1.0

    # Clamp to avoid numerical issues
    std_f = std_f.clamp(min=1e-8)
    std_t = std_t.clamp(min=1e-8)

    torch.save({
        "mean_feat": mean_f.float(), "std_feat": std_f.float(),
        "mean_target": mean_t.float(), "std_target": std_t.float()
    }, os.path.join(out, "egnn_stats.pt"))
    
    torch.save({"index": index_list}, os.path.join(out, "egnn_index.pt"))

if __name__ == "__main__":
    with open("config.yaml", "r") as f: cfg = yaml.safe_load(f)
    preprocess(cfg)