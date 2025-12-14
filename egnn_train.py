import os
import torch
import yaml
import numpy as np
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.cuda.amp import GradScaler, autocast

# Ensure these match your filenames
from egnn_data import EGNNTFRecordDataset 
from egnn_transform import OverwriteKinematicVelocity, AddDynamicWorldEdges
from model_egnn import MeshEGNN
from egnn_plots import make_final_plots

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def run_epoch(model, loader, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0
    total_v = 0
    total_s = 0
    
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        
        with torch.set_grad_enabled(is_train):
            # Autocast manages mixed precision (Float16)
            # We enable it on CUDA, or CPU if supported (usually just CUDA)
            use_amp = (device.type == 'cuda')
            
            with autocast(enabled=use_amp):
                # Input: Features (x) and Coords (pos)
                # Note: 'x' is normalized [Pos, Vel, Type, Stress] (9 dim)
                # 'pos' must be extracted from 'x' so it is also normalized
                pos = batch.x[:, 0:3]
                
                # Forward Pass
                # batch.batch is needed for scatter operations if batch_size > 1
                pred_vel, pred_stress = model(batch.x, pos, batch.edge_index, batch.batch)
                
                # Targets
                target_vel = batch.y[:, 0:3]
                target_stress = batch.y[:, 3:4]
                
                # Masking: Train Velocity on Plate(0), Stress on Plate(0)+Handle(3)
                # Note: Preprocessing MUST use: 0=[0,0], 1=[1,0], 3=[0,1]
                # If so: batch.node_type is the raw integer (0, 1, 3)
                mask_plate = (batch.node_type == 0)
                mask_stress = (batch.node_type == 0) | (batch.node_type == 3)
                
                # Loss Calculation
                loss_v = torch.tensor(0.0, device=device)
                if mask_plate.sum() > 0:
                    loss_v = torch.mean((pred_vel[mask_plate] - target_vel[mask_plate])**2)

                loss_s = torch.tensor(0.0, device=device)
                if mask_stress.sum() > 0:
                    loss_s = torch.mean((pred_stress[mask_stress] - target_stress[mask_stress])**2)
                    
                loss = loss_v + loss_s

        if is_train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            # Store for plotting (Only Plate nodes to avoid noise)
            p_cat = torch.cat([pred_vel, pred_stress], dim=1)
            t_cat = torch.cat([target_vel, target_stress], dim=1)
            if mask_plate.sum() > 0:
                all_preds.append(p_cat[mask_plate].detach().cpu())
                all_targets.append(t_cat[mask_plate].detach().cpu())

        total_loss += loss.item()
        total_v += loss_v.item()
        total_s += loss_s.item()

    num_batches = max(len(loader), 1)
    avg_loss = total_loss / num_batches
    avg_v = total_v / num_batches
    avg_s = total_s / num_batches
    
    if not is_train and len(all_preds) > 0:
        return avg_loss, avg_v, avg_s, torch.cat(all_preds), torch.cat(all_targets)
    return avg_loss, avg_v, avg_s, None, None

def train(cfg):
    # Device Setup
    if cfg['training']['device'] == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(cfg['training']['device'])
    
    print(f"Using device: {device}")
    torch.manual_seed(cfg['training']['seed'])
    
    # 1. Transforms & Data
    # 'OverwriteKinematicVelocity' allows the model to see the sphere's future speed
    # 'AddDynamicWorldEdges' allows the model to see collisions
    transform = Compose([
        OverwriteKinematicVelocity(),
        AddDynamicWorldEdges(radius=cfg['data']['radius'])
    ])

    mode = cfg['training'].get('mode', 'standard')
    
    if mode == "overfit":
        print(f"!!! OVERFIT MODE ON !!!")
        print(f"Training on Traj {cfg['training']['overfit_traj_ids']}")
        
        dataset = EGNNTFRecordDataset(
            data_dir=cfg['data']['data_dir'],
            preprocessed_dir=cfg['data']['preprocessed_dir'],
            split=cfg['data']['split'],
            transform=transform,
            allowed_traj_ids=cfg['training']['overfit_traj_ids'],
            allowed_time_ids=cfg['training']['overfit_time_ids'],
            cache_all_in_ram=True
        )
        # Same data for train/val to verify memorization
        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        val_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        full_dataset = dataset
        
    else:
        # Standard Training
        full_dataset = EGNNTFRecordDataset(
            data_dir=cfg['data']['data_dir'],
            preprocessed_dir=cfg['data']['preprocessed_dir'],
            split=cfg['data']['split'],
            transform=transform,
            cache_all_in_ram=True
        )
        # 80/20 Split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)

    # 2. Model
    # Input is 9 (Pos+Vel+Type+Stress). Model handles stripping internally.
    model = MeshEGNN(
        in_dim=cfg['model']['in_node_nf'],
        hidden_dim=cfg['model']['hidden_nf'],
        depth=cfg['model']['n_layers']
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'])
    scaler = GradScaler() if device.type == 'cuda' else None
    
    if mode == "overfit":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.001, epochs=cfg['training']['epochs'], 
            steps_per_epoch=len(train_loader)
        )

    # 3. Training Loop
    stats = {'t_loss': [], 'v_loss': [], 't_vel': [], 'v_vel': [], 't_str': [], 'v_str': []}
    print(f"Start Training: {len(train_loader.dataset)} samples")

    for epoch in range(cfg['training']['epochs']):
        t_loss, t_v, t_s, _, _ = run_epoch(model, train_loader, optimizer, scaler, device, True)
        scheduler.step()
        
        v_loss, v_v, v_s, val_preds, val_targets = run_epoch(model, val_loader, None, None, device, False)
        
        # Logging
        stats['t_loss'].append(t_loss); stats['v_loss'].append(v_loss)
        stats['t_vel'].append(t_v);     stats['v_vel'].append(v_v)
        stats['t_str'].append(t_s);     stats['v_str'].append(v_s)
        
        print(f"Epoch {epoch+1}: Train {t_loss:.6f} | Val {v_loss:.6f}")
        
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), "egnn_checkpoint.pt")

    # 4. Final Plots & Save
    print("Generating Plots...")
    torch.save(model.state_dict(), "egnn_final.pt")
    
    # Denormalize & Plot
    if val_preds is not None:
        mean_t = full_dataset.mean_target.to('cpu')
        std_t = full_dataset.std_target.to('cpu')
        
        final_preds = val_preds * std_t + mean_t
        final_targs = val_targets * std_t + mean_t
        
        make_final_plots(
            save_dir="plots",
            train_losses=stats['t_loss'], val_losses=stats['v_loss'],
            train_vel_losses=stats['t_vel'], val_vel_losses=stats['v_vel'],     # <--- FIXED KEYS
            train_stress_losses=stats['t_str'], val_stress_losses=stats['v_str'], # <--- FIXED KEYS
            predictions=final_preds.numpy(),
            targets=final_targs.numpy()
        )

if __name__ == "__main__":
    train(load_config("config.yaml"))