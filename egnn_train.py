import os
# Fix OpenMP conflict on macOS (must be set before importing torch/numpy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import yaml
import numpy as np
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.cuda.amp import GradScaler, autocast

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
            # Use autocast only on CUDA, otherwise regular computation
            if device.type == 'cuda':
                with autocast():
                    pos = batch.x[:, 0:3]
                    batch_attr = getattr(batch, 'batch', None)
                    pred_vel, pred_stress = model(batch.x, pos, batch.edge_index, batch=batch_attr)
                    
                    target_vel = batch.y[:, 0:3]
                    target_stress = batch.y[:, 3:4]
                    
                    # Mask: 0=Plate, 3=Handle
                    mask_plate = (batch.node_type == 0)
                    mask_stress = (batch.node_type == 0) | (batch.node_type == 3)
                    
                    # Safety check for empty masks (rare edge case)
                    if mask_plate.sum() > 0:
                        loss_v = torch.mean((pred_vel[mask_plate] - target_vel[mask_plate])**2)
                    else:
                        loss_v = torch.tensor(0.0, device=device)

                    if mask_stress.sum() > 0:
                        loss_s = torch.mean((pred_stress[mask_stress] - target_stress[mask_stress])**2)
                    else:
                        loss_s = torch.tensor(0.0, device=device)
                        
                    loss = loss_v + loss_s
            else:
                # CPU path: no autocast
                pos = batch.x[:, 0:3]
                batch_attr = getattr(batch, 'batch', None)
                pred_vel, pred_stress = model(batch.x, pos, batch.edge_index, batch=batch_attr)
                
                target_vel = batch.y[:, 0:3]
                target_stress = batch.y[:, 3:4]
                
                # Mask: 0=Plate, 3=Handle
                mask_plate = (batch.node_type == 0)
                mask_stress = (batch.node_type == 0) | (batch.node_type == 3)
                
                # Safety check for empty masks (rare edge case)
                if mask_plate.sum() > 0:
                    loss_v = torch.mean((pred_vel[mask_plate] - target_vel[mask_plate])**2)
                else:
                    loss_v = torch.tensor(0.0, device=device)

                if mask_stress.sum() > 0:
                    loss_s = torch.mean((pred_stress[mask_stress] - target_stress[mask_stress])**2)
                else:
                    loss_s = torch.tensor(0.0, device=device)
                    
                loss = loss_v + loss_s

        if is_train:
            optimizer.zero_grad()
            if device.type == 'cuda' and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
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
    # Auto-detect device: use CUDA if available and requested, otherwise CPU
    requested_device = cfg['training']['device'].lower()
    if requested_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif requested_device == 'cuda':
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(requested_device)
    
    print(f"Using device: {device}")
    torch.manual_seed(cfg['training']['seed'])
    
    # 1. DATA SETUP
    transform = Compose([
        OverwriteKinematicVelocity(),
        AddDynamicWorldEdges(radius=cfg['data']['radius'])
    ])

    mode = cfg['training'].get('mode', 'standard')
    
    if mode == "overfit":
        print(f"!!! OVERFIT MODE ON !!!")
        print(f"Training on Traj {cfg['training']['overfit_traj_ids']} steps {cfg['training']['overfit_time_ids']}")
        
        # Load ONLY the specific data
        dataset = EGNNTFRecordDataset(
            data_dir=cfg['data']['data_dir'],
            preprocessed_dir=cfg['data']['preprocessed_dir'],
            split=cfg['data']['split'],
            transform=transform,
            allowed_traj_ids=cfg['training']['overfit_traj_ids'],
            allowed_time_ids=cfg['training']['overfit_time_ids'], # Only these times
            cache_all_in_ram=True
        )
        
        # Same loader for Train and Val
        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        val_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        
        # Use full dataset for stats (hack for denormalization)
        full_dataset = dataset 
        
    else:
        # STANDARD MODE
        full_dataset = EGNNTFRecordDataset(
            data_dir=cfg['data']['data_dir'],
            preprocessed_dir=cfg['data']['preprocessed_dir'],
            split=cfg['data']['split'],
            transform=transform,
            cache_all_in_ram=True
        )
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)

    # 2. MODEL
    model = MeshEGNN(
        in_dim=cfg['model']['in_node_nf'],
        hidden_dim=cfg['model']['hidden_nf'],
        depth=cfg['model']['n_layers']
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'])
    # Only use GradScaler on CUDA
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Scheduler logic (different for overfit)
    if mode == "overfit":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.001, epochs=cfg['training']['epochs'], 
            steps_per_epoch=len(train_loader)
        )

    stats = {'t_loss': [], 'v_loss': [], 't_vel': [], 'v_vel': [], 't_str': [], 'v_str': []}

    print(f"Start Training: {len(train_loader.dataset)} Train samples")

    for epoch in range(cfg['training']['epochs']):
        t_loss, t_v, t_s, _, _ = run_epoch(model, train_loader, optimizer, scaler, device, True)
        scheduler.step()
        
        v_loss, v_v, v_s, val_preds, val_targets = run_epoch(model, val_loader, None, None, device, False)
        
        stats['t_loss'].append(t_loss); stats['v_loss'].append(v_loss)
        stats['t_vel'].append(t_v);     stats['v_vel'].append(v_v)
        stats['t_str'].append(t_s);     stats['v_str'].append(v_s)
        
        print(f"Epoch {epoch+1}: Train {t_loss:.5f} | Val {v_loss:.5f}")

    # 3. PLOTTING
    print("Generating Plots...")
    mean_t = full_dataset.mean_target
    std_t = full_dataset.std_target
    
    if val_preds is not None:
        final_preds = val_preds * std_t + mean_t
        final_targs = val_targets * std_t + mean_t
        
        make_final_plots(
            save_dir="plots",
            train_losses=stats['t_loss'], val_losses=stats['v_loss'],
            train_vel=stats['t_vel'], val_vel=stats['v_vel'],
            train_stress=stats['t_str'], val_stress=stats['v_str'],
            predictions=final_preds.numpy(),
            targets=final_targs.numpy()
        )
    
    torch.save(model.state_dict(), "egnn_final.pt")

if __name__ == "__main__":
    train(load_config("config.yaml"))