import torch
import yaml
import os
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.cuda.amp import GradScaler, autocast

# Reuse optimized modules
from pyg_data import GraphUNetTFRecordDataset 
from egnn_transforms import OverwriteKinematicVelocity, AddDynamicWorldEdges
from egnn_model import MeshEGNN

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def train(cfg):
    device = torch.device(cfg['training']['device'])
    torch.manual_seed(cfg['training']['seed'])
    
    # 1. Transforms (The "Clairvoyant" Logic)
    transform = Compose([
        OverwriteKinematicVelocity(), # Swap v_t for v_t+1 on Sphere
        AddDynamicWorldEdges(radius=cfg['data']['radius'])
    ])

    # 2. Dataset
    dataset = GraphUNetTFRecordDataset(
        data_dir=cfg['data']['data_dir'],
        preprocessed_dir=cfg['data']['preprocessed_dir'],
        split=cfg['data']['split'],
        transform=transform,
        cache_all_in_ram=True
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=0
    )

    # 3. Model
    model = MeshEGNN(
        in_dim=cfg['model']['in_node_nf'],
        hidden_dim=cfg['model']['hidden_nf'],
        depth=cfg['model']['n_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    scaler = GradScaler()

    # 4. Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with autocast():
                # Input Slice: Pos is 0:3
                pos = batch.x[:, 0:3]
                
                # Forward
                pred_vel, pred_stress = model(batch.x, pos, batch.edge_index, batch.batch)
                
                # Targets
                target_vel = batch.y[:, 0:3]
                target_stress = batch.y[:, 3:4]
                
                # Masking (Train only on Plate/Normal nodes)
                mask = (batch.node_type == 0) # 0=Plate
                
                loss_v = torch.mean((pred_vel[mask] - target_vel[mask])**2)
                loss_s = torch.mean((pred_stress[mask] - target_stress[mask])**2)
                loss = loss_v + loss_s

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.6f}")
        
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), "egnn_checkpoint.pt")

if __name__ == "__main__":
    train(load_config("config.yaml"))