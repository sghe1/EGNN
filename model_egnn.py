import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network

class MeshEGNN(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=128, depth=4):
        super().__init__()
        
        # 1. Input Projection (9 -> 128)
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        
        # 2. EGNN Backbone
        self.egnn = EGNN_Network(
            num_tokens=None,
            dim=hidden_dim,
            depth=depth,
            edge_dim=0,
            only_sparse_neighbors=True,
            update_coors=False, # We predict velocity, not update pos directly inside layers
            update_feats=True
        )
        
        # 3. Velocity Head (Hidden -> 3)
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # 4. Stress Head (Hidden -> 1)
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, pos, edge_index, batch=None):
        """
        x: [B*N, 9] Features
        pos: [B*N, 3] Coordinates (extracted from x[:, 0:3])
        edge_index: [2, E]
        """
        # Embed
        h = self.input_mlp(x)
        
        # Message Passing
        # Note: EGNN_Network.forward signature: (feats, coors, mask, adj_mat, edge_index)
        # We pass edge_index as kwarg
        h_out, _ = self.egnn(h, pos, edge_index=edge_index, batch=batch)
        
        # Predict
        vel = self.vel_head(h_out)
        stress = self.stress_head(h_out)
        
        return vel, stress