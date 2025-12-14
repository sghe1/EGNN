import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EGNN_DefPlate(nn.Module):
    def __init__(self, in_dim, vel_out_dim, stress_out_dim, model_cfg):
        super().__init__()
        hidden_dim = model_cfg['hid_gnn_layer_dim']
        depth = model_cfg['depth']
        
        # Heuristic: If in_dim > 12, coords are likely at [3:6] (MeshPos included)
        # Else at [0:3]
        self.pos_slice = slice(3, 6) if in_dim > 12 else slice(0, 3)
        feat_dim = in_dim - 3
        
        self.input_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.egnn_layers = EGNN_Network(
            num_tokens=None,
            dim=hidden_dim,
            depth=depth,
            edge_dim=0,
            only_sparse_neighbors=False, 
            update_coors=False,
            update_feats=True
        )

        # Physics Heads
        self.phi_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, vel_out_dim)
        )
        
        edge_input_dim = 2 * hidden_dim + 1
        self.phi_e_proj = nn.Sequential(nn.Linear(edge_input_dim, hidden_dim), SiLU())
        self.phi_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), SiLU(), nn.Linear(hidden_dim, 1))
        
        self.C = nn.Parameter(torch.tensor(0.001))
        
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, stress_out_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m == self.stress_head[-1]: nn.init.constant_(m.bias, -0.8)
                    else: nn.init.zeros_(m.bias)

    def forward(self, batch_adj_A, batch_feat_X, feat_tp1_mat_list, node_types):
        return [self.embed_one(A, X) for A, X in zip(batch_adj_A, batch_feat_X)]

    def embed_one(self, adj, x):
        pos = x[:, self.pos_slice]
        # Concatenate features excluding coords
        h_in = torch.cat([x[:, :self.pos_slice.start], x[:, self.pos_slice.stop:]], dim=1)
        
        # Batch for library
        h_b, pos_b, adj_b = h_in.unsqueeze(0), pos.unsqueeze(0), adj.unsqueeze(0)
        
        h_emb = self.input_mlp(h_b)
        h_updated, _ = self.egnn_layers(h_emb, pos_b, adj_mat=adj_b)
        
        # Physics Update
        v_direct = self.phi_v(h_updated)
        
        # Recompute edges for Eq 7
        rel_pos = pos_b.unsqueeze(2) - pos_b.unsqueeze(1)
        dist_sq = (rel_pos**2).sum(-1, keepdim=True)
        h_exp = h_updated.unsqueeze(2).expand(-1, -1, h_updated.shape[1], -1)
        edge_feat = torch.cat([h_exp, h_exp.transpose(1,2), dist_sq], dim=-1)
        
        weights = self.phi_x(self.phi_e_proj(edge_feat))
        weights = weights * (adj_b.unsqueeze(-1) > 0).float()
        
        neighbor_term = (rel_pos * weights).sum(2)
        pred_vel = v_direct + self.C * neighbor_term
        pred_stress = self.stress_head(h_updated)
        
        return torch.cat([pred_vel, pred_stress], dim=-1).squeeze(0)