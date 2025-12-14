import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network
from torch_scatter import scatter_add

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MeshEGNN(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=128, depth=4, num_nodes_avg=1271):
        super().__init__()
        
        # 1. Input Projection
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU()
        )
        
        # 2. EGNN Backbone (Message Passing)
        # Updates embeddings 'h' based on geometry
        self.egnn_layers = EGNN_Network(
            num_tokens=None,
            dim=hidden_dim,
            depth=depth,
            edge_dim=0,
            only_sparse_neighbors=True,
            update_coors=False, 
            update_feats=True,
        )
        
        # 3. Physics Heads (Equation 7 implementation)
        # Direct Velocity Term
        self.phi_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, 3) 
        )
        
        # Coordinate Update Weights (Neighbor Term)
        # φ_e: Calculates m_ij 
        edge_input_dim = 2 * hidden_dim + 1
        self.phi_e = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2), SiLU(),
            nn.Linear(edge_input_dim * 2, edge_input_dim * 2), SiLU(),
            nn.Linear(edge_input_dim * 2, hidden_dim), SiLU()
        )

        # φ_x: m_ij -> scalar weight
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # C: Learnable global scaling factor (initialized small)
        C_init = 1.0 / (num_nodes_avg - 1) if num_nodes_avg > 1 else 1e-3
        self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))
        
        # 4. Stress Head
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Heuristic: Init stress bias negative so softplus/relu starts near 0
                    if m == self.stress_head[-1]:
                        nn.init.constant_(m.bias, -0.8)
                    else:
                        nn.init.zeros_(m.bias)

    def forward(self, x, pos, edge_index, batch=None):
        """
        x: [B*N, 9] Features [Pos(3), Vel(3), IsSphere(1), IsHandle(1), Stress(1)]
        pos: [B*N, 3] Coordinates
        edge_index: [2, E]
        """
        row, col = edge_index
        
        # A. Encode
        h = self.input_mlp(x) 
        
        # B. Message Passing
        h_updated, _ = self.egnn_layers(h, pos, edge_index=edge_index)
        
        # C. Velocity Prediction (Equation 7)
        # Term 1: Direct from features
        direct_vel = self.phi_v(h_updated)
        
        # Term 2: Neighbor Interaction C * sum( (xi - xj) * phi_x(mij) )
        pos_i, pos_j = pos[row], pos[col]
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        
        h_i, h_j = h_updated[row], h_updated[col]
        edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        
        # Compute weights
        m_ij = self.phi_e(edge_input)
        weights = self.phi_x(m_ij) 
        
        # Scatter Sum
        weighted_diff = rel_pos * weights
        neighbor_term = scatter_add(weighted_diff, row, dim=0, dim_size=x.size(0))
        
        pred_vel = direct_vel + self.C * neighbor_term
        
        # D. Stress Prediction
        pred_stress = self.stress_head(h_updated)
        
        # --- E. MASKING (THE FIX) ---
        # Indices in Preprocess: 
        # 6: IsSphere (Obstacle) -> [1, 0]
        # 7: IsHandle (Boundary) -> [0, 1]
        
        is_obstacle = x[:, 6] > 0.5
        is_handle   = x[:, 7] > 0.5
        
        # 1. Velocity Mask: Zero out Obstacles AND Handles
        # (Only Normal nodes move freely)
        vel_mask = (~is_obstacle) & (~is_handle)
        vel_mask = vel_mask.float().unsqueeze(-1)
        
        # 2. Stress Mask: Zero out Obstacles ONLY
        # (Handles experience stress, so we MUST predict it there)
        stress_mask = (~is_obstacle)
        stress_mask = stress_mask.float().unsqueeze(-1)
        
        return pred_vel * vel_mask, pred_stress * stress_mask