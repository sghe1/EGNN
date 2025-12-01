import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network

class MeshEGNN(nn.Module):
    """
    Mesh EGNN following the paper formulas:
    - v_i^{l+1} = phi_v(h_i^l) * v_i^init + C * sum_j (x_i^l - x_j^l) * phi_x(m_ij)
    - x_i^{l+1} = x_i^l + v_i^{l+1}
    """
    def __init__(self, in_dim, hidden_dim, out_dim=None, edge_dim=0, depth=4, C=1.0):
        super().__init__()
        self.input_mlp  = nn.Linear(in_dim, hidden_dim)
        self.egnn       = EGNN_Network(
            num_tokens=None,
            dim=hidden_dim,
            depth=depth,
            edge_dim=edge_dim,
            only_sparse_neighbors=True,
        )
        
        # phi_v: MLP that takes h and outputs a 1D vector (scalar) to modulate initial velocity
        self.phi_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # φ_e: edge/message MLP, builds m_ij from (h_i, h_j, ||x_i - x_j||^2, a_ij)
        # Here we have no explicit edge attributes a_ij, so we omit them.
        self.phi_e = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # φ_x: takes m_ij and outputs a scalar weight (Eq. (7))
        # Add tanh to constrain output to [-1, 1] to prevent explosion
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Constrain to [-1, 1] to prevent large neighbor terms
        )
        
        # Constant C (can be learnable) - initialize to small value to match velocity scale
        # Velocities are ~1e-4, so C should be small (e.g., 1e-4 or 1e-5)
        self.C = nn.Parameter(torch.tensor(C * 1e-4, dtype=torch.float32))  # Scale down C
        
        # Stress head (unchanged)
        self.stress_head = nn.Linear(hidden_dim, 1)
        
        # Extract initial velocity from features: feats = [node_type(1), vel(3), acc(3), stress(1)]
        self.vel_init_start_idx = 1  # After node_type
        self.vel_init_end_idx = 4    # vel has 3 dims

    def forward(self, feats, coors, adj_mat, edges=None):
        """
        Args:
            feats: (B, N, in_dim) - features including [node_type(1), vel(3), acc(3), stress(1)]
            coors: (B, N, 3) - current node coordinates
            adj_mat: (B, N, N) or (N, N) - adjacency matrix
        Returns:
            pred_vel: (B, N, 3) - predicted velocity
            pred_stress: (B, N, 1) - predicted stress
            pred_coors: (B, N, 3) - predicted coordinates (coors + pred_vel)
        """
        B, N, _ = coors.shape
        
        # Extract initial velocity from features
        v_init = feats[:, :, self.vel_init_start_idx:self.vel_init_end_idx]  # (B, N, 3)
        
        # Project features to hidden dimension and pass through EGNN
        # EGNN internally handles: m_i = sum_{j != i} m_ij and h_i^{l+1} = phi_h(h_i^l, m_i)
        h = self.input_mlp(feats)  # (B, N, hidden_dim)
        h, _ = self.egnn(h, coors, adj_mat=adj_mat, edges=edges, mask=None)
        
        # Compute messages m_ij for all pairs (we'll use only neighbors via adj_mat)
        # For each node i, compute interaction with all neighbors j
        # Expand h for pairwise computation
        h_i = h.unsqueeze(2)  # (B, N, 1, hidden_dim)
        h_j = h.unsqueeze(1)  # (B, 1, N, hidden_dim)
        
        # Compute relative positions and squared distances
        coors_i = coors.unsqueeze(2)  # (B, N, 1, 3)
        coors_j = coors.unsqueeze(1)  # (B, 1, N, 3)
        rel_pos = coors_i - coors_j  # (B, N, N, 3)
        sq_dist = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)  # (B, N, N, 1)
        
        # Concatenate h_i, h_j, and squared distance to form input to φ_e
        h_i_expanded = h_i.expand(-1, -1, N, -1)   # (B, N, N, hidden_dim)
        h_j_expanded = h_j.expand(-1, N, -1, -1)   # (B, N, N, hidden_dim)
        message_input = torch.cat([h_i_expanded, h_j_expanded, sq_dist], dim=-1)  # (B, N, N, 2*hidden_dim+1)

        # m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
        m_ij = self.phi_e(message_input)          # (B, N, N, hidden_dim)

        # φ_x(m_ij) -> scalar per edge
        phi_x_output = self.phi_x(m_ij).squeeze(-1)  # (B, N, N)
        
        # Apply adjacency mask (only consider neighbors)
        if len(adj_mat.shape) == 2:
            adj_mat = adj_mat.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        adj_mask = adj_mat.float()  # (B, N, N)
        
        # Mask out self-connections (j != i)
        identity = torch.eye(N, device=adj_mat.device, dtype=torch.float32)
        if len(identity.shape) == 2:
            identity = identity.unsqueeze(0).expand(B, -1, -1)
        adj_mask = adj_mask * (1 - identity)  # Remove self-connections
        
        # Compute neighbor interaction term: sum_j (x_i - x_j) * phi_x(m_ij)
        # rel_pos: (B, N, N, 3), phi_x_output: (B, N, N)
        phi_x_masked = phi_x_output.unsqueeze(-1) * adj_mask.unsqueeze(-1)  # (B, N, N, 1)
        neighbor_term = torch.sum(rel_pos * phi_x_masked, dim=2)  # (B, N, 3)
        
        # Normalize neighbor_term by number of neighbors to prevent explosion
        # Count neighbors per node
        num_neighbors = adj_mask.sum(dim=2, keepdim=True)  # (B, N, 1)
        num_neighbors = torch.clamp(num_neighbors, min=1.0)  # Avoid division by zero
        neighbor_term = neighbor_term / num_neighbors  # Normalize by neighbor count
        
        # Compute scalar gate φ_v(h_i^l) (γ_i in Eq. (7))
        # Add sigmoid to constrain gamma to [0, 1] to prevent velocity explosion
        gamma = torch.sigmoid(self.phi_v(h))  # (B, N, 1) - now in [0, 1]

        # Velocity prediction (Eq. (7)):
        # v_i^{l+1} = φ_v(h_i^l) v_i^{init} + C ∑_{j≠i} (x_i^l - x_j^l) φ_x(m_ij)
        pred_vel = gamma * v_init + self.C * neighbor_term   # (B, N, 3)
        
        # Stress prediction from node embeddings
        pred_stress = self.stress_head(h)
        
        # Coordinate prediction: x_i^{l+1} = x_i^l + v_i^{l+1}
        pred_coors = coors + pred_vel
        
        return pred_vel, pred_stress, pred_coors