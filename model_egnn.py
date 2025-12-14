"""
Optimized EGNN model with sparse operations for faster computation on Mac.

Key optimizations:
1. Physics update uses sparse edge operations - only computes edges that exist
2. Uses index_add for efficient sparse aggregation instead of dense matrix operations
3. Reduces O(N²) operations to O(E) where E is the number of edges
"""
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
        
        # Physics Update - OPTIMIZED: Only compute edges that exist (sparse)
        v_direct = self.phi_v(h_updated)
        
        # Convert adjacency to sparse format for efficient edge computation
        # adj_b is [1, N, N], we need edge indices
        N = pos_b.shape[1]
        adj_flat = adj_b.squeeze(0)  # [N, N]
        
        # Get edge indices (only existing edges, no self-loops)
        edge_mask = (adj_flat > 0) & (~torch.eye(N, device=adj_flat.device, dtype=torch.bool))
        edge_indices = edge_mask.nonzero(as_tuple=False).t()  # [2, E]
        
        if edge_indices.shape[1] > 0:
            # Only compute for existing edges (sparse)
            i_idx, j_idx = edge_indices[0], edge_indices[1]
            
            # Compute relative positions only for existing edges
            pos_i = pos_b.squeeze(0)[i_idx]  # [E, 3]
            pos_j = pos_b.squeeze(0)[j_idx]  # [E, 3]
            rel_pos_edges = pos_i - pos_j  # [E, 3]
            dist_sq_edges = (rel_pos_edges ** 2).sum(-1, keepdim=True)  # [E, 1]
            
            # Get node features for edges
            h_i = h_updated.squeeze(0)[i_idx]  # [E, hidden_dim]
            h_j = h_updated.squeeze(0)[j_idx]  # [E, hidden_dim]
            
            # Compute edge features only for existing edges
            edge_feat = torch.cat([h_i, h_j, dist_sq_edges], dim=-1)  # [E, 2*hidden_dim + 1]
            
            # Compute weights for edges
            weights_edges = self.phi_x(self.phi_e_proj(edge_feat)).squeeze(-1)  # [E]
            
            # Aggregate neighbor contributions using sparse operations
            # Original: neighbor_term[i] = sum_j (pos_i - pos_j) * weight[i,j] for all neighbors j
            # We compute this efficiently by only iterating over existing edges
            neighbor_term = torch.zeros_like(v_direct.squeeze(0))  # [N, 3]
            
            # Use index_add for efficient sparse aggregation
            # For each edge (i->j), add contribution to node i: (pos_i - pos_j) * weight[i,j]
            # Note: We only add to source node i, not to target j (to match original dense computation)
            neighbor_term.index_add_(0, i_idx, rel_pos_edges * weights_edges.unsqueeze(-1))
        else:
            neighbor_term = torch.zeros_like(v_direct.squeeze(0))
        
        neighbor_term = neighbor_term.unsqueeze(0)  # [1, N, 3]
        pred_vel = v_direct + self.C * neighbor_term
        pred_stress = self.stress_head(h_updated)
        
        return torch.cat([pred_vel, pred_stress], dim=-1).squeeze(0)