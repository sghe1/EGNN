import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EGNN_DefPlate(nn.Module):
    """
    EGNN model wrapped to perfectly replace GraphUNet_DefPlate.
    It adheres to the exact same API (methods and signatures).
    """

    def __init__(self, in_dim, vel_out_dim, stress_out_dim, model_config_hyperparams, adj_norm):
        """
        Exact same signature as GraphUNet_DefPlate.
        
        Args:
            in_dim: Input node feature dimension.
            vel_out_dim: Output dimension for velocity.
            stress_out_dim: Output dimension for stress.
            model_config_hyperparams: argparse.Namespace (requires .hid_gnn_layer_dim)
            adj_norm: (Ignored in EGNN but kept for compatibility)
        """
        super().__init__()
        
        # 1. Map hyperparams from the Namespace object
        # We use 'hid_gnn_layer_dim' as the main hidden dimension for EGNN
        hidden_dim = model_config_hyperparams.hid_gnn_layer_dim
        
        # Try to find a depth/layer parameter, default to 4 if not found in the config namespace
        if hasattr(model_config_hyperparams, 'num_layers'):
            depth = model_config_hyperparams.num_layers
        elif hasattr(model_config_hyperparams, 'depth'):
            depth = model_config_hyperparams.depth
        elif hasattr(model_config_hyperparams, 'k_pool_ratios'):
            depth = len(model_config_hyperparams.k_pool_ratios) + 1
        else:
            depth = 4
            
        # 2. Heuristic for coordinate slicing (copied from EGNN snippet)
        # If in_dim > 12, coords are likely at [3:6] (MeshPos included), else at [0:3]
        self.pos_slice = slice(3, 6) if in_dim > 12 else slice(0, 3)
        feat_dim = in_dim - 3 # We remove the 3 coordinate channels from the features
        
        # 3. Define Architecture
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
        
        # Learnable parameter for residual connection
        self.C = nn.Parameter(torch.tensor(0.001))
        
        # Stress Head
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
                    if m == self.stress_head[-1]: 
                        nn.init.constant_(m.bias, -0.8)
                    else: 
                        nn.init.zeros_(m.bias)

    def forward(self, batch_adj_A, batch_feat_X, feat_tp1_mat_list=None, **kwargs):
        """
        Exact signature match for the original GraphUNet forward.
        Accepts optional args to be safe, but primarily uses A and X.
        """
        # Prediction
        preds_list = self.embed(batch_adj_A, batch_feat_X)
        return preds_list

    def rollout_step(self, A, X_t):
        """
        Single-step prediction (no loss), for rollouts.
        Matches GraphUNet API.
        """
        return self.embed(A, X_t)

    def embed(self, adj_A_list, X_list):
        """
        Process a batch of graphs.
        Matches GraphUNet API.
        """
        # Single-graph case: tensors
        if isinstance(adj_A_list, torch.Tensor):
            return self.embed_one(adj_A_list, X_list)
        # Batch case: list of tensors
        return [self.embed_one(A, X) for A, X in zip(adj_A_list, X_list)]

    def embed_one(self, adj, x):
        """
        Process a single graph using the Optimized EGNN logic.
        """
        pos = x[:, self.pos_slice]
        
        # Concatenate features excluding coords
        # This handles the case where coords are in the middle or beginning
        h_in = torch.cat([x[:, :self.pos_slice.start], x[:, self.pos_slice.stop:]], dim=1)
        
        # Unsqueeze for batch dimension required by library [1, N, C]
        h_b, pos_b, adj_b = h_in.unsqueeze(0), pos.unsqueeze(0), adj.unsqueeze(0)
        
        # Initial Embedding
        h_emb = self.input_mlp(h_b)
        
        # EGNN Layers
        h_updated, _ = self.egnn_layers(h_emb, pos_b, adj_mat=adj_b)
        
        # --- Physics Update (Sparse Optimization) ---
        v_direct = self.phi_v(h_updated)
        
        # Convert adjacency to sparse format for efficient edge computation
        N = pos_b.shape[1]
        adj_flat = adj_b.squeeze(0)  # [N, N]
        
        # Get edge indices (only existing edges, no self-loops)
        edge_mask = (adj_flat > 0) & (~torch.eye(N, device=adj_flat.device, dtype=torch.bool))
        edge_indices = edge_mask.nonzero(as_tuple=False).t()  # [2, E]
        
        if edge_indices.shape[1] > 0:
            i_idx, j_idx = edge_indices[0], edge_indices[1]
            
            # Compute relative positions
            pos_i = pos_b.squeeze(0)[i_idx]  # [E, 3]
            pos_j = pos_b.squeeze(0)[j_idx]  # [E, 3]
            rel_pos_edges = pos_i - pos_j
            dist_sq_edges = (rel_pos_edges ** 2).sum(-1, keepdim=True)
            
            # Get node features for edges
            h_i = h_updated.squeeze(0)[i_idx]
            h_j = h_updated.squeeze(0)[j_idx]
            
            # Compute edge features
            edge_feat = torch.cat([h_i, h_j, dist_sq_edges], dim=-1)
            
            # Compute weights for edges
            weights_edges = self.phi_x(self.phi_e_proj(edge_feat)).squeeze(-1)
            
            # Aggregate neighbor contributions
            neighbor_term = torch.zeros_like(v_direct.squeeze(0))
            # Sparse Add: For each edge (i->j), add contribution to node i
            neighbor_term.index_add_(0, i_idx, rel_pos_edges * weights_edges.unsqueeze(-1))
        else:
            neighbor_term = torch.zeros_like(v_direct.squeeze(0))
        
        neighbor_term = neighbor_term.unsqueeze(0)
        
        # Final predictions
        pred_vel = v_direct + self.C * neighbor_term
        pred_stress = self.stress_head(h_updated)
        
        # Concatenate and remove batch dim: [1, N, F_out] -> [N, F_out]
        return torch.cat([pred_vel, pred_stress], dim=-1).squeeze(0)