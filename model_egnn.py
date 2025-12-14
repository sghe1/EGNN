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
        batch: [B*N] batch assignment (optional, from PyTorch Geometric Data.batch)
        """
        row, col = edge_index
        
        # Handle batching: if batch is None, assume single graph
        if batch is None:
            num_nodes = x.size(0)
            batch_idx = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
            is_single_batch = True
        else:
            batch_idx = batch
            # Check if all nodes belong to the same graph
            unique_batches = torch.unique(batch_idx)
            is_single_batch = len(unique_batches) == 1
        
        # Handle both single and multi-batch cases
        if is_single_batch:
            num_nodes = x.size(0)
            
            # Reshape to (B, N, D) format for EGNN_Network
            x_batched = x.unsqueeze(0)  # (1, N, 9)
            pos_batched = pos.unsqueeze(0)  # (1, N, 3)
            
            # Convert edge_index to adjacency matrix
            adj_mat = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=x.device)
            adj_mat[row, col] = True
            
            # A. Encode
            h = self.input_mlp(x_batched)  # (1, N, hidden_dim)
            
            # B. Message Passing - EGNN_Network expects (B, N, D) format
            h_updated, _ = self.egnn_layers(h, pos_batched, adj_mat=adj_mat)
            
            # Flatten back to (N, hidden_dim) for rest of computation
            h_updated = h_updated.squeeze(0)  # (N, hidden_dim)
        else:
            # Multi-batch case: process each graph separately
            unique_batches, counts = torch.unique(batch_idx, return_counts=True)
            num_batches = len(unique_batches)
            
            h_updated_list = []
            for b_idx in unique_batches:
                # Get nodes for this batch
                mask = (batch_idx == b_idx)
                x_b = x[mask]
                pos_b = pos[mask]
                
                # Get edges for this batch (edges where both nodes are in this batch)
                batch_row_mask = mask[row]
                batch_col_mask = mask[col]
                edge_mask = batch_row_mask & batch_col_mask
                
                if edge_mask.sum() == 0:
                    # No edges in this batch, skip or use identity
                    h_b = self.input_mlp(x_b.unsqueeze(0)).squeeze(0)
                    h_updated_list.append(h_b)
                    continue
                
                # Remap node indices to local indices for this batch
                local_node_map = torch.zeros(mask.size(0), dtype=torch.long, device=x.device)
                local_node_map[mask] = torch.arange(mask.sum().item(), device=x.device)
                
                row_b = local_node_map[row[edge_mask]]
                col_b = local_node_map[col[edge_mask]]
                
                num_nodes_b = x_b.size(0)
                
                # Reshape to (B, N, D) format
                x_batched = x_b.unsqueeze(0)  # (1, N_b, 9)
                pos_batched = pos_b.unsqueeze(0)  # (1, N_b, 3)
                
                # Convert edge_index to adjacency matrix for this batch
                adj_mat_b = torch.zeros(num_nodes_b, num_nodes_b, dtype=torch.bool, device=x.device)
                adj_mat_b[row_b, col_b] = True
                
                # A. Encode
                h_b = self.input_mlp(x_batched)  # (1, N_b, hidden_dim)
                
                # B. Message Passing
                h_updated_b, _ = self.egnn_layers(h_b, pos_batched, adj_mat=adj_mat_b)
                
                # Flatten and store
                h_updated_list.append(h_updated_b.squeeze(0))
            
            # Concatenate all batches back together
            h_updated = torch.cat(h_updated_list, dim=0)
        
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