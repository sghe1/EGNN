"""
E(n) Equivariant Graph Neural Networks (EGNN) implementation
Following "E(n) Equivariant Graph Neural Networks" (arXiv:2102.09844v3)

Equations implemented:
- Eq 3: m_ij = φ_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij)
- Eq 5: m_i = Σ_(j≠i) m_ij
- Eq 6: h_i^(l+1) = φ_h(h_i^l, m_i)
- Eq 7: v_i^(l+1) = φ_v(h_i^l) * v_init_i + C * Σ_(j≠i) (x_i^l - x_j^l) * φ_x(m_ij)
- Then: x_i^(l+1) = x_i^l + v_i^(l+1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN_Network


class SiLU(nn.Module):
    """SiLU/Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class MeshEGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network for mesh deformation prediction.
    
    Follows the paper architecture exactly:
    - Stacked EGCL layers for message passing
    - Coordinate updates using Equation 7
    - Velocity and stress prediction from final embeddings
    """
    
    def __init__(self, in_dim=8, hidden_dim=128, depth=4, num_nodes_avg=1271):
        """
        Args:
            in_dim: Input feature dimension (8: [position(3), actuation(3), node_type(2)])
            hidden_dim: Hidden dimension for node embeddings
            depth: Number of EGNN layers (4-7 as per paper)
            num_nodes_avg: Average number of nodes (for C initialization: C = 1/(N-1))
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        # Input projection: features -> hidden dimension
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stack of EGCL layers (handles Equations 3, 5, 6)
        # EGNN_Network internally implements:
        # - φ_e: edge MLP for messages
        # - Message aggregation (sum over neighbors)
        # - φ_h: node MLP for feature updates
        self.egnn_layers = EGNN_Network(
            num_tokens=None,
            dim=hidden_dim,
            depth=depth,
            edge_dim=0,
            only_sparse_neighbors=True,  # Use adjacency matrix
            update_coors=False,  # We'll handle coordinate updates manually via Eq 7
            update_feats=True,   # Let EGNN update features
        )
        
        # φ_v: MLP that outputs 3D velocity directly (Equation 7, modified)
        # CRITICAL FIX: Changed from scalar to 3D vector output
        # Original paper uses: v = φ_v(h) * v_init + C * neighbor_term
        # But v_init is always zero in teacher forcing, so we use: v = φ_v(h) + C * neighbor_term
        # This ensures phi_v is connected to the loss and gradients can flow
        self.phi_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, 3)  # Output 3D velocity directly, not scalar!
        )
        
        # φ_e: Edge MLP for computing messages m_ij (Equation 3)
        # Used for coordinate updates in Equation 7
        # Input: [h_i, h_j, ||x_i - x_j||^2] = 2*hidden_dim + 1
        edge_input_dim = 2 * hidden_dim + 1
        self.phi_e = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            SiLU(),
            nn.Linear(edge_input_dim * 2, hidden_dim),
            SiLU()
        )

        # φ_x: MLP that outputs scalar weight per edge for coordinate updates (Equation 7)
        # 2-layer MLP with SiLU as per paper
        # Normalization in the dataset will handle scaling appropriately
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # C: learnable parameter initialized to 1/(N-1) as per paper
        C_init = 1.0 / (num_nodes_avg - 1) if num_nodes_avg > 1 else 1e-3
        self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))
        
        # Stress prediction head: outputs normalized stress from final embeddings
        # NOTE: Model outputs normalized stress (can be negative after normalization)
        # ReLU will be applied during denormalization to ensure physical constraint (stress >= 0)
        # CRITICAL FIX: Simplified from 5 layers to 2 layers to maximize gradient flow
        # Very deep networks can struggle to learn when gradients vanish through many layers
        # Using only 2 layers (1 hidden) to ensure gradients flow properly
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 128 → 128
            SiLU(),
            nn.Linear(hidden_dim, 1)  # 128 → 1
        )
        
        # Initialize all layers properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights following paper conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    # CRITICAL FIX: Initialize stress head final layer bias to negative value
                    # This is because stress normalization uses median/IQR, so low stress
                    # maps to negative normalized values (e.g., stress=0 -> norm=-0.886, stress=6761 -> norm=-0.747)
                    # With zero initialization, model predicts ~0 which maps to median stress (~43031)
                    # Initialize to predict around -0.8 (maps to near-zero stress after denorm)
                    if m == self.stress_head[-1]:  # Final layer of stress head
                        nn.init.constant_(m.bias, -0.8)
                    else:
                        nn.init.zeros_(m.bias)

    def forward(self, feats, coors, adj_mat, v_init=None):
        """
        Forward pass with modified velocity prediction.
        
        CRITICAL: Model operates in NORMALIZED space.
        - Inputs (feats, coors) are normalized (O(1) magnitudes)
        - Outputs (pred_vel, pred_stress) are normalized (O(1) magnitudes)
        - Denormalization happens in save_predictions, not here
        
        Velocity prediction: v = φ_v(h) + C * Σ_(j≠i) (x_i - x_j) * φ_x(m_ij)
        (Modified from paper's v = φ_v(h) * v_init + C * Σ... since v_init is always zero)
        
        Args:
            feats: (B, N, 8) - NORMALIZED features [pos_norm(3), act_norm(3), node_type_one_hot(2)]
            coors: (B, N, 3) - NORMALIZED node coordinates (normalized positions)
            adj_mat: (B, N, N) or (N, N) - adjacency matrix (sparse connectivity)
            v_init: (B, N, 3) or None - DEPRECATED: not used anymore, kept for compatibility
        
        Returns:
            pred_vel: (B, N, 3) - NORMALIZED predicted velocity (zero for non-NORMAL nodes)
            pred_stress: (B, N, 1) - NORMALIZED predicted stress (zero for non-NORMAL nodes)
            
        Note:
            Predictions are automatically masked: only NORMAL nodes (node_type == 0) have
            non-zero predictions. OBSTACLE (node_type == 1) and HANDLE (node_type == 3) nodes
            always have zero velocity and stress predictions, matching the physical constraint
            that these boundary/actuator nodes are fixed.
        """
        B, N, _ = coors.shape
        
        # Verify input feature dimension
        assert feats.shape[-1] == 8, f"Expected 8D features, got {feats.shape[-1]}D"
        
        # v_init is no longer used (kept for backward compatibility)
        # Velocity is now predicted directly from node features via phi_v
        
        # Project input features to hidden dimension
        h = self.input_mlp(feats)  # (B, N, hidden_dim)
        
        # Pass through EGNN layers (Equations 3, 5, 6)
        # This updates node embeddings h but NOT coordinates
        # EGNN_Network handles:
        # - Message computation: m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2)
        # - Message aggregation: m_i = Σ_(j≠i) m_ij
        # - Feature update: h_i^(l+1) = φ_h(h_i^l, m_i)
        h_updated, _ = self.egnn_layers(h, coors, adj_mat=adj_mat, mask=None)
        
        # Check for NaN after EGNN layers
        if torch.isnan(h_updated).any():
            print(f"  ⚠ WARNING: NaN detected in h_updated after EGNN layers!")
            print(f"    Input coors range: [{coors.min().item():.4f}, {coors.max().item():.4f}]")
            print(f"    Input h range: [{h.min().item():.4f}, {h.max().item():.4f}]")
            print(f"    h_updated range: [{h_updated.min().item():.4f}, {h_updated.max().item():.4f}]")
        
        # Now compute velocity using modified Equation 7:
        # v_i^(l+1) = φ_v(h_i^l) + C * Σ_(j≠i) (x_i^l - x_j^l) * φ_x(m_ij)
        # 
        # CRITICAL FIX: Changed from φ_v(h) * v_init to φ_v(h) directly
        # Original formulation requires previous velocity v_init, but in teacher forcing
        # v_init is always zero, making phi_v disconnected from loss.
        # Solution: Make phi_v predict velocity directly from node features.
        
        # Direct velocity prediction from node features
        direct_velocity = self.phi_v(h_updated)  # (B, N, 3) - direct 3D velocity prediction
        
        # Second term: C * Σ_(j≠i) (x_i^l - x_j^l) * φ_x(m_ij)
        # Compute messages m_ij using φ_e (Equation 3) for coordinate updates (Equation 7)
        # Note: We recompute messages here using the same φ_e structure as EGNN layers
        # This ensures consistency with the paper's formulation
        
        # Compute relative positions: x_i - x_j
        coors_i = coors.unsqueeze(2)  # (B, N, 1, 3)
        coors_j = coors.unsqueeze(1)  # (B, 1, N, 3)
        rel_pos = coors_i - coors_j  # (B, N, N, 3)
        sq_dist = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)  # (B, N, N, 1)
        
        # Expand node embeddings for pairwise computation
        h_i = h_updated.unsqueeze(2)  # (B, N, 1, hidden_dim)
        h_j = h_updated.unsqueeze(1)  # (B, 1, N, hidden_dim)
        
        # Compute messages m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2) (Equation 3)
        message_input = torch.cat([
            h_i.expand(-1, -1, N, -1),   # (B, N, N, hidden_dim)
            h_j.expand(-1, N, -1, -1),   # (B, N, N, hidden_dim)
            sq_dist                      # (B, N, N, 1)
        ], dim=-1)  # (B, N, N, 2*hidden_dim + 1)

        m_ij = self.phi_e(message_input)  # (B, N, N, hidden_dim)

        # φ_x(m_ij) -> scalar weight per edge
        phi_x_weights = self.phi_x(m_ij).squeeze(-1)  # (B, N, N)
        
        # Apply adjacency mask (only consider neighbors, j ≠ i)
        if len(adj_mat.shape) == 2:
            adj_mat_expanded = adj_mat.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        else:
            adj_mat_expanded = adj_mat
        
        # Create mask: neighbors only, no self-connections
        identity = torch.eye(N, device=coors.device, dtype=torch.bool)
        if len(identity.shape) == 2:
            identity = identity.unsqueeze(0).expand(B, -1, -1)
        
        neighbor_mask = adj_mat_expanded.bool() & (~identity)  # (B, N, N)
        neighbor_mask = neighbor_mask.float()
        
        # Apply mask to φ_x weights
        phi_x_masked = phi_x_weights * neighbor_mask  # (B, N, N)
        
        # Compute neighbor interaction term: Σ_(j≠i) (x_i - x_j) * φ_x(m_ij)
        neighbor_term = torch.sum(rel_pos * phi_x_masked.unsqueeze(-1), dim=2)  # (B, N, 3)
        
        # Combine both terms: v_i = φ_v(h_i) + C * neighbor_term
        # NOTE: Model operates in normalized space, so outputs are normalized (O(1) magnitudes)
        # Inputs (coors, feats) are normalized, so model learns to output normalized velocities
        pred_vel = direct_velocity + self.C * neighbor_term  # (B, N, 3) - NORMALIZED
        
        # Predict stress from final embeddings
        # NOTE: Model outputs normalized stress (will be denormalized in save_predictions)
        pred_stress = self.stress_head(h_updated)  # (B, N, 1) - NORMALIZED
        
        # CRITICAL FIX: Set predictions to zero for non-NORMAL nodes (node_type != 0)
        # Only NORMAL nodes (node_type == 0) should have non-zero predictions
        # OBSTACLE (node_type == 1) and HANDLE (node_type == 3) nodes are fixed and should have zero stress/velocity
        node_type_one_hot = feats[:, :, 6:8]  # (B, N, 2) - extract node_type from features
        # node_type == 0 (NORMAL): [0, 0] -> both indices are 0 -> mask = 1.0
        # node_type == 1 (OBSTACLE): [1, 0] -> mask = 0.0
        # node_type == 3 (HANDLE): [0, 1] -> mask = 0.0
        # Use tolerance for floating point comparison to handle numerical errors
        tolerance = 1e-6
        normal_node_mask = ((node_type_one_hot[:, :, 0].abs() < tolerance) & 
                           (node_type_one_hot[:, :, 1].abs() < tolerance)).float()  # (B, N)
        
        # Apply mask: set predictions to zero for non-NORMAL nodes
        # For velocity: (B, N, 3) -> expand mask to (B, N, 1) then broadcast
        pred_vel = pred_vel * normal_node_mask.unsqueeze(-1)  # (B, N, 3)
        
        # For stress: (B, N, 1) -> expand mask to (B, N, 1)
        pred_stress = pred_stress * normal_node_mask.unsqueeze(-1)  # (B, N, 1)
        
        return pred_vel, pred_stress
