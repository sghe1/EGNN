# C Initialization Change Summary

## Changes Made

1. **Changed C initialization from fixed `0.001` to `1/(N-1)`** where N is the number of nodes in the graph
   - Location: `egnn_deforming_plate.py:82-85`
   - Uses lazy initialization pattern: C starts as sentinel value (-1.0), initialized on first forward pass
   - Computes N from input tensor shape: `N = x.shape[0]`

2. **Added checkpoint compatibility**
   - Location: `egnn_deforming_plate.py:85` - `register_buffer('_c_initialized', ...)`
   - If checkpoint loads a trained C value, it overwrites the sentinel and initialization is skipped
   - Buffer `_c_initialized` tracks whether C has been set (either from init or checkpoint)

3. **No other changes**
   - Stress loss masking unchanged (still includes BOUNDARY nodes)
   - Forward-pass masking unchanged (still only in loss function)
   - All other architecture/training logic unchanged

## EGNN Verification Evidence

The model is confirmed to be EGNN-style (not transformer/GAT/etc.):

### 1. Edge/Message Function Uses r²
- **Location**: `egnn_pytorch.py:233`
- **Code**: `rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)`
- **Evidence**: Computes squared distance `||x_i - x_j||²`

### 2. Message Computation: m_ij = φ_e(h_i, h_j, r²)
- **Location**: `egnn_deforming_plate.py:196`
- **Code**: `edge_feat = torch.cat([h_i, h_j, dist_sq_edges], dim=-1)`
- **Evidence**: Concatenates node embeddings h_i, h_j with squared distance

### 3. Node Update Uses Aggregated Messages
- **Location**: `egnn_pytorch.py:332-337`
- **Code**: 
  ```python
  m_i = m_ij.sum(dim=-2)  # Sum aggregation
  node_mlp_input = torch.cat((normed_feats, m_i), dim=-1)
  node_out = self.node_mlp(node_mlp_input) + feats  # Residual
  ```
- **Evidence**: Sums messages over neighbors, feeds to φ_h MLP with residual connection

### 4. Velocity Uses Equivariant Directional Term
- **Location**: `egnn_deforming_plate.py:189, 197, 200, 207`
- **Code**:
  ```python
  rel_pos_edges = pos0[i_idx] - pos0[j_idx]  # (x_i - x_j)
  weights_edges = self.phi_x(self.phi_e_proj(edge_feat)).squeeze(-1)  # φ_x(m_ij) scalar
  neighbor_term.index_add_(0, i_idx, rel_pos_edges * weights_edges.unsqueeze(-1))
  pred_vel = v_direct + self.C * neighbor_term
  ```
- **Evidence**: Uses coordinate differences (x_i - x_j) weighted by scalar φ_x(m_ij)

### 5. No Coordinate Updates Inside Network
- **Location**: `egnn_deforming_plate.py:66`
- **Code**: `update_coors=False`
- **Evidence**: Coordinates are NOT updated inside EGNN layers, only used for messages and velocity head

### 6. No Transformer/GAT Patterns
- **Search Results**: No `MultiheadAttention`, `GAT`, or `transformer` modules found
- **Only mentions**: Comments about attention in egnn_pytorch.py (global attention option, not used here)

## Verification Results

Run: `python verify_egnn_c_init.py`

All tests pass:
- ✓ C correctly initialized to 1/(N-1) for N=10, 100, 1000
- ✓ Message passing uses r² and (x_i - x_j)
- ✓ phi_v outputs 3D vector
- ✓ phi_x outputs scalar per edge
- ✓ No coordinate updates inside network

## Git Diff

```diff
--- a/EMA_EXPERIMENT/pytorch_model/model_egnn/egnn_deforming_plate.py
+++ b/EMA_EXPERIMENT/pytorch_model/model_egnn/egnn_deforming_plate.py
@@ -79,7 +79,10 @@ class EGNN_DefPlate(nn.Module):
         self.phi_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), SiLU(), nn.Linear(hidden_dim, 1))
 
         # Learnable parameter for residual connection
-        self.C = nn.Parameter(torch.tensor(0.001))
+        # Initialize with sentinel value; will be set to 1/(N-1) on first forward pass
+        # Checkpoint loading will overwrite this if C was already trained
+        self.C = nn.Parameter(torch.tensor(-1.0))
+        self.register_buffer('_c_initialized', torch.tensor(False))
 
         # Stress Head
         self.stress_head = nn.Sequential(
@@ -158,6 +161,19 @@ class EGNN_DefPlate(nn.Module):
         Process a single graph.
         This EGNN fork expects adj_mat as a boolean mask and derives neighborhoods internally.
         """
+        # Lazy initialization of C: set to 1/(N-1) on first forward if not already initialized
+        # (e.g., from checkpoint loading). If C was loaded from checkpoint (C != -1.0), skip init.
+        N = x.shape[0]
+        if not self._c_initialized.item():
+            # Check if C was loaded from checkpoint (not sentinel value)
+            if self.C.item() < 0:
+                # Initialize C = 1/(N-1), with fallback for edge cases
+                c_init = 1.0 / (N - 1) if N > 1 else 1e-3
+                with torch.no_grad():
+                    self.C.data.fill_(c_init)
+            # Mark as initialized regardless (either from checkpoint or just initialized)
+            self._c_initialized.fill_(True)
+        
         pos = x[:, self.pos_slice]
```

## Implementation Decisions

1. **N is variable per graph**: Computed from `x.shape[0]` in `embed_one()`
2. **Lazy initialization**: C initialized on first forward pass to avoid reinitializing every forward
3. **Checkpoint compatibility**: 
   - Sentinel value (-1.0) allows detection of uninitialized C
   - If checkpoint loads C (value != -1.0), initialization is skipped
   - Buffer `_c_initialized` prevents re-initialization after checkpoint load

## Command to Run Verification

```bash
cd EMA_EXPERIMENT/pytorch_model
python verify_egnn_c_init.py
```

Expected output: `✓ ALL VERIFICATIONS PASSED`
