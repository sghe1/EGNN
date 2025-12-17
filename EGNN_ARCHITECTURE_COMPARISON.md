# EGNN Architecture Comparison: EMA_EXPERIMENT vs egnn-pytorch_new

## Executive Summary

**Verdict: Mostly similar but needs report tweak**

Both implementations follow the EGNN paper architecture with velocity/stress heads, but there are key differences in:
1. C initialization (EMA uses fixed 0.001, egnn-pytorch_new uses 1/(N-1))
2. Input feature dimensions (EMA: 12D with mesh_pos, egnn-pytorch_new: 8D without)
3. Loss masking (EMA: NORMAL+BOUNDARY for stress, egnn-pytorch_new: NORMAL only)
4. Coordinate update handling (both correctly avoid internal updates)

---

## 1. EGNN Core Module Locations

### EMA_EXPERIMENT/pytorch_model/model_egnn/

**Entry Point:**
- `egnn_deforming_plate.py:11` - `EGNN_DefPlate` class

**EGNN Layer/Network:**
- `egnn_deforming_plate.py:60` - Uses `EGNN_Network` from `egnn_pytorch.py`
- `egnn_pytorch.py:148` - `EGNN` class (single layer)
- `egnn_pytorch.py:343` - `EGNN_Network` class (stacked layers)

**Forward Pass:**
- `egnn_deforming_plate.py:103` - `forward()` method
- `egnn_deforming_plate.py:115` - `embed()` method (main logic)
- `egnn_deforming_plate.py:156` - `embed_one()` method (single graph processing)

**Heads:**
- `egnn_deforming_plate.py:71-75` - `phi_v` (velocity head)
- `egnn_deforming_plate.py:78-79` - `phi_e_proj` + `phi_x` (edge weight computation)
- `egnn_deforming_plate.py:85-89` - `stress_head` (stress prediction)

### egnn-pytorch_new/

**Entry Point:**
- `EGNN.py:34` - `MeshEGNN` class
- `model_egnn.py:18` - `MeshEGNN` class (duplicate, same structure)

**EGNN Layer/Network:**
- `EGNN.py:70` - Uses `EGNN_Network` from `egnn-pytorch/egnn_pytorch/egnn_pytorch.py`
- Same underlying library as EMA_EXPERIMENT

**Forward Pass:**
- `EGNN.py:153` - `forward()` method
- `model_egnn.py:137` - `forward()` method (identical)

**Heads:**
- `EGNN.py:85-91` - `phi_v` (velocity head)
- `EGNN.py:97-104` - `phi_e` (edge message MLP)
- `EGNN.py:109-115` - `phi_x` (scalar edge weight)
- `EGNN.py:125-131` - `stress_head` (stress prediction)

---

## 2. Structured Spec Sheets

### EMA_EXPERIMENT/pytorch_model/model_egnn/egnn_deforming_plate.py

| Component | Specification | Code Evidence |
|-----------|--------------|---------------|
| **Node embedding dim d** | `hidden_dim` (default 128) | Line 30: `hidden_dim = model_config_hyperparams.hid_gnn_layer_dim` |
| **Number of EGNN layers** | `depth` (default 4, from `k_pool_ratios` or config) | Lines 32-39: depth computation |
| **Residuals** | Yes, in EGNN_Network | `egnn_pytorch.py:337`: `node_out = self.node_mlp(node_mlp_input) + feats` |
| **LayerNorm/BatchNorm** | LayerNorm in EGNN layers | `egnn_pytorch.py:191`: `self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()` |
| **Dropout** | Configurable in EGNN_Network | `egnn_pytorch.py:176`: `dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()` |
| **φ_e inputs** | `[h_i, h_j, ||x_i - x_j||^2]` | Line 196: `edge_feat = torch.cat([h_i, h_j, dist_sq_edges], dim=-1)` (Note: Uses separate `phi_e_proj` for velocity head, not EGNN layer's internal `edge_mlp`) |
| **Message aggregation** | Sum (default) | `egnn_pytorch.py:332-333`: `m_i = m_ij.sum(dim=-2)` |
| **φ_h update form** | Residual: `h_out = MLP([h_norm, m_i]) + h` | `egnn_pytorch.py:335-337`: `node_mlp_input = torch.cat((normed_feats, m_i), dim=-1); node_out = self.node_mlp(node_mlp_input) + feats` |
| **Coordinate updates inside network?** | **NO** | Line 66: `update_coors=False` |
| **Velocity head** | `v = phi_v(h) + C * Σ (x_i - x_j) * phi_x(m_ij)` | Lines 181, 197-200, 207: `v_direct = self.phi_v(h_updated)`; `neighbor_term.index_add_(0, i_idx, rel_pos_edges * weights_edges.unsqueeze(-1))`; `pred_vel = v_direct + self.C * neighbor_term` |
| **phi_v output** | 3D vector (vel_out_dim=3) | Line 74: `nn.Linear(hidden_dim, vel_out_dim)` where vel_out_dim=3 |
| **phi_x output** | Scalar (1D) | Line 79: `nn.Linear(hidden_dim, 1)` |
| **C initialization** | **0.001 (fixed)** | Line 82: `self.C = nn.Parameter(torch.tensor(0.001))` |
| **Stress head** | Separate MLP from embeddings | Lines 85-89: `self.stress_head = nn.Sequential(...)` |
| **Node-type masking** | In loss function: NORMAL for vel, NORMAL+BOUNDARY for stress | `train.py:151-152`: `vel_mask = (nodetype == NORMAL_NODE)`; `stress_mask = (nodetype == NORMAL_NODE) | (nodetype == BOUNDARY_NODE)` |
| **Normalization** | Applied in dataloader (mean/std) | Data comes pre-normalized to model |
| **Denormalization** | Outside model (in evaluation/rollout) | `visualize_simulation.py:379-380`: `vel_pred = vel_norm * std_vec[vel_idxs] + mean_vec[vel_idxs]` |
| **Euler integration** | Outside network (in rollout) | `visualize_simulation.py:392`: `p_hat_next[deform_mask] = p_hat[deform_mask] + vel_pred[deform_mask]` |

### egnn-pytorch_new/EGNN.py

| Component | Specification | Code Evidence |
|-----------|--------------|---------------|
| **Node embedding dim d** | `hidden_dim` (default 128) | Line 44: `hidden_dim=128` |
| **Number of EGNN layers** | `depth` (default 4) | Line 44: `depth=4` |
| **Residuals** | Yes, in EGNN_Network | Same library as EMA_EXPERIMENT |
| **LayerNorm/BatchNorm** | LayerNorm in EGNN layers | Same library as EMA_EXPERIMENT |
| **Dropout** | Configurable in EGNN_Network | Same library as EMA_EXPERIMENT |
| **φ_e inputs** | `[h_i, h_j, ||x_i - x_j||^2]` | Lines 238-242: `message_input = torch.cat([h_i.expand(...), h_j.expand(...), sq_dist], dim=-1)` |
| **Message aggregation** | Sum (default) | Same library as EMA_EXPERIMENT |
| **φ_h update form** | Residual: `h_out = MLP([h_norm, m_i]) + h` | Same library as EMA_EXPERIMENT |
| **Coordinate updates inside network?** | **NO** | Line 76: `update_coors=False` |
| **Velocity head** | `v = phi_v(h) + C * Σ (x_i - x_j) * phi_x(m_ij)` | Lines 220, 247, 267, 272: `direct_velocity = self.phi_v(h_updated)`; `neighbor_term = torch.sum(rel_pos * phi_x_masked.unsqueeze(-1), dim=2)`; `pred_vel = direct_velocity + self.C * neighbor_term` |
| **phi_v output** | 3D vector | Line 90: `nn.Linear(hidden_dim, 3)` |
| **phi_x output** | Scalar (1D) | Line 114: `nn.Linear(hidden_dim, 1)` |
| **C initialization** | **1/(N-1) (adaptive)** | Lines 118-119: `C_init = 1.0 / (num_nodes_avg - 1) if num_nodes_avg > 1 else 1e-3`; `self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))` |
| **Stress head** | Separate MLP from embeddings | Lines 125-131: `self.stress_head = nn.Sequential(...)` |
| **Node-type masking** | In forward pass: NORMAL only for both | Lines 287-295: `normal_node_mask = ((node_type_one_hot[:, :, 0].abs() < tolerance) & (node_type_one_hot[:, :, 1].abs() < tolerance)).float()`; Applied to both `pred_vel` and `pred_stress` |
| **Normalization** | Applied in dataloader (mean/std) | Data comes pre-normalized to model |
| **Denormalization** | Outside model (in save_predictions) | `egnn_train.py:201, 209`: `pred_vel_denorm = pred_vel_norm * vel_std`; `pred_stress_denorm = pred_stress_norm * stress_std + stress_mean` |
| **Euler integration** | Outside network (in rollout) | `egnn_train.py:221`: `pred_pos_denorm = pos_prev_denorm + pred_vel_denorm` |

---

## 3. Direct Code Comparison Mapping

| Report Component | EMA_EXPERIMENT Code | egnn-pytorch_new Code | Match Status |
|------------------|---------------------|----------------------|--------------|
| **φ_e: m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2)** | `egnn_deforming_plate.py:196-197` - `edge_feat = torch.cat([h_i, h_j, dist_sq_edges], dim=-1)`; Uses `phi_e_proj` (separate from EGNN layer's `edge_mlp`) | `EGNN.py:238-244` - `message_input = torch.cat([h_i.expand(...), h_j.expand(...), sq_dist], dim=-1)`; Uses separate `phi_e` MLP | ✅ **Match** (both recompute messages for velocity head, structure equivalent) |
| **φ_h: h_i^(l+1) = φ_h(h_i^l, Σ m_ij)** | `egnn_pytorch.py:335-337` - Residual MLP with sum aggregation | `EGNN.py:182` - Same library, same implementation | ✅ **Match** |
| **Velocity: φ_v(h_i)** | `egnn_deforming_plate.py:71-75` - `phi_v = nn.Sequential(..., nn.Linear(hidden_dim, vel_out_dim))` | `EGNN.py:85-91` - `phi_v = nn.Sequential(..., nn.Linear(hidden_dim, 3))` | ✅ **Match** (both output 3D) |
| **Velocity: φ_x(m_ij) scalar** | `egnn_deforming_plate.py:79` - `nn.Linear(hidden_dim, 1)` | `EGNN.py:109-115` - `nn.Linear(hidden_dim, 1)` | ✅ **Match** |
| **Velocity: (x_i - x_j) coordinate diffs** | `egnn_deforming_plate.py:189` - `rel_pos_edges = pos0[i_idx] - pos0[j_idx]` | `EGNN.py:230` - `rel_pos = coors_i - coors_j` | ✅ **Match** |
| **Velocity: C learnable scalar** | `egnn_deforming_plate.py:82` - `self.C = nn.Parameter(torch.tensor(0.001))` | `EGNN.py:118-119` - `C_init = 1.0 / (num_nodes_avg - 1)` | ⚠️ **Different init** |
| **Velocity: Final formula** | `egnn_deforming_plate.py:207` - `pred_vel = v_direct + self.C * neighbor_term` | `EGNN.py:272` - `pred_vel = direct_velocity + self.C * neighbor_term` | ✅ **Match** |
| **Stress: φ_s(h_i)** | `egnn_deforming_plate.py:85-89` - `stress_head = nn.Sequential(...)` | `EGNN.py:125-131` - `stress_head = nn.Sequential(...)` | ✅ **Match** |
| **No coordinate updates inside** | `egnn_deforming_plate.py:66` - `update_coors=False` | `EGNN.py:76` - `update_coors=False` | ✅ **Match** |
| **Masking: NORMAL nodes only** | `train.py:151` - `vel_mask = (nodetype == NORMAL_NODE)` | `EGNN.py:287-288` - `normal_node_mask = ((node_type_one_hot[:, :, 0].abs() < tolerance) & (node_type_one_hot[:, :, 1].abs() < tolerance))` | ⚠️ **Different: EMA also includes BOUNDARY for stress** |
| **Normalization outside** | Data pre-normalized, model outputs normalized | Data pre-normalized, model outputs normalized | ✅ **Match** |
| **Euler outside network** | `visualize_simulation.py:392` - `p_hat_next[deform_mask] = p_hat[deform_mask] + vel_pred[deform_mask]` | `egnn_train.py:221` - `pred_pos_denorm = pos_prev_denorm + pred_vel_denorm` | ✅ **Match** |

---

## 4. Identified Mismatches

### Mismatch 1: C Initialization
**Location:**
- EMA_EXPERIMENT: `egnn_deforming_plate.py:82` - Fixed `0.001`
- egnn-pytorch_new: `EGNN.py:118-119` - `1.0 / (num_nodes_avg - 1)`

**Impact:** 
- EMA's fixed initialization may not scale well across different graph sizes
- egnn-pytorch_new follows paper recommendation more closely
- **Report alignment:** Report says "C learnable scalar init 1/(N-1) (or very close)" - egnn-pytorch_new matches, EMA does not

**Recommendation:** Update EMA to use adaptive initialization: `C_init = 1.0 / (num_nodes_avg - 1) if num_nodes_avg > 1 else 1e-3`

### Mismatch 2: Stress Loss Masking
**Location:**
- EMA_EXPERIMENT: `train.py:152` - `stress_mask = (nodetype == NORMAL_NODE) | (nodetype == BOUNDARY_NODE)`
- egnn-pytorch_new: `EGNN.py:287-295` - Only NORMAL nodes (node_type == 0)

**Impact:**
- EMA includes BOUNDARY nodes (node_type == 3) in stress loss, which may be physically reasonable if boundary nodes can have stress
- egnn-pytorch_new excludes boundary nodes from stress predictions entirely (masks to zero in forward pass)
- **Report alignment:** Report says "loss computed only on NORMAL/plate nodes; boundary/handle excluded" - egnn-pytorch_new matches exactly, EMA includes BOUNDARY

**Recommendation:** 
- If report is correct, EMA should exclude BOUNDARY from stress loss
- If BOUNDARY nodes should have stress, update report to clarify

### Mismatch 3: Input Feature Dimensions
**Location:**
- EMA_EXPERIMENT: `egnn_deforming_plate.py:46` - Handles `in_dim > 12` (includes mesh_pos at [3:6])
- egnn-pytorch_new: `EGNN.py:44, 168` - Expects `in_dim=12` (with mesh_pos) or `in_dim=8` (without mesh_pos)

**Impact:**
- Different feature encodings, but both handle normalization correctly
- Not a functional mismatch, just different data preprocessing

**Recommendation:** Document the feature dimension differences in report

### Mismatch 4: Forward Pass Masking Location
**Location:**
- EMA_EXPERIMENT: Masking in loss function only (`train.py:151-152`)
- egnn-pytorch_new: Masking in forward pass (`EGNN.py:287-295`) AND loss function

**Impact:**
- egnn-pytorch_new ensures predictions are zero for non-NORMAL nodes at model output
- EMA relies on loss masking only, but model may still predict non-zero values for boundary nodes
- **Report alignment:** Report says "masking by node type" but doesn't specify forward vs loss - both are valid

**Recommendation:** 
- EMA could add forward-pass masking for consistency
- Or document that masking is loss-only (acceptable if predictions are ignored for non-NORMAL nodes)

---

## 5. Conclusion

### Verdict: **"Mostly similar but needs report tweak"**

**What Matches:**
✅ Core EGNN message passing (φ_e, φ_h)  
✅ Velocity head formula: `v = φ_v(h) + C * Σ (x_i - x_j) * φ_x(m_ij)`  
✅ φ_v outputs 3D vector (not scalar)  
✅ φ_x outputs scalar (not vector)  
✅ No coordinate updates inside network  
✅ Stress head as separate MLP  
✅ Normalization/denormalization outside network  
✅ Euler integration outside network  

**What Needs Alignment:**

1. **C Initialization** (Minor)
   - **Action:** Update EMA to use `1/(N-1)` initialization OR update report to say "C initialized to small value (e.g., 0.001)"

2. **Stress Loss Masking** (Moderate)
   - **Action:** Clarify in report whether BOUNDARY nodes should be included in stress loss
   - If yes: Update egnn-pytorch_new to include BOUNDARY
   - If no: Update EMA to exclude BOUNDARY

3. **Forward Pass Masking** (Minor)
   - **Action:** Document whether masking should be in forward pass or loss only
   - Both are valid, but consistency is preferred

### Proposed Minimal Edits

**Option A: Update EMA_EXPERIMENT to match report**
1. Change C initialization: `self.C = nn.Parameter(torch.tensor(1.0 / (num_nodes_avg - 1) if num_nodes_avg > 1 else 1e-3))`
2. Update stress mask: `stress_mask = (nodetype == NORMAL_NODE)` (remove BOUNDARY)
3. Add forward-pass masking (optional, for consistency)

**Option B: Update report to match implementations**
1. Change C description: "C learnable scalar initialized to small value (e.g., 0.001) or 1/(N-1)"
2. Clarify stress masking: "loss computed on NORMAL nodes for velocity; NORMAL+BOUNDARY for stress (if boundary stress is physically meaningful)"
3. Note that masking can be applied in forward pass or loss function

**Recommendation:** **Option A** - Update EMA_EXPERIMENT to match report spec, as the report's specification (C=1/(N-1), NORMAL-only stress) is more aligned with the paper's intent.
