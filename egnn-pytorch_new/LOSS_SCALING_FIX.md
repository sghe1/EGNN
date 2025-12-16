# Loss Scaling Fix - Root Cause Analysis and Solutions

## Root Cause: Loss Scale ~1e16 (Expected ~1e-3)

### Primary Issues Identified

1. **Velocity Normalization Mismatch** (CRITICAL BUG)
   - **Dataset normalization**: `velocity_norm = (velocity - vel_mean) / vel_std` (subtracts mean)
   - **Training target normalization**: `target_vel_norm = target_vel / vel_std` (missing mean subtraction!)
   - **Impact**: Predictions and targets are in different normalization spaces, causing huge errors

2. **Very Small `vel_std`**
   - From output: `vel_std = 0.000125` (extremely small)
   - When dividing by this, values are multiplied by ~8000, making normalized values huge
   - Combined with normalization mismatch, this causes massive loss values

3. **Huge Loss Weight**
   - Default `velocity_loss_weight = 50000.0` multiplies velocity loss by 50,000x
   - This scales already-large losses to astronomical values

4. **High Proximity Edge Density**
   - Average degree: 237.37 (way too high)
   - Causes numerical instability and memory issues
   - Should be capped to reasonable values (e.g., avg degree < 50)

5. **Loss Reduction Verification**
   - Code correctly uses MEAN (divides by mask.sum()), but need to verify in all paths

---

## Fixes Applied

### 1. Fixed Velocity Normalization Mismatch

**File:** `egnn_train.py`, function `train_epoch()`, lines ~670-690

**Change:**
```python
# BEFORE (WRONG):
target_vel_t_norm = target_vel_t / vel_std  # Missing mean subtraction!

# AFTER (CORRECT):
vel_mean = torch.tensor(dataset.vel_mean, device=device, dtype=torch.float32)  # (3,)
target_vel_t_norm = (target_vel_t - vel_mean.unsqueeze(0).unsqueeze(0)) / vel_std  # Matches dataset normalization
```

**Why:** Predictions are normalized as `(v - mean) / std`, so targets must be normalized the same way for loss computation to be in consistent space.

---

### 2. Fixed Denormalization for Rollout

**File:** `egnn_train.py`, function `train_epoch()`, line ~683

**Change:**
```python
# BEFORE:
pred_vel_denorm = pred_vel_norm * vel_std  # Missing mean addition!

# AFTER:
pred_vel_denorm = pred_vel_norm * vel_std + vel_mean.unsqueeze(0).unsqueeze(0)  # Correct denormalization
```

**Why:** Denormalization must reverse normalization: `v_orig = v_norm * std + mean`

---

### 3. Added Debug Prints for First Batch

**File:** `egnn_train.py`, function `train_epoch()`, lines ~648-690

**Added:**
- Print predictions (normalized): mean, std, max
- Print targets (original scale): mean, std, max
- Print targets (normalized): mean, std, max
- Print normalization constants: vel_mean, vel_std, stress_mean, stress_std
- Print mask sums: velocity_mask.sum(), stress_mask.sum()
- Print loss components: error values, masked sums, final losses

**Purpose:** Verify normalization is consistent and loss scale is reasonable

---

### 4. Debug Mode: Force Loss Weights to 1.0

**File:** `egnn_train.py`, function `train_epoch()`, lines ~885-894

**Change:**
```python
# DEBUG MODE: Force loss weights to 1.0 for sanity check
is_debug_mode = (max_timesteps is not None and max_timesteps <= 10)
if is_debug_mode:
    velocity_loss_weight_actual = 1.0
    stress_loss_weight_actual = 1.0
else:
    velocity_loss_weight_actual = velocity_loss_weight
    stress_loss_weight_actual = 1.0
```

**Why:** In debug mode, we want to see raw loss values without huge scaling factors.

---

### 5. Debug Mode: Reduce Proximity Edge Density

**File:** `egnn_train.py`, function `main()`, lines ~1162-1171

**Change:**
```python
if args.debug_one_traj:
    dataset = DeformingPlateDataset(
        ...
        proximity_radius=0.05,  # Reduced from default 0.1
        max_proximity_edges_per_node=16  # Cap edges per node
    )
```

**Why:** High edge density (avg degree 237) causes numerical issues. Capping to 16 edges per node keeps avg degree reasonable.

---

### 6. Verified Loss Reduction Uses MEAN

**File:** `egnn_train.py`, function `train_epoch()`, lines ~782, ~817

**Verified:**
- Velocity loss: `loss_vel = vel_error_masked.sum() / (velocity_mask.sum() + 1e-8)` ✓ MEAN
- Stress loss: `loss_stress = stress_error_masked.sum() / (stress_mask.sum() + 1e-8) + penalties` ✓ MEAN

**Status:** Already correct, no changes needed.

---

## Expected Results After Fixes

1. **Normalized loss values**: Should be O(1) to O(1e-3) in normalized space
2. **Consistent normalization**: Predictions and targets in same space
3. **Reasonable loss scale**: Total loss should be ~1-10 in first epoch, decreasing over time
4. **Lower edge density**: Average degree should be < 50 (with max_edges_per_node=16)

---

## Files Modified

1. **`egnn_train.py`**:
   - Fixed velocity normalization (add mean subtraction)
   - Fixed velocity denormalization (add mean addition)
   - Added debug prints for first batch
   - Added debug mode loss weight override (1.0 for both)
   - Added debug mode proximity edge reduction

---

## Testing

After fixes, run debug training and verify:
- Loss values are reasonable (O(1) to O(1e-3))
- Debug prints show consistent normalization
- Loss decreases over epochs
- Edge statistics show reduced density

---

**End of Fix Summary**
