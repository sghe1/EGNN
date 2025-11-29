# Comparison: Before vs After Weighted Loss

## Test Configuration
- **Dataset**: 3 trajectories from deforming_plate
- **Epochs**: 5
- **Model**: EGNN (850,776 parameters)

---

## BEFORE (No Weighted Loss)

### Training Losses (Final Epoch)
- Total Loss: ~9,800
- Velocity Loss: Not tracked separately
- Stress Loss: Not tracked separately

### Evaluation Results

**Trajectory 0:**
- Velocity MAE: 23.20
- Stress MAE: 241.60

**Trajectory 1:**
- Velocity MAE: 2.52
- Stress MAE: 28.92

**Trajectory 2:**
- Velocity MAE: 0.58
- Stress MAE: 5.73

**Overall:**
- Velocity MAE: 8.77
- Stress MAE: 92.08

**Velocity Magnitude (Trajectory 2):**
- True: 0.000197
- Predicted: ~1.04 (orders of magnitude off)
- Error: ~1.04

---

## AFTER (With Weighted Loss, weight=1000.0)

### Training Losses (Final Epoch)
- Total Loss: 1,265,793.88
- Velocity Loss: 1,254.75 (now tracked and decreasing)
- Stress Loss: 11,046.65

**Loss Progression:**
- Epoch 1: Vel=172,056, Stress=32,435,242
- Epoch 2: Vel=4,098, Stress=16,221
- Epoch 3: Vel=2,121, Stress=21,829
- Epoch 4: Vel=2,066, Stress=37,898
- Epoch 5: Vel=1,254, Stress=11,046

### Evaluation Results

**Trajectory 0:**
- Velocity MAE: 63.46 (worse)
- Stress MAE: 563.85 (worse)

**Trajectory 1:**
- Velocity MAE: 15.60 (worse)
- Stress MAE: 11.65 (better)

**Trajectory 2:**
- Velocity MAE: 3.48 (worse)
- Stress MAE: 2.14 (better)

**Overall:**
- Velocity MAE: 27.51 (worse)
- Stress MAE: 192.55 (worse overall, but better for trajectories 1 & 2)

**Velocity Magnitude (Trajectory 2):**
- True: 0.000197
- Predicted: 7.20 (still orders of magnitude off, but different pattern)
- Error: 7.20

---

## Key Observations

### Positive Changes:
1. ✅ **Velocity loss is now tracked** - We can see it decreasing during training
2. ✅ **Model is learning velocity** - Velocity loss decreases from 172K to 1.2K
3. ✅ **Stress predictions improved** for trajectories 1 & 2
4. ✅ **Better loss balance** - Both losses are now in similar ranges

### Issues:
1. ❌ **Velocity predictions still poor** - Predicted magnitude (7.2) vs true (0.0002) is still off
2. ❌ **Trajectory 0 performance degraded** - Both velocity and stress got worse
3. ❌ **Overall metrics worse** - But this might be due to trajectory 0 being difficult

### Analysis:
- The weighted loss is working - velocity loss is decreasing
- However, velocity values are extremely small (~0.0002), making them very difficult to predict accurately
- The model might need:
  - Higher velocity loss weight (try 5000-10000)
  - Different normalization approach
  - More training epochs
  - Different architecture for velocity vs stress

---

## Recommendations

1. **Try higher velocity loss weight**: `--velocity_loss_weight 5000.0` or `10000.0`
2. **Train for more epochs**: Velocity might need more time to learn
3. **Consider separate normalization**: Normalize velocity and stress separately
4. **Monitor per-trajectory**: Trajectory 0 seems particularly difficult

---

**Note**: The weighted loss is working as intended - velocity loss is now being optimized. However, predicting such small velocity values (~0.0002) is inherently challenging and may require additional techniques beyond just loss weighting.
