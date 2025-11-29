# Running EGNN Training on Google Colab

## Setup Instructions

### 1. Upload your project to Colab

You can either:
- **Option A**: Upload the entire project folder to Google Drive and mount it
- **Option B**: Clone from GitHub (if you have it in a repo)
- **Option C**: Upload files directly to Colab

### 2. Install Dependencies

```python
# Install PyTorch (Colab usually has it, but ensure correct version)
!pip install torch torchvision torchaudio

# Install egnn-pytorch
!pip install egnn-pytorch

# Install other dependencies
!pip install tensorflow  # For TFRecord reading
!pip install tqdm
```

### 3. Mount Google Drive (if using Option A)

```python
from google.colab import drive
drive.mount('/content/drive')

# Set your project path
PROJECT_PATH = '/content/drive/MyDrive/MLproject2'  # Adjust to your path
import os
os.chdir(PROJECT_PATH)
```

### 4. Run Training with GPU

The code is already GPU-ready! Just make sure to:
- Set `--device=cuda` in the training command
- Colab will automatically use the free GPU (T4 or V100)

### 5. Training Command

```python
# In Colab notebook cell
!cd egnn-pytorch && python train_egnn.py \
  --data_dir="../data/deforming_plate" \
  --checkpoint_dir="../checkpoints/egnn" \
  --dataset_fraction=0.1 \
  --num_epochs=10 \
  --batch_size=1 \
  --learning_rate=1e-4 \
  --hidden_dim=128 \
  --depth=4 \
  --device=cuda
```

### 6. Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Advantages of Colab

✅ **Free GPU**: T4 or V100 GPUs (much faster than CPU)  
✅ **No local resources**: Doesn't use your computer's resources  
✅ **Easy sharing**: Can share notebooks with results  
✅ **Pre-installed packages**: Many packages already available  

## Disadvantages

⚠️ **Session limits**: Free tier has time limits (~12 hours)  
⚠️ **Data upload**: Need to upload dataset to Drive or Colab  
⚠️ **Disconnections**: May disconnect if idle  

## Quick Start Notebook

Here's a complete Colab notebook you can use:

```python
# Cell 1: Install dependencies
!pip install egnn-pytorch tensorflow tqdm

# Cell 2: Setup paths (adjust to your setup)
import os
from google.colab import drive

# Mount drive (if using Google Drive)
drive.mount('/content/drive')
PROJECT_PATH = '/content/drive/MyDrive/MLproject2'  # CHANGE THIS
os.chdir(PROJECT_PATH)

# Cell 3: Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Cell 4: Run training
!cd egnn-pytorch && python train_egnn.py \
  --data_dir="../data/deforming_plate" \
  --checkpoint_dir="../checkpoints/egnn" \
  --dataset_fraction=0.1 \
  --num_epochs=10 \
  --device=cuda \
  --hidden_dim=128 \
  --depth=4

# Cell 5: Run evaluation
!cd egnn-pytorch && python evaluate_egnn.py \
  --checkpoint="../checkpoints/egnn/best_model.pt" \
  --data_dir="../data/deforming_plate" \
  --output_dir="../results/egnn" \
  --num_trajectories=10 \
  --device=cuda
```

## Tips

1. **Save checkpoints frequently**: Colab sessions can disconnect
2. **Download results**: Download checkpoints and results before session ends
3. **Use persistent storage**: Consider using Google Drive for data/checkpoints
4. **Monitor GPU usage**: Use `nvidia-smi` to check GPU utilization

