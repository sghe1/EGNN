# Uploading Dataset to Colab - Complete Guide

Since Google Drive authorization isn't working, here are **5 alternative methods** to get your data into Colab:

---

## 🔧 Method 0: Create Smaller Subset First (REQUIRED!)

**Your dataset is too large!** `train.tfrecord` is 9.2GB. Create a subset first:

### On Your Local Machine:

```bash
# Create subset with 100 trajectories (~900MB instead of 9.2GB)
bash scripts/create_subset_for_colab.sh 100

# Or create with 50 trajectories (~450MB)
bash scripts/create_subset_for_colab.sh 50
```

This creates `data/deforming_plate_colab/` with:
- First N trajectories from `train.tfrecord`
- Full `test.tfrecord` and `valid.tfrecord` (smaller files)
- `meta.json`

Then create zip:
```bash
cd /path/to/MLproject2
zip -r deforming_plate_colab.zip data/deforming_plate_colab/
```

**Note**: Since you're using `--dataset_fraction=0.1` anyway, 100 trajectories is plenty for training!

---

## 📊 Check Your Dataset Size

Run this locally to see file sizes:
```bash
cd data/deforming_plate
du -sh *.tfrecord meta.json
```

This helps you choose the best upload method.

---

## 🚀 Method 1: Direct File Upload (Best for <100MB per file)

**When to use**: If each TFRecord file is <100MB

### Steps:

1. **In Colab notebook, run:**
```python
from google.colab import files
import os

# Create directory
os.makedirs('data/deforming_plate', exist_ok=True)

# Upload files (will open file picker)
uploaded = files.upload()

# Move files to correct location
for filename in uploaded.keys():
    if filename.endswith('.tfrecord') or filename == 'meta.json':
        os.rename(filename, f'data/deforming_plate/{filename}')
        print(f"✓ Uploaded {filename}")
```

2. **Select your files** from the file picker that appears
3. **Wait for upload** (shows progress bar)

**Pros**: Simple, no external services  
**Cons**: Slow for large files, may timeout

---

## 📦 Method 2: Upload as ZIP (Recommended for multiple files)

**When to use**: Best overall method - works for any size, organizes files together

### Steps:

1. **On your local machine, create a zip:**
```bash
cd /path/to/MLproject2
zip -r deforming_plate_data.zip data/deforming_plate/
```

2. **In Colab notebook:**
```python
from google.colab import files
import zipfile
import os

# Upload zip file
print("Upload your deforming_plate_data.zip file...")
uploaded = files.upload()

# Find and extract zip
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(filename)  # Clean up
        print(f"✓ Extracted {filename}")

# Verify
import os
files = os.listdir('data/deforming_plate')
print("Files:", files)
```

**Pros**: Fast, reliable, handles multiple files  
**Cons**: Need to zip first

---

## 🌐 Method 3: Upload to GitHub and Clone

**When to use**: If you have GitHub, or want version control

### Steps:

1. **Create a private GitHub repo** (or use existing)
2. **Upload data folder** to the repo
3. **In Colab:**
```python
# Clone repo (replace with your repo URL)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
# Or if private, use: !git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git

# Navigate to project
import os
os.chdir('YOUR_REPO')
```

**Pros**: Version control, easy updates  
**Cons**: Need GitHub account, large files need Git LFS

---

## ☁️ Method 4: Use Dropbox/OneDrive Direct Link

**When to use**: If you have cloud storage

### Steps:

1. **Upload zip to Dropbox/OneDrive**
2. **Get direct download link** (make it public)
3. **In Colab:**
```python
import urllib.request
import zipfile
import os

# Download from Dropbox/OneDrive
url = 'YOUR_DIRECT_DOWNLOAD_LINK_HERE'
dest = 'data.zip'

print("Downloading...")
urllib.request.urlretrieve(url, dest)

# Extract
with zipfile.ZipFile(dest, 'r') as zip_ref:
    zip_ref.extractall('.')

os.remove(dest)
print("✓ Done!")
```

**Dropbox direct link format**:  
`https://www.dropbox.com/s/FILE_ID/filename.zip?dl=1` (add `?dl=1` at end)

**OneDrive direct link**:  
Share file > Get link > Change `/view` to `/download` in URL

---

## 📥 Method 5: Google Drive via gdown (No Auth Needed!)

**When to use**: If you can upload to Google Drive (even if mount doesn't work)

### Steps:

1. **Upload your zip to Google Drive**
2. **Right-click > Get link > Make it "Anyone with the link"**
3. **Extract file ID from URL:**
   - URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`
   - Copy the `FILE_ID_HERE` part

4. **In Colab:**
```python
!pip install gdown
import gdown
import zipfile
import os

# Download using file ID
file_id = 'YOUR_FILE_ID_HERE'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'data.zip'

gdown.download(url, output, quiet=False)

# Extract
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('.')

os.remove(output)
print("✓ Done!")
```

**Pros**: Works even if Drive mount fails, fast  
**Cons**: Need to make file public (or use service account)

---

## ✅ Verify Your Upload

After any method, verify the dataset:

```python
from pathlib import Path

data_dir = Path('data/deforming_plate')
required = ['train.tfrecord', 'valid.tfrecord', 'test.tfrecord', 'meta.json']

print("Checking dataset...")
for f in required:
    path = data_dir / f
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ {f} ({size_mb:.2f} MB)")
    else:
        print(f"✗ {f} - MISSING")
```

---

## 🎯 Recommended Workflow

**For your large dataset, I recommend:**

1. **Create subset (Local):**
```bash
cd /Users/tommasobasile/Desktop/SCRIVANIA/MA3/ML/ML_project/MLproject2
bash scripts/create_subset_for_colab.sh 100  # Creates ~900MB subset
zip -r deforming_plate_colab.zip data/deforming_plate_colab/
```

2. **Upload to Colab using Method 5 (gdown with Google Drive):**
   - Upload `deforming_plate_colab.zip` to Google Drive
   - Get shareable link (make public)
   - Extract file ID
   - Use gdown in Colab (see Method 5)

**OR use Method 2 (ZIP upload) if the subset is <2GB:**

2. **Colab:**
```python
from google.colab import files
import zipfile
import os

uploaded = files.upload()
for f in uploaded.keys():
    if f.endswith('.zip'):
        zipfile.ZipFile(f, 'r').extractall('.')
        os.remove(f)
        print(f"✓ Extracted {f}")
```

3. **Verify:**
```python
import os
print(os.listdir('data/deforming_plate'))
```

---

## 🆘 Troubleshooting

**"File too large" error:**
- Use Method 2 (ZIP) or Method 5 (gdown)
- Or split into smaller chunks

**"Upload timeout":**
- Use Method 4 or 5 (external hosting)
- Or upload files one at a time

**"Permission denied":**
- Check file permissions in Colab: `!ls -la data/deforming_plate`

---

## 📝 Complete Colab Notebook Template

```python
# Cell 1: Install dependencies
!pip install egnn-pytorch tensorflow tqdm

# Cell 2: Upload data (Method 5 - Google Drive with gdown)
!pip install gdown
import gdown
import zipfile
import os

# Replace FILE_ID with your Google Drive file ID
file_id = 'YOUR_FILE_ID_HERE'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, 'data.zip', quiet=False)

# Extract
with zipfile.ZipFile('data.zip', 'r') as z:
    z.extractall('.')
os.remove('data.zip')

# Cell 3: Verify (adjust path if using subset)
print(os.listdir('data/deforming_plate_colab'))  # or 'data/deforming_plate'

# Cell 4: Run training (adjust data_dir if using subset)
!cd egnn-pytorch && python train_egnn.py \
  --data_dir="../data/deforming_plate_colab" \
  --checkpoint_dir="../checkpoints/egnn" \
  --dataset_fraction=0.1 \
  --num_epochs=10 \
  --device=cuda
```

---

**Need help?** Check file sizes first, then choose the method that fits your situation!

