#!/usr/bin/env python3
"""
Quick script to verify the dataset is correctly uploaded in Colab.
Run this in a Colab cell to check your dataset.
"""

import os
from pathlib import Path
from tfrecord.reader import tfrecord_loader
import json

def verify_dataset(data_dir='../data/deforming_plate'):
    """Verify the dataset files exist and contain data."""
    
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        print(f"✗ Directory not found: {data_path}")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Try: data_dir='../data/deforming_plate_colab' if you used the subset")
        return False
    
    print(f"✓ Directory found: {data_path}")
    print()
    
    # Check required files
    required_files = {
        'train.tfrecord': False,
        'valid.tfrecord': False,
        'test.tfrecord': False,
        'meta.json': False
    }
    
    print("Checking files...")
    for filename in required_files.keys():
        filepath = data_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            required_files[filename] = True
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {filename} - MISSING")
    
    print()
    
    # Check meta.json
    meta_path = data_path / 'meta.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print("✓ meta.json is valid")
            print(f"  Trajectory length: {meta.get('trajectory_length', 'N/A')}")
        except Exception as e:
            print(f"✗ Error reading meta.json: {e}")
            return False
    else:
        print("✗ meta.json not found")
        return False
    
    print()
    
    # Count trajectories in train.tfrecord
    train_path = data_path / 'train.tfrecord'
    if train_path.exists():
        print("Counting trajectories in train.tfrecord...")
        try:
            count = 0
            loader = tfrecord_loader(str(train_path), index_path=None)
            for record in loader:
                count += 1
                if count >= 10:  # Just check first 10 to verify it works
                    print(f"  ✓ Found at least {count} trajectories (checking first 10 only)")
                    break
            
            if count == 0:
                print("  ✗ No trajectories found in train.tfrecord!")
                print("  The file might be empty or corrupted.")
                return False
            elif count < 10:
                print(f"  ✓ Found {count} trajectories")
            
        except Exception as e:
            print(f"  ✗ Error reading train.tfrecord: {e}")
            print("  The file might be corrupted or in wrong format.")
            return False
    else:
        print("✗ train.tfrecord not found")
        return False
    
    print()
    print("=" * 60)
    print("✓ Dataset verification complete!")
    print("=" * 60)
    print()
    print("You can now run training with:")
    print(f"  --data_dir=\"{data_dir}\"")
    
    return True

if __name__ == "__main__":
    import sys
    
    # First, try to find the files
    print("First, let's find where your files are located...\n")
    try:
        from find_dataset_colab import find_dataset_files
        find_dataset_files()
        print("\n" + "=" * 60)
        print("Now verifying the dataset...")
        print("=" * 60 + "\n")
    except ImportError:
        print("Note: Run find_dataset_colab.py first to locate files\n")
    
    # Try different possible paths
    possible_paths = [
        '../data/deforming_plate',
        '../data/deforming_plate_colab',
        'data/deforming_plate',
        'data/deforming_plate_colab',
        '/content/data/deforming_plate',
        '/content/data/deforming_plate_colab',
    ]
    
    if len(sys.argv) > 1:
        verify_dataset(sys.argv[1])
    else:
        # Try each path
        found = False
        for path in possible_paths:
            if Path(path).exists():
                print(f"Found dataset at: {path}\n")
                if verify_dataset(path):
                    found = True
                    break
        
        if not found:
            print("Could not find dataset. Run find_dataset_colab.py to locate files.")
            print("Or specify the path manually: python verify_dataset_colab.py <path>")

