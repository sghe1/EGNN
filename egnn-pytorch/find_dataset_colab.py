#!/usr/bin/env python3
"""
Find where your dataset files are located in Colab.
Run this to locate your uploaded files.
"""

import os
from pathlib import Path

def find_dataset_files():
    """Search for dataset files in common locations."""
    
    print("=" * 60)
    print("Searching for dataset files...")
    print("=" * 60)
    print(f"Current working directory: {os.getcwd()}")
    print()
    
    # Common locations to search
    search_paths = [
        '.',  # Current directory
        '..',  # Parent directory
        '../data',
        '../data/deforming_plate',
        '../data/deforming_plate_colab',
        'data',
        'data/deforming_plate',
        'data/deforming_plate_colab',
        '/content',  # Colab root
        '/content/data',
        '/content/data/deforming_plate',
    ]
    
    # Files to look for
    target_files = ['train.tfrecord', 'valid.tfrecord', 'test.tfrecord', 'meta.json']
    
    found_files = {}
    
    # Search recursively from current directory
    print("Searching for files...")
    print()
    
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            # Check if files are directly here
            for filename in target_files:
                filepath = path / filename
                if filepath.exists():
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    found_files[filename] = {
                        'path': str(filepath.absolute()),
                        'size_mb': size_mb,
                        'relative': str(filepath)
                    }
    
    # Also do a recursive search from current directory
    print("Recursive search from current directory...")
    current = Path('.')
    for root, dirs, files in os.walk(current):
        for filename in target_files:
            if filename in files:
                filepath = Path(root) / filename
                if filename not in found_files:  # Don't overwrite if already found
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    found_files[filename] = {
                        'path': str(filepath.absolute()),
                        'size_mb': size_mb,
                        'relative': str(filepath)
                    }
    
    # Report findings
    if found_files:
        print("=" * 60)
        print("✓ FOUND FILES:")
        print("=" * 60)
        
        # Group by directory
        by_dir = {}
        for filename, info in found_files.items():
            dir_path = str(Path(info['path']).parent)
            if dir_path not in by_dir:
                by_dir[dir_path] = []
            by_dir[dir_path].append((filename, info))
        
        for dir_path, files in by_dir.items():
            print(f"\nDirectory: {dir_path}")
            print("-" * 60)
            for filename, info in files:
                print(f"  ✓ {filename:20s} {info['size_mb']:8.2f} MB")
                print(f"    Relative: {info['relative']}")
        
        print()
        print("=" * 60)
        print("RECOMMENDED DATA_DIR:")
        print("=" * 60)
        
        # Find the directory that has all required files
        complete_dirs = {}
        for filename, info in found_files.items():
            dir_path = str(Path(info['path']).parent)
            if dir_path not in complete_dirs:
                complete_dirs[dir_path] = []
            complete_dirs[dir_path].append(filename)
        
        for dir_path, files in complete_dirs.items():
            if len(files) == len(target_files):
                print(f"✓ Complete dataset found at:")
                print(f"  {dir_path}")
                print()
                print("Use this path in training:")
                rel_path = Path(dir_path).relative_to(Path.cwd())
                print(f"  --data_dir=\"{rel_path}\"")
                break
        else:
            # No complete directory found
            print("⚠️  No single directory contains all files.")
            print("Files are scattered. You may need to:")
            print("1. Re-upload the dataset as a zip and extract it")
            print("2. Or manually organize the files into one directory")
            
            # Show what's missing where
            print()
            print("Files by location:")
            for dir_path, files in complete_dirs.items():
                missing = set(target_files) - set(files)
                print(f"  {dir_path}:")
                print(f"    Has: {', '.join(files)}")
                if missing:
                    print(f"    Missing: {', '.join(missing)}")
    else:
        print("=" * 60)
        print("✗ NO FILES FOUND")
        print("=" * 60)
        print()
        print("The dataset files were not found. Please:")
        print("1. Check that you uploaded/extracted the dataset")
        print("2. Verify the files are in the Colab environment")
        print("3. Try uploading again using one of these methods:")
        print()
        print("   Method 1: Direct upload")
        print("   ```python")
        print("   from google.colab import files")
        print("   uploaded = files.upload()")
        print("   ```")
        print()
        print("   Method 2: Google Drive (gdown)")
        print("   See: egnn-pytorch/colab_data_upload.md")
        print()
        print("   Method 3: Check if files are in /content/")
        print("   ```python")
        print("   import os")
        print("   print(os.listdir('/content'))")
        print("   ```")
    
    print()
    print("=" * 60)
    print("Current directory contents:")
    print("=" * 60)
    try:
        for item in sorted(Path('.').iterdir()):
            if item.is_dir():
                print(f"  📁 {item}/")
            else:
                size = item.stat().st_size / (1024 * 1024) if item.is_file() else 0
                print(f"  📄 {item.name} ({size:.2f} MB)" if size > 0 else f"  📄 {item.name}")
    except Exception as e:
        print(f"  Error listing directory: {e}")
    
    print()
    print("Parent directory contents:")
    print("=" * 60)
    try:
        parent = Path('..')
        if parent.exists():
            for item in sorted(parent.iterdir()):
                if item.is_dir():
                    print(f"  📁 {item}/")
                else:
                    size = item.stat().st_size / (1024 * 1024) if item.is_file() else 0
                    print(f"  📄 {item.name} ({size:.2f} MB)" if size > 0 else f"  📄 {item.name}")
    except Exception as e:
        print(f"  Error listing parent: {e}")

if __name__ == "__main__":
    find_dataset_files()

