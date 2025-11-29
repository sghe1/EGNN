"""
Helper script to upload data to Colab using various methods.
Run this in a Colab notebook cell.
"""

import os
import zipfile
from pathlib import Path

# ============================================================
# METHOD 1: Direct File Upload (Best for small-medium files)
# ============================================================
def upload_files_direct():
    """
    Upload files directly to Colab using files.upload()
    Best for files < 100MB each
    """
    from google.colab import files
    
    print("=" * 60)
    print("METHOD 1: Direct File Upload")
    print("=" * 60)
    print("This will open a file picker. Select your TFRecord files and meta.json")
    print("Files will be uploaded to the current directory")
    print()
    
    uploaded = files.upload()
    
    # Create data directory structure
    os.makedirs('data/deforming_plate', exist_ok=True)
    
    # Move uploaded files to correct location
    for filename in uploaded.keys():
        if filename.endswith('.tfrecord') or filename == 'meta.json':
            dest = f'data/deforming_plate/{filename}'
            os.rename(filename, dest)
            print(f"✓ Moved {filename} to {dest}")
    
    print("\n✓ Upload complete!")
    return uploaded


# ============================================================
# METHOD 2: Upload as ZIP (Best for multiple files)
# ============================================================
def upload_zip_and_extract():
    """
    Upload a zip file containing the dataset and extract it.
    Best for organizing multiple files together.
    """
    from google.colab import files
    
    print("=" * 60)
    print("METHOD 2: Upload ZIP and Extract")
    print("=" * 60)
    print("1. Create a zip file of your data/deforming_plate folder")
    print("2. Upload it here")
    print()
    
    uploaded = files.upload()
    
    # Find the zip file
    zip_filename = None
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            zip_filename = filename
            break
    
    if zip_filename:
        # Extract to data directory
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('data/')
        print(f"✓ Extracted {zip_filename} to data/")
        os.remove(zip_filename)  # Clean up
        print("✓ Upload and extraction complete!")
    else:
        print("✗ No zip file found in upload")
    
    return uploaded


# ============================================================
# METHOD 3: Download from URL (if you host it somewhere)
# ============================================================
def download_from_url(url, dest_path='data/deforming_plate'):
    """
    Download dataset from a URL (e.g., Google Drive direct link, Dropbox, etc.)
    """
    import urllib.request
    
    print("=" * 60)
    print("METHOD 3: Download from URL")
    print("=" * 60)
    
    os.makedirs(dest_path, exist_ok=True)
    filename = url.split('/')[-1]
    dest_file = os.path.join(dest_path, filename)
    
    print(f"Downloading from: {url}")
    print(f"Destination: {dest_file}")
    
    urllib.request.urlretrieve(url, dest_file)
    print(f"✓ Downloaded to {dest_file}")
    
    # If it's a zip, extract it
    if filename.endswith('.zip'):
        with zipfile.ZipFile(dest_file, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        os.remove(dest_file)
        print("✓ Extracted zip file")
    
    return dest_file


# ============================================================
# METHOD 4: Use wget/curl (for public URLs)
# ============================================================
def download_with_wget(url, dest_path='data/deforming_plate'):
    """
    Use wget to download (useful for large files)
    """
    import subprocess
    
    print("=" * 60)
    print("METHOD 4: Download with wget")
    print("=" * 60)
    
    os.makedirs(dest_path, exist_ok=True)
    
    cmd = f'wget -P {dest_path} {url}'
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print("✓ Download complete!")


# ============================================================
# METHOD 5: Use gdown (for Google Drive files)
# ============================================================
def download_from_gdrive(file_id, dest_path='data/deforming_plate'):
    """
    Download from Google Drive using file ID (no auth needed for public files)
    
    To get file ID:
    1. Upload your zip to Google Drive
    2. Right-click > Get link > Make it "Anyone with the link"
    3. Extract ID from URL: https://drive.google.com/file/d/FILE_ID/view
    """
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system('pip install gdown')
        import gdown
    
    print("=" * 60)
    print("METHOD 5: Download from Google Drive (gdown)")
    print("=" * 60)
    
    os.makedirs(dest_path, exist_ok=True)
    
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(dest_path, 'deforming_plate.zip')
    
    print(f"Downloading from Google Drive...")
    gdown.download(url, output, quiet=False)
    
    # Extract if zip
    if output.endswith('.zip'):
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        os.remove(output)
        print("✓ Extracted zip file")
    
    print("✓ Download complete!")


# ============================================================
# VERIFY UPLOAD
# ============================================================
def verify_dataset():
    """
    Verify that the dataset is correctly uploaded
    """
    data_dir = Path('data/deforming_plate')
    
    print("=" * 60)
    print("Verifying Dataset")
    print("=" * 60)
    
    required_files = ['train.tfrecord', 'valid.tfrecord', 'test.tfrecord', 'meta.json']
    
    all_present = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {filename} - MISSING")
            all_present = False
    
    if all_present:
        print("\n✓ All required files are present!")
        return True
    else:
        print("\n✗ Some files are missing!")
        return False


# ============================================================
# MAIN USAGE EXAMPLES
# ============================================================
if __name__ == "__main__":
    print("""
    ============================================================
    Colab Data Upload Helper
    ============================================================
    
    Choose a method based on your file sizes:
    
    1. Small files (<100MB each): Use upload_files_direct()
    2. Multiple files: Create zip, use upload_zip_and_extract()
    3. Public URL: Use download_from_url() or download_with_wget()
    4. Google Drive (public): Use download_from_gdrive()
    
    Example usage in Colab:
    
    # Method 1: Direct upload
    upload_files_direct()
    
    # Method 2: ZIP upload
    upload_zip_and_extract()
    
    # Method 3: From URL
    download_from_url('https://example.com/data.zip')
    
    # Method 4: Google Drive (replace FILE_ID)
    download_from_gdrive('YOUR_FILE_ID_HERE')
    
    # Verify
    verify_dataset()
    """)

