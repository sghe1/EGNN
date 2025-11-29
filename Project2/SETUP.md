# MeshGraphNet Setup for Deforming Plate

## Compatibility Issue

MeshGraphNet requires Python 3.6-3.8 and TensorFlow 2.x with dm-sonnet<2. 
The current Python 3.12 environment may cause segmentation faults.

## Recommended Setup

### Option 1: Use Python 3.8 (Recommended)

```bash
# Create a new conda environment with Python 3.8
conda create -n meshgraphnet python=3.8 -y
conda activate meshgraphnet

# Install dependencies
cd Project2/deepmind-research/meshgraphnets
pip install -r requirements.txt

# Run the script
cd ../..
./run_deforming_plate.sh
```

### Option 2: Use Python 3.9 or 3.10

```bash
# Create a new conda environment
conda create -n meshgraphnet python=3.9 -y
conda activate meshgraphnet

# Install dependencies
cd Project2/deepmind-research/meshgraphnets
pip install -r requirements.txt
```

### Option 3: Use Virtual Environment

```bash
# Create virtual environment with Python 3.8
python3.8 -m venv venv_meshgraphnet
source venv_meshgraphnet/bin/activate  # On macOS/Linux
# or
venv_meshgraphnet\Scripts\activate  # On Windows

# Install dependencies
cd Project2/deepmind-research/meshgraphnets
pip install -r requirements.txt
```

## Running the Model

After setting up the environment:

```bash
cd Project2
./run_deforming_plate.sh
```

Or manually:

```bash
cd Project2/deepmind-research
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m meshgraphnets.run_model \
    --mode=train \
    --model=deforming_plate \
    --checkpoint_dir=../checkpoints/deforming_plate \
    --dataset_dir=../../data/deforming_plate \
    --num_training_steps=1000000
```

## Troubleshooting

If you encounter `ModuleNotFoundError: No module named 'sonnet'`:
- Make sure you're in the correct conda/virtual environment
- Verify dm-sonnet is installed: `pip list | grep sonnet`
- Reinstall: `pip install "dm-sonnet<2"`

If you encounter segmentation faults:
- This is likely due to Python 3.12 incompatibility
- Use Python 3.8-3.10 instead

