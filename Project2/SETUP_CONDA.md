# MeshGraphNet Conda Environment Setup

This guide helps you set up a conda environment with compatible versions for running MeshGraphNet training.

## Quick Start

1. **Run the setup script:**
   ```bash
   cd Project2
   bash setup_conda_env.sh
   ```

2. **Activate the environment:**
   ```bash
   conda activate meshgraphnet
   ```

3. **Run training:**
   ```bash
   bash run_meshgraphnets.sh
   ```

## Manual Setup (Alternative)

If the automated script doesn't work, you can set up manually:

### 1. Create Conda Environment

```bash
conda create -n meshgraphnet python=3.11 -y
conda activate meshgraphnet
```

### 2. Install Dependencies

**For Apple Silicon (M1/M2 Macs):**
```bash
pip install tensorflow-macos==2.13.1 tensorflow-metal==1.1.0
pip install "tensorflow-probability>=0.18.0,<0.20.0"
```

**For Intel/x86_64:**
```bash
pip install "tensorflow>=2.13.0,<2.16.0"
pip install "tensorflow-probability>=0.18.0,<0.20.0"
```

### 3. Install DeepMind Sonnet

**Important:** dm-sonnet requires tensorflow-probability < 0.9.0, but TensorFlow 2.13+ requires newer versions. Try this workaround:

```bash
# First, try with compatible tensorflow-probability
pip install "tensorflow-probability>=0.8.0,<0.9.0" --force-reinstall
pip install "dm-sonnet<2"

# If that causes TensorFlow issues, try:
pip install "tensorflow-probability>=0.18.0,<0.20.0" --force-reinstall
pip install "dm-sonnet<2" --no-deps
```

### 4. Install Other Dependencies

```bash
pip install -r requirements_conda.txt
```

### 5. Set Environment Variable

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

To make this permanent, add it to your shell profile (`~/.zshrc` or `~/.bashrc`):
```bash
echo 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' >> ~/.zshrc
```

## Troubleshooting

### Issue: dm-sonnet installation fails

**Solution:** dm-sonnet has strict version requirements. Try installing in this order:
```bash
pip install "tensorflow-probability>=0.8.0,<0.9.0"
pip install "dm-sonnet<2"
pip install "tensorflow-probability>=0.18.0,<0.20.0" --force-reinstall
```

### Issue: TensorFlow version conflicts

**Solution:** Make sure you're using Python 3.10 or 3.11. Python 3.12 doesn't support TensorFlow < 2.16.0.

### Issue: Segmentation fault when running

**Solution:** 
1. Make sure you're using the conda environment
2. Check TensorFlow version: `python -c "import tensorflow as tf; print(tf.__version__)"`
3. Should be 2.13.x, 2.14.x, or 2.15.x (but < 2.16.0)

### Issue: Protobuf errors

**Solution:** Set the environment variable:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Environment Details

- **Python Version:** 3.10 or 3.11 (3.11 recommended)
- **TensorFlow:** 2.13.0 - 2.15.x (< 2.16.0 required)
- **TensorFlow Probability:** 0.18.0 - 0.19.x (or 0.8.x for dm-sonnet compatibility)
- **dm-sonnet:** < 2.0

## Verification

After setup, verify the installation:

```bash
conda activate meshgraphnet
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import tensorflow_probability as tfp; print('TFP:', tfp.__version__)"
python -c "import sonnet as snt; print('Sonnet: OK')"
```

All imports should succeed without errors.

## Running Training

Once the environment is set up:

```bash
cd Project2
conda activate meshgraphnet
bash run_meshgraphnets.sh
```

This will train MeshGraphNet on trajectory 0 for 500 epochs and generate plots for:
- Velocity errors over time
- Stress magnitude over time  
- Position and velocity predictions
- Training convergence curves

