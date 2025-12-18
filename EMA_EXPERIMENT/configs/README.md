# Ablation Study Configurations

This directory contains example configuration files for running ablation studies.

## Available Configs

- **`config_baseline.yaml`**: Baseline configuration with standard hyperparameters
- **`config_no_dropout.yaml`**: Ablation without dropout (dropout_gnn=0, dropout_mlps_final=0)
- **`config_small_model.yaml`**: Ablation with smaller model (hid_gnn_layer_dim=64, hid_mlp_dim=128)
- **`config_amp.yaml`**: Ablation with Automatic Mixed Precision enabled (amp=True)

## Usage

Run a specific config:
```bash
cd pytorch_model
python train.py --config ../configs/config_baseline.yaml
```

Run multiple configs sequentially:
```bash
for config in configs/*.yaml; do
    python train.py --config "$config"
done
```

Or use SLURM array job (see main README.md for `run_ablation.sbatch` example).

## Creating Your Own Configs

1. Copy an existing config:
   ```bash
   cp config_baseline.yaml config_my_experiment.yaml
   ```

2. Modify hyperparameters in `config_my_experiment.yaml`

3. **Important**: Set a unique `model_path_out` to avoid overwriting outputs:
   ```yaml
   training:
     model_path_out: "model_out/my_experiment/"
   ```

4. Run:
   ```bash
   python train.py --config configs/config_my_experiment.yaml
   ```
