# explore_data.py
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Note: We implement our own data loading function below


def parse_example(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                     for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out


def load_sample_data(data_path, split='train', num_samples=1):
    """Load sample data from TFRecord files."""
    # Check if file exists
    tfrecord_path = os.path.join(data_path, split+'.tfrecord')
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
    
    # Check file size
    file_size = os.path.getsize(tfrecord_path)
    print(f"Loading from {tfrecord_path} (size: {file_size / (1024*1024):.2f} MB)")
    
    with open(os.path.join(data_path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    
    # Create dataset
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(lambda x: parse_example(x, meta), num_parallel_calls=1)
    ds = ds.take(num_samples)
    
    samples = []
    with tf.Session() as sess:
        # Initialize iterator using the recommended approach
        iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        next_element = iterator.get_next()
        
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        try:
            count = 0
            while count < num_samples:
                print(f"  Loading sample {count + 1}...")
                sample = sess.run(next_element)
                # Convert to numpy and remove batch dimension if present
                sample_np = {}
                for k, v in sample.items():
                    # Handle different shapes - remove batch dimension if it's 1
                    if isinstance(v, np.ndarray):
                        if len(v.shape) > 0 and v.shape[0] == 1:
                            # Remove batch dimension
                            sample_np[k] = v[0]
                        else:
                            sample_np[k] = v
                    else:
                        sample_np[k] = v
                samples.append(sample_np)
                count += 1
                print(f"  Successfully loaded sample {count}")
        except tf.errors.OutOfRangeError:
            if len(samples) == 0:
                raise ValueError(
                    f"No samples found in {tfrecord_path}. The dataset might be empty. "
                    f"File size: {file_size} bytes. "
                    f"Please check if the file contains valid TFRecord data."
                )
            print(f"  Warning: Only loaded {len(samples)} sample(s) instead of {num_samples}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error loading data from {tfrecord_path}: {e}")
    
    print(f"Successfully loaded {len(samples)} sample(s)")
    return samples, meta


def print_statistics(sample, meta):
    """Print detailed statistics about a sample."""
    print("\n" + "=" * 60)
    print("SAMPLE STATISTICS")
    print("=" * 60)
    
    for key, value in sample.items():
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")
        print(f"  Min: {np.min(value):.6f}" if np.issubdtype(value.dtype, np.floating) else f"  Min: {np.min(value)}")
        print(f"  Max: {np.max(value):.6f}" if np.issubdtype(value.dtype, np.floating) else f"  Max: {np.max(value)}")
        if np.issubdtype(value.dtype, np.floating):
            print(f"  Mean: {np.mean(value):.6f}")
            print(f"  Std: {np.std(value):.6f}")
        
        if key == 'world_pos':
            if len(value.shape) == 3:
                # Shape: [time_steps, num_nodes, 3]
                print(f"  Position range (x): [{np.min(value[:, :, 0]):.4f}, {np.max(value[:, :, 0]):.4f}]")
                print(f"  Position range (y): [{np.min(value[:, :, 1]):.4f}, {np.max(value[:, :, 1]):.4f}]")
                print(f"  Position range (z): [{np.min(value[:, :, 2]):.4f}, {np.max(value[:, :, 2]):.4f}]")
            elif len(value.shape) == 2:
                # Shape: [num_nodes, 3] - single time step
                print(f"  Position range (x): [{np.min(value[:, 0]):.4f}, {np.max(value[:, 0]):.4f}]")
                print(f"  Position range (y): [{np.min(value[:, 1]):.4f}, {np.max(value[:, 1]):.4f}]")
                print(f"  Position range (z): [{np.min(value[:, 2]):.4f}, {np.max(value[:, 2]):.4f}]")
        elif key == 'stress':
            print(f"  Stress range: [{np.min(value):.4f}, {np.max(value):.4f}]")
        elif key == 'cells':
            if len(value.shape) >= 2:
                print(f"  Number of cells: {value.shape[-2] if len(value.shape) > 2 else value.shape[0]}")
            else:
                print(f"  Number of cells: {len(value)}")
        elif key == 'node_type':
            unique_types, counts = np.unique(value, return_counts=True)
            print(f"  Node types: {dict(zip(unique_types, counts))}")


def visualize_mesh_3d(sample, time_step=0, save_path=None):
    """Visualize the mesh in 3D at a given time step."""
    # Handle different shapes for mesh_pos
    mesh_pos = sample['mesh_pos']
    if len(mesh_pos.shape) == 3:
        mesh_pos = mesh_pos[0]  # Remove batch/time dimension if present
    elif len(mesh_pos.shape) > 3:
        mesh_pos = mesh_pos[0, 0]  # Remove batch and time dimensions
    
    # Handle different shapes for world_pos
    world_pos = sample['world_pos']
    if len(world_pos.shape) == 3:
        world_pos = world_pos[time_step]  # [time_steps, num_nodes, 3]
    elif len(world_pos.shape) == 2:
        world_pos = world_pos  # Already [num_nodes, 3], single time step
    elif len(world_pos.shape) == 4:
        world_pos = world_pos[0, time_step]  # [batch, time_steps, num_nodes, 3]
    
    # Handle different shapes for cells
    cells = sample['cells']
    if len(cells.shape) == 3:
        cells = cells[0]  # Remove batch dimension
    elif len(cells.shape) == 2:
        cells = cells  # Already correct shape
    
    # Handle different shapes for stress
    stress = sample['stress']
    if len(stress.shape) == 3:
        stress = stress[time_step]  # [time_steps, num_nodes, 1]
    elif len(stress.shape) == 2:
        stress = stress  # Already [num_nodes, 1] or [num_nodes]
    elif len(stress.shape) == 4:
        stress = stress[0, time_step]  # [batch, time_steps, num_nodes, 1]
    
    # Flatten stress if needed
    if len(stress.shape) > 1:
        stress = stress.flatten()
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Initial mesh
    ax1 = fig.add_subplot(131, projection='3d')
    if len(cells.shape) >= 2 and cells.shape[-1] == 4:  # Tetrahedral cells
        # Limit for performance
        num_cells_to_plot = min(1000, len(cells))
        for cell in cells[:num_cells_to_plot]:
            try:
                vertices = mesh_pos[cell]
                if len(vertices.shape) == 2 and vertices.shape[1] == 3:
                    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.3)
            except (IndexError, ValueError):
                pass  # Skip invalid cells
    # Fallback to scatter plot
    ax1.scatter(mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2], 
               c='blue', s=1, alpha=0.5)
    ax1.set_title(f'Initial Mesh (t=0)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Deformed mesh at time step
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(world_pos[:, 0], world_pos[:, 1], world_pos[:, 2],
                         c=stress.flatten(), cmap='viridis', s=2)
    ax2.set_title(f'Deformed Mesh (t={time_step})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax2, label='Stress')
    
    # Plot 3: Deformation magnitude
    ax3 = fig.add_subplot(133, projection='3d')
    deformation = np.linalg.norm(world_pos - mesh_pos, axis=1)
    scatter2 = ax3.scatter(world_pos[:, 0], world_pos[:, 1], world_pos[:, 2],
                          c=deformation, cmap='hot', s=2)
    ax3.set_title(f'Deformation Magnitude (t={time_step})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax3, label='Deformation')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_trajectory_statistics(sample):
    """Plot time series of key quantities."""
    world_pos = sample['world_pos']
    stress = sample['stress']
    mesh_pos = sample['mesh_pos']
    
    # Handle shape for mesh_pos
    if len(mesh_pos.shape) == 3:
        mesh_pos = mesh_pos[0]
    elif len(mesh_pos.shape) > 3:
        mesh_pos = mesh_pos[0, 0]
    
    # Handle shape for world_pos
    if len(world_pos.shape) == 2:
        # Single time step - can't plot trajectory
        print("Warning: world_pos is 2D (single time step), skipping trajectory plots")
        return
    elif len(world_pos.shape) == 4:
        world_pos = world_pos[0]  # Remove batch dimension
    
    # Handle shape for stress
    if len(stress.shape) == 2:
        # Single time step - can't plot trajectory
        print("Warning: stress is 2D (single time step), skipping trajectory plots")
        return
    elif len(stress.shape) == 4:
        stress = stress[0]  # Remove batch dimension
    
    # Flatten stress if it has extra dimensions
    if len(stress.shape) == 3:
        stress = stress.squeeze()  # Remove last dimension if it's 1
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average stress over time
    if len(stress.shape) == 2:
        avg_stress = np.mean(stress, axis=1)
    else:
        avg_stress = np.mean(stress)
        avg_stress = np.array([avg_stress])  # Single value
    axes[0, 0].plot(avg_stress)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Average Stress')
    axes[0, 0].set_title('Average Stress Over Time')
    axes[0, 0].grid(True)
    
    # Plot 2: Max stress over time
    if len(stress.shape) == 2:
        max_stress = np.max(stress, axis=1)
    else:
        max_stress = np.max(stress)
        max_stress = np.array([max_stress])  # Single value
    axes[0, 1].plot(max_stress)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Max Stress')
    axes[0, 1].set_title('Maximum Stress Over Time')
    axes[0, 1].grid(True)
    
    # Plot 3: Center of mass displacement
    if len(world_pos.shape) == 3:
        com_initial = np.mean(world_pos[0], axis=0)
        com_displacement = np.array([np.mean(world_pos[t], axis=0) - com_initial 
                                     for t in range(len(world_pos))])
        axes[1, 0].plot(com_displacement[:, 0], label='X')
        axes[1, 0].plot(com_displacement[:, 1], label='Y')
        axes[1, 0].plot(com_displacement[:, 2], label='Z')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Displacement')
        axes[1, 0].set_title('Center of Mass Displacement')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Total deformation energy (approximate)
        deformation = np.array([np.linalg.norm(world_pos[t] - mesh_pos, axis=1) 
                               for t in range(len(world_pos))])
        total_deformation = np.sum(deformation, axis=1)
        axes[1, 1].plot(total_deformation)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Total Deformation')
        axes[1, 1].set_title('Total Deformation Over Time')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def compare_trajectories(samples, max_trajectories=5):
    """Compare multiple trajectories side-by-side.
    
    Args:
        samples: List of sample dictionaries
        max_trajectories: Maximum number of trajectories to compare (for readability)
    """
    if len(samples) == 0:
        return
    
    # Limit number of trajectories for readability
    samples_to_plot = samples[:max_trajectories]
    
    # Check if we have time series data
    first_sample = samples_to_plot[0]
    world_pos = first_sample['world_pos']
    stress = first_sample['stress']
    
    if len(world_pos.shape) != 3:
        print("Warning: Cannot compare trajectories - data is not time series")
        return
    
    # Handle stress shape
    if len(stress.shape) == 3:
        stress = stress.squeeze()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average stress comparison
    for idx, sample in enumerate(samples_to_plot):
        stress_data = sample['stress']
        if len(stress_data.shape) == 3:
            stress_data = stress_data.squeeze()
        if len(stress_data.shape) == 2:
            avg_stress = np.mean(stress_data, axis=1)
        else:
            avg_stress = np.array([np.mean(stress_data)])
        axes[0, 0].plot(avg_stress, label=f'Trajectory {idx + 1}', alpha=0.7)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Average Stress')
    axes[0, 0].set_title('Average Stress Over Time (Comparison)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Max stress comparison
    for idx, sample in enumerate(samples_to_plot):
        stress_data = sample['stress']
        if len(stress_data.shape) == 3:
            stress_data = stress_data.squeeze()
        if len(stress_data.shape) == 2:
            max_stress = np.max(stress_data, axis=1)
        else:
            max_stress = np.array([np.max(stress_data)])
        axes[0, 1].plot(max_stress, label=f'Trajectory {idx + 1}', alpha=0.7)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Max Stress')
    axes[0, 1].set_title('Maximum Stress Over Time (Comparison)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Center of mass displacement comparison (X component)
    for idx, sample in enumerate(samples_to_plot):
        world_pos_data = sample['world_pos']
        if len(world_pos_data.shape) == 3:
            com_initial = np.mean(world_pos_data[0], axis=0)
            com_displacement = np.array([np.mean(world_pos_data[t], axis=0) - com_initial 
                                       for t in range(len(world_pos_data))])
            axes[1, 0].plot(com_displacement[:, 0], label=f'Trajectory {idx + 1} (X)', alpha=0.7)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Displacement (X)')
    axes[1, 0].set_title('Center of Mass X-Displacement (Comparison)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Total deformation comparison
    for idx, sample in enumerate(samples_to_plot):
        world_pos_data = sample['world_pos']
        mesh_pos_data = sample['mesh_pos']
        if len(mesh_pos_data.shape) == 3:
            mesh_pos_data = mesh_pos_data[0]
        elif len(mesh_pos_data.shape) > 3:
            mesh_pos_data = mesh_pos_data[0, 0]
        
        if len(world_pos_data.shape) == 3:
            deformation = np.array([np.linalg.norm(world_pos_data[t] - mesh_pos_data, axis=1) 
                                   for t in range(len(world_pos_data))])
            total_deformation = np.sum(deformation, axis=1)
            axes[1, 1].plot(total_deformation, label=f'Trajectory {idx + 1}', alpha=0.7)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Total Deformation')
    axes[1, 1].set_title('Total Deformation Over Time (Comparison)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def explore_deforming_plate_data(data_dir, dataset_name='deforming_plate', split='train', 
                                 num_trajectories=1, visualize=True):
    """Explore a meshgraphnets dataset structure.
    
    Args:
        data_dir: Base directory containing datasets
        dataset_name: Name of the dataset to explore (e.g., 'deforming_plate', 'flag_simple', etc.)
        split: Which split to load ('train', 'valid', 'test')
        num_trajectories: Number of trajectories to load and visualize (default: 1)
        visualize: Whether to generate visualizations
    
    Note: The datasets use Lagrangian coordinates - mesh nodes move with the material.
    The 'world_pos' field contains the Lagrangian positions over time.
    """
    
    data_path = Path(data_dir) / dataset_name
    
    # 1. Read metadata
    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    print("Note: This dataset uses LAGRANGIAN coordinates.")
    print("      Mesh nodes move with the material (world_pos tracks node positions over time).")
    print("=" * 60)
    with open(data_path / 'meta.json', 'r') as f:
        metadata = json.load(f)
    
    print(json.dumps(metadata, indent=2))
    
    # 2. Load sample data
    print("\n" + "=" * 60)
    print("LOADING SAMPLE DATA")
    print("=" * 60)
    print(f"Loading {num_trajectories} trajectory/trajectories from {split} split...")
    
    try:
        samples, meta = load_sample_data(str(data_path), split=split, num_samples=num_trajectories)
        if len(samples) == 0:
            print("No samples found!")
            return
        
        print(f"Successfully loaded {len(samples)} trajectory/trajectories!")
        
        # 3. Print statistics for each trajectory
        for idx, sample in enumerate(samples):
            print(f"\n{'='*60}")
            print(f"TRAJECTORY {idx + 1} STATISTICS")
            print(f"{'='*60}")
            print_statistics(sample, meta)
        
        # 4. Visualizations
        if visualize:
            print("\n" + "=" * 60)
            print("GENERATING VISUALIZATIONS")
            print("=" * 60)
            
            # If multiple trajectories, show comparison first
            if len(samples) > 1:
                print(f"\nComparing {len(samples)} trajectories...")
                compare_trajectories(samples)
            
            # Plot trajectory statistics for each trajectory
            for idx, sample in enumerate(samples):
                print(f"\nPlotting trajectory {idx + 1} statistics...")
                plot_trajectory_statistics(sample)
            
            # Visualize mesh at different time steps for each trajectory
            for idx, sample in enumerate(samples):
                print(f"\nVisualizing trajectory {idx + 1} mesh at different time steps...")
                world_pos = sample['world_pos']
                if len(world_pos.shape) == 3:
                    # 3D: [time_steps, num_nodes, 3]
                    num_time_steps = world_pos.shape[0]
                    time_steps = [0, num_time_steps//4, num_time_steps//2, max(0, num_time_steps-1)]
                    for t in time_steps:
                        print(f"  Trajectory {idx + 1}, Time step {t}...")
                        visualize_mesh_3d(sample, time_step=t, save_path=None)
                else:
                    # 2D: [num_nodes, 3] - single time step
                    print(f"  Trajectory {idx + 1}, Single time step (showing t=0)...")
                    visualize_mesh_3d(sample, time_step=0)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nTrying alternative loading method...")
        import traceback
        traceback.print_exc()


# Run exploration
if __name__ == '__main__':
    # Use the correct path where we downloaded the data
    data_dir = '/Users/tommasobasile/Desktop/SCRIVANIA/MA3/ML/ML_project/MLproject2/data'
    
    # Specify which dataset to explore
    # Available datasets: deforming_plate, flag_simple, cylinder_flow, airfoil, etc.
    # All meshgraphnets datasets use Lagrangian coordinates (mesh nodes move with material)
    dataset_name = 'deforming_plate'  # Change this to use a different dataset
    
    # Number of trajectories to load and visualize
    num_trajectories = 1  # Change this to load multiple trajectories (e.g., 3, 5, 10)
    
    # Explore training data
    print(f"Exploring {dataset_name} dataset (Lagrangian coordinates)")
    print(f"Loading {num_trajectories} trajectory/trajectories...")
    explore_deforming_plate_data(data_dir, dataset_name=dataset_name, split='train', 
                                 num_trajectories=num_trajectories, visualize=True)
    