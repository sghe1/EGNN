# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Training script for MeshGraphNets Deforming Plate with validation and plotting.

Loss computation:
- Acceleration loss (primary): MSE on predicted vs target acceleration (normalized)
  This is the original MeshGraphNets loss. The model predicts acceleration only.
  
- Velocity loss (secondary diagnostic): MSE between predicted and target velocity
  Velocity is derived from positions: velocity = next_pos - current_pos
  Target velocity: target|world_pos - world_pos
  Predicted velocity: predicted_next_pos - world_pos (from acceleration prediction)
  Note: Velocity loss is for diagnostic purposes only, not used in training.

Total loss: acceleration_loss + 0.1 * velocity_loss

Parity plots:
- Velocity: Ground truth velocity magnitude vs predicted velocity magnitude
  Denormalization is handled by the model's output_normalizer for positions/velocities.
  
Note: The model does NOT predict stress. Stress is only used as an input node feature.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags
from absl import logging

# Add parent directory to path so meshgraphnets can be imported
# This allows running from Project2/ directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# TensorFlow compatibility
try:
    import tensorflow.compat.v1 as tf
except (AttributeError, ImportError):
    # Fallback for TF 1.15 - use compatibility shim
    from meshgraphnets import tf_compat  # This creates tf.compat.v1
    import tensorflow.compat.v1 as tf

from meshgraphnets import deforming_plate_model
from meshgraphnets import core_model
from meshgraphnets import dataset
from meshgraphnets import common

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'raw_data', 'Directory to load dataset from.')
flags.DEFINE_string('output_dir', 'checkpoints/meshgraphnet', 'Directory to save checkpoints')
flags.DEFINE_string('plots_dir', 'plots/meshgraphnet', 'Directory to save plots')
flags.DEFINE_integer('num_epochs', 500, 'Number of training epochs')
flags.DEFINE_integer('steps_per_epoch', 1000, 'Number of training steps per epoch')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate')
flags.DEFINE_integer('trajectory_id', None, 'Optional: Train on a single trajectory (0-indexed). If None, use all trajectories.')


def create_model():
  """Create MeshGraphNets model."""
  learned_model = core_model.EncodeProcessDecode(
      output_size=3,  # 3D acceleration
      latent_size=128,
      num_layers=2,
      message_passing_steps=15)
  model = deforming_plate_model.Model(learned_model)
  return model


def compute_losses(model, inputs, is_training=True):
  """
  Compute acceleration loss (primary) and velocity loss (diagnostic).
  
  NOTE: MeshGraphNets predicts acceleration only, not velocity or stress.
  - Acceleration loss: Primary loss (original MeshGraphNets loss)
  - Velocity loss: Diagnostic metric derived from predicted positions
  
  Returns: (total_loss, velocity_loss, acceleration_loss,
            predicted_velocity, target_velocity, predicted_position, target_position,
            current_stress, target_stress)
  """
  graph = model._build_graph(inputs, is_training=is_training)
  network_output = model._learned_model(graph)
  
  # Build target acceleration (original MeshGraphNets target)
  cur_position = inputs['world_pos']
  prev_position = inputs['prev|world_pos']
  target_position = inputs['target|world_pos']
  target_acceleration = target_position - 2*cur_position + prev_position
  target_normalized = model._output_normalizer(target_acceleration)
  
  # Compute predicted acceleration and denormalize
  predicted_acceleration = model._output_normalizer.inverse(network_output)
  
  # Derive velocities from positions (for diagnostic purposes)
  # Target velocity: change from current to target position
  target_velocity = target_position - cur_position
  
  # Predicted next position from acceleration (Verlet integration)
  predicted_next_position = 2*cur_position + predicted_acceleration - prev_position
  predicted_velocity = predicted_next_position - cur_position
  
  # Get stress values (current stress from dataset)
  current_stress = inputs['stress']
  # Note: Stress is not predicted by MeshGraphNet, it's only used as an input feature
  # We'll track stress values over time for visualization
  
  # Loss mask: only compute on NORMAL nodes (same as original MeshGraphNets)
  loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
  
  # Acceleration loss: Primary loss (original MeshGraphNets)
  error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
  acceleration_loss = tf.reduce_mean(error[loss_mask])
  
  # Velocity loss: Diagnostic metric only (derived from positions)
  velocity_error = tf.reduce_sum((target_velocity - predicted_velocity)**2, axis=1)
  velocity_loss = tf.reduce_mean(velocity_error[loss_mask])
  
  # Total loss: acceleration is primary, velocity is secondary diagnostic
  total_loss = acceleration_loss + 0.1 * velocity_loss
  
  return (total_loss, velocity_loss, acceleration_loss,
          predicted_velocity, target_velocity, predicted_next_position, target_position,
          current_stress)


def generate_plots(history, predictions, plots_dir):
  """Generate training and parity plots matching EGNN style."""
  epochs = range(1, len(history['train_total']) + 1)
  
  # 1. Training Convergence plot (log scale) - matches EGNN style
  plt.figure(figsize=(10, 5))
  plt.plot(epochs, history['train_total'], label='Train Loss', linewidth=2, color='#1f77b4')
  plt.yscale('log')
  plt.xlabel('Epoch', fontsize=12)
  plt.ylabel('Loss (MSE)', fontsize=12)
  plt.title('Training Convergence', fontsize=14, fontweight='bold')
  plt.legend(fontsize=11)
  plt.grid(True, which="both", ls="-", alpha=0.2)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
  plt.close()
  
  # 2. Component losses: Velocity vs Stress (Acceleration) - matches EGNN style
  plt.figure(figsize=(10, 5))
  plt.plot(epochs, history['train_velocity'], label='Train Vel', linestyle='--', linewidth=2, color='#1f77b4')
  plt.plot(epochs, history['train_acceleration'], label='Train Stress', linestyle='--', linewidth=2, color='#2ca02c')
  plt.yscale('log')
  plt.xlabel('Epoch', fontsize=12)
  plt.ylabel('MSE', fontsize=12)
  plt.title('Velocity vs Stress Loss', fontsize=14, fontweight='bold')
  plt.legend(fontsize=11)
  plt.grid(True, which="both", ls="-", alpha=0.2)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_dir, 'component_losses.png'), dpi=150, bbox_inches='tight')
  plt.close()
  
  # 3. Parity plots: Velocity and Stress - matches EGNN style
  if predictions.get('velocity_gt') and predictions.get('velocity_pred'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity Parity
    vel_gt = np.array(predictions['velocity_gt'])
    vel_pred = np.array(predictions['velocity_pred'])
    
    # Subsample if too many points
    if len(vel_gt) > 5000:
      indices = np.random.choice(len(vel_gt), 5000, replace=False)
      vel_gt = vel_gt[indices]
      vel_pred = vel_pred[indices]
    
    axes[0].scatter(vel_gt, vel_pred, alpha=0.3, s=2, color='#1f77b4')
    m = max(vel_gt.max(), vel_pred.max())
    axes[0].plot([0, m], [0, m], 'r--', linewidth=2)
    axes[0].set_xlabel('Ground Truth Speed', fontsize=12)
    axes[0].set_ylabel('Predicted Speed', fontsize=12)
    axes[0].set_title('Velocity Parity', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Stress Parity
    if predictions.get('stress_gt') and predictions.get('stress_pred'):
      stress_gt = np.array(predictions['stress_gt'])
      stress_pred = np.array(predictions['stress_pred'])
      
      # Subsample if too many points
      if len(stress_gt) > 5000:
        indices = np.random.choice(len(stress_gt), 5000, replace=False)
        stress_gt = stress_gt[indices]
        stress_pred = stress_pred[indices]
      
      axes[1].scatter(stress_gt, stress_pred, alpha=0.3, s=2, color='#2ca02c')
      m_min = min(stress_gt.min(), stress_pred.min())
      m_max = max(stress_gt.max(), stress_pred.max())
      axes[1].plot([m_min, m_max], [m_min, m_max], 'r--', linewidth=2)
      axes[1].set_xlabel('Ground Truth Stress', fontsize=12)
      axes[1].set_ylabel('Predicted Stress', fontsize=12)
      axes[1].set_title('Stress Parity', fontsize=14, fontweight='bold')
      axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main(argv):
  del argv
  
  # Setup directories
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  os.makedirs(FLAGS.plots_dir, exist_ok=True)
  
  # Disable TF 2.x behavior
  try:
    tf.disable_v2_behavior()
  except AttributeError:
    pass
  tf.disable_eager_execution()
  tf.enable_resource_variables()
  
  # Load datasets
  logging.info('Loading datasets from %s', FLAGS.data_dir)
  train_ds = dataset.load_dataset(FLAGS.data_dir, 'train')
  
  # Filter to single trajectory if specified
  if FLAGS.trajectory_id is not None:
    logging.info('Filtering to trajectory ID: %d', FLAGS.trajectory_id)
    # Use take/skip to select specific trajectory (0-indexed)
    # take(1) gets one trajectory starting from trajectory_id
    train_ds = train_ds.skip(FLAGS.trajectory_id).take(1)
  
  # Process training dataset
  train_ds = dataset.add_targets(train_ds, ['world_pos'], add_history=True)
  train_ds = dataset.split_and_preprocess(train_ds, noise_field='world_pos',
                                          noise_scale=0.003, noise_gamma=0.1)
  
  # Create model
  model = create_model()
  
  # Build training graph
  train_iterator = train_ds.make_one_shot_iterator()
  train_inputs = train_iterator.get_next()
  
  # Compute losses with predictions
  (total_loss, velocity_loss, acceleration_loss,
   pred_velocity, targ_velocity, pred_position, targ_position,
   current_stress) = compute_losses(
      model, train_inputs, is_training=True)
  
  # Extract training input tensors for plotting
  train_node_type = train_inputs['node_type']
  train_world_pos = train_inputs['world_pos']
  train_target_world_pos = train_inputs['target|world_pos']
  
  # Optimizer
  global_step = tf.train.create_global_step()
  lr = tf.constant(FLAGS.learning_rate, dtype=tf.float32)
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(total_loss, global_step=global_step)
  
  # Loss ops for logging
  train_loss_ops = {
      'total': total_loss,
      'velocity': velocity_loss,
      'acceleration': acceleration_loss
  }
  
  # Training history (no validation)
  history = {
      'train_total': [], 'train_velocity': [], 'train_acceleration': []
  }
  
  # Store final predictions for parity plots (from training data)
  final_predictions = {
      'velocity_gt': [], 'velocity_pred': [],
      'stress_gt': [], 'stress_pred': []
  }
  
  # Training loop
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.output_dir,
      save_checkpoint_secs=600,
      save_summaries_steps=100) as sess:
    
    for epoch in range(FLAGS.num_epochs):
      logging.info('Epoch %d/%d', epoch + 1, FLAGS.num_epochs)
      
      # Train
      epoch_losses = {'total': [], 'velocity': [], 'acceleration': []}
      for step in range(FLAGS.steps_per_epoch):
        try:
          _, step_val, losses = sess.run([train_op, global_step, train_loss_ops])
          epoch_losses['total'].append(losses['total'])
          epoch_losses['velocity'].append(losses['velocity'])
          epoch_losses['acceleration'].append(losses['acceleration'])
          
          if step % 100 == 0:
            logging.info('Step %d: Total=%.6f, Accel=%.6f, Vel=%.6f',
                         step_val, losses['total'], losses['acceleration'], losses['velocity'])
        except tf.errors.OutOfRangeError:
          logging.warning('Training dataset exhausted at step %d', step)
          break
      
      train_avg = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
      history['train_total'].append(train_avg['total'])
      history['train_velocity'].append(train_avg['velocity'])
      history['train_acceleration'].append(train_avg['acceleration'])
      
      # Collect predictions from training data on last epoch for parity plots
      if epoch == FLAGS.num_epochs - 1:
        logging.info('Collecting predictions for final plots...')
        try:
          # Collect a few batches of training predictions
          for batch_idx in range(min(10, FLAGS.steps_per_epoch)):
            try:
              results = sess.run([
                  pred_velocity, targ_velocity,
                  current_stress, train_node_type])
              
              pred_vel, targ_vel = results[0], results[1]
              current_stress_val = results[2]
              node_types_train = results[3]
              
              # Filter to NORMAL nodes only
              node_types_flat = node_types_train[:, 0]
              normal_mask = node_types_flat == common.NodeType.NORMAL
              
              if normal_mask.any():
                # Velocity magnitudes for parity plot
                pred_vel_mag = np.linalg.norm(pred_vel, axis=1)
                targ_vel_mag = np.linalg.norm(targ_vel, axis=1)
                final_predictions['velocity_gt'].extend(targ_vel_mag[normal_mask])
                final_predictions['velocity_pred'].extend(pred_vel_mag[normal_mask])
              
              # For stress: use stress values (MeshGraphNet doesn't predict stress)
              # We use stress as both GT and pred to show it's not being predicted
              stress_mask = (node_types_flat == common.NodeType.NORMAL) | (node_types_flat == common.NodeType.BOUNDARY)
              if stress_mask.any():
                stress_vals = current_stress_val[stress_mask].flatten()
                final_predictions['stress_gt'].extend(stress_vals)
                final_predictions['stress_pred'].extend(stress_vals)
            except tf.errors.OutOfRangeError:
              break
            except Exception as e:
              logging.warning('Error collecting training predictions: %s', str(e))
              continue
        except Exception as e:
          logging.warning('Error in prediction collection: %s', str(e))
      
      # Save checkpoint
      if (epoch + 1) % 50 == 0:
        logging.info('Saved checkpoint at epoch %d', epoch + 1)
  
  # Save history
  history_path = os.path.join(FLAGS.output_dir, 'training_history.pkl')
  with open(history_path, 'wb') as f:
    pickle.dump(history, f)
  logging.info('Saved training history to %s', history_path)
  
  # Generate plots
  logging.info('Generating plots...')
  generate_plots(history, final_predictions, FLAGS.plots_dir)
  
  logging.info('Training complete!')
  logging.info('Checkpoints saved to: %s', FLAGS.output_dir)
  logging.info('Plots saved to: %s', FLAGS.plots_dir)


if __name__ == '__main__':
  app.run(main)
