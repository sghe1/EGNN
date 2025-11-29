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
"""Runs the learner/evaluator."""

import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
# Compatibility shim for TF 1.15
try:
    import tensorflow.compat.v1 as tf
except (AttributeError, ImportError):
    # Fallback for TF 1.15 - use compatibility shim
    from meshgraphnets import tf_compat  # This creates tf.compat.v1
    import tensorflow.compat.v1 as tf
# Only import deforming_plate modules to avoid unnecessary imports
from meshgraphnets import deforming_plate_eval
from meshgraphnets import deforming_plate_model
from meshgraphnets import core_model
from meshgraphnets import dataset


FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'deforming_plate', ['deforming_plate'],
                  'Select model to run (only deforming_plate supported).')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_float('dataset_fraction', 1.0, 'Fraction of dataset to use for training (0.0 to 1.0)')

# Only support deforming_plate model
PARAMETERS = {
    'deforming_plate': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                            size=3, batch=1, model=deforming_plate_model, evaluator=deforming_plate_eval)
}


def learner(model, params):
  """Run a learner job."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
  
  # Limit dataset to specified fraction (limit at trajectory level)
  if FLAGS.dataset_fraction < 1.0:
    # Count total trajectories first to calculate how many to take
    # For deforming_plate, there are typically ~1000 trajectories
    # We'll use a simple approach: take first N trajectories based on fraction
    # Since we can't easily count without iterating, we'll use a reasonable estimate
    # For 10% of ~1000 trajectories = 100 trajectories
    if FLAGS.model == 'deforming_plate':
      # Approximately 1000 trajectories in deforming_plate dataset
      num_trajectories_to_take = int(1000 * FLAGS.dataset_fraction)
      logging.info('Limiting dataset to %d trajectories (%.1f%% of dataset)', 
                   num_trajectories_to_take, FLAGS.dataset_fraction * 100)
      ds = ds.take(num_trajectories_to_take)
    else:
      # For other models, use a similar approach
      logging.warning('Dataset fraction limiting may need adjustment for model: %s', FLAGS.model)
      # Use a conservative estimate - can be adjusted per model
      estimated_trajectories = 1000
      num_trajectories_to_take = int(estimated_trajectories * FLAGS.dataset_fraction)
      ds = ds.take(num_trajectories_to_take)
  
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
                                    noise_scale=params['noise'],
                                    noise_gamma=params['gamma'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()

  loss_op = model.loss(inputs)
  global_step = tf.train.create_global_step()
  # Use a simple learning rate schedule compatible with graph mode
  # Start with 1e-4 and decay by 0.1 every 5e6 steps
  step_float = tf.cast(global_step, tf.float32)
  decay_steps = tf.constant(5e6, dtype=tf.float32)
  decay_rate = tf.constant(0.1, dtype=tf.float32)
  decay_factor = tf.pow(decay_rate, tf.floor(step_float / decay_steps))
  lr = tf.constant(1e-4, dtype=tf.float32) * decay_factor + tf.constant(1e-6, dtype=tf.float32)
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step)
  # Don't train for the first few steps, just accumulate normalization stats
  # Use a smaller normalization period (100 steps) to allow more training
  normalization_steps = 100
  train_op = tf.cond(tf.less(global_step, normalization_steps),
                     lambda: tf.group(tf.assign_add(global_step, 1)),
                     lambda: tf.group(train_op))

  # Add TensorBoard summaries for loss and learning rate
  tf.summary.scalar('loss', loss_op)
  tf.summary.scalar('learning_rate', lr)
  summary_op = tf.summary.merge_all()
  
  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps),
             tf.train.SummarySaverHook(save_steps=100,
                                      output_dir=FLAGS.checkpoint_dir,
                                      summary_op=summary_op)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=600) as sess:

    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])
      if step % 1000 == 0:
        logging.info('Step %d: Loss %g', step, loss)
    logging.info('Training complete.')


def evaluator(model, params):
  """Run a model rollout trajectory."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      scalars.append(scalar_data)
    for key in scalars[0]:
      logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    with open(FLAGS.rollout_path, 'wb') as fp:
      pickle.dump(trajectories, fp)


def main(argv):
  del argv
  # For TensorFlow 2.x with dm-sonnet 2.0
  # Disable eager execution for graph mode compatibility
  try:
    tf.disable_v2_behavior()  # Disable TF 2.x behavior
  except AttributeError:
    pass  # Not available in TF 1.x
  tf.disable_eager_execution()  # Use graph mode
  tf.enable_resource_variables()
  params = PARAMETERS[FLAGS.model]
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=15)
  model = params['model'].Model(learned_model)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  app.run(main)
