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
"""Core learned graph net model."""

import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class GraphNetBlock(snt.Module):
  """Multi-Edge Interaction Network with residual connections."""

  def __init__(self, model_fn, name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._model_fn = model_fn
    # Cache MLPs for edges and nodes separately since they have different input sizes
    self._edge_mlp_cache = {}
    self._node_mlp_cache = {}

  def _get_edge_mlp(self, input_size):
    """Get or create MLP for edge features with given input size."""
    cache_key = f"edge_{input_size}"
    if cache_key not in self._edge_mlp_cache:
      self._edge_mlp_cache[cache_key] = self._model_fn()
    return self._edge_mlp_cache[cache_key]

  def _get_node_mlp(self, input_size):
    """Get or create MLP for node features with given input size."""
    cache_key = f"node_{input_size}"
    if cache_key not in self._node_mlp_cache:
      self._node_mlp_cache[cache_key] = self._model_fn()
    return self._node_mlp_cache[cache_key]

  def _update_edge_features(self, node_features, edge_set):
    """Aggregrates node features, and applies edge function."""
    sender_features = tf.gather(node_features, edge_set.senders)
    receiver_features = tf.gather(node_features, edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    concat_features = tf.concat(features, axis=-1)
    input_size = tf.shape(concat_features)[-1]
    with tf.variable_scope(edge_set.name+'_edge_fn'):
      # Use the MLP, it will infer input size on first call
      return self._model_fn()(concat_features)

  def _update_node_features(self, node_features, edge_sets):
    """Aggregrates edge features, and applies node function."""
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes))
    concat_features = tf.concat(features, axis=-1)
    with tf.variable_scope('node_fn'):
      # Use the MLP, it will infer input size on first call
      return self._model_fn()(concat_features)

  def _build(self, graph):
    """Applies GraphNetBlock and returns updated MultiGraph."""

    # apply edge functions
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated_features = self._update_edge_features(graph.node_features,
                                                    edge_set)
      new_edge_sets.append(edge_set._replace(features=updated_features))

    # apply node function
    new_node_features = self._update_node_features(graph.node_features,
                                                   new_edge_sets)

    # add residual connections
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)

  def __call__(self, graph):
    """Make GraphNetBlock callable - routes to _build."""
    return self._build(graph)


class EncodeProcessDecode(snt.Module):
  """Encode-Process-Decode GraphNet model."""

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    # Cache MLPs to avoid creating new modules inside while_loops
    self._mlp_cache = {}

  def _make_mlp(self, output_size, layer_norm=True, cache_key=None, input_size=None):
    """Builds an MLP, with caching to avoid recreating in while_loops."""
    if cache_key is None:
      cache_key = f"mlp_{output_size}_{layer_norm}_{input_size}"
    
    if cache_key not in self._mlp_cache:
      # MLP input size: if specified, use it; otherwise use latent_size
      # For process MLPs: edges need 384, nodes need 256
      # For encoder/decoder: use latent_size (input size matches output of previous layer)
      if input_size is not None:
        widths = [input_size] + [self._latent_size] * self._num_layers + [output_size]
      else:
        widths = [self._latent_size] * self._num_layers + [output_size]
      network = snt.nets.MLP(widths, activate_final=False)
      if layer_norm:
        # dm-sonnet 2.0 LayerNorm requires axis, create_scale, create_offset
        network = snt.Sequential([network, snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
      self._mlp_cache[cache_key] = network
    return self._mlp_cache[cache_key]

  def _encoder(self, graph):
    """Encodes node and edge features into latent features."""
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size, cache_key='encoder_node')(graph.node_features)
      new_edges_sets = []
      for i, edge_set in enumerate(graph.edge_sets):
        latent = self._make_mlp(self._latent_size, cache_key=f'encoder_edge_{i}')(edge_set.features)
        new_edges_sets.append(edge_set._replace(features=latent))
    return MultiGraph(node_latents, new_edges_sets)

  def _decoder(self, graph):
    """Decodes node features from graph."""
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp(self._output_size, layer_norm=False, cache_key='decoder')
      return decoder(graph.node_features)

  def _build(self, graph):
    """Encodes and processes a multigraph, and returns node features."""
    latent_graph = self._encoder(graph)
    
    # Create separate MLPs for edges and nodes with correct input sizes
    # Edge: sender (latent) + receiver (latent) + edge (latent) = 3 * latent_size = 384
    # Node: node (latent) + aggregated_edge (latent) = 2 * latent_size = 256
    edge_input_size = 3 * self._latent_size
    node_input_size = 2 * self._latent_size
    
    # Create model_fn that returns the appropriate MLP based on variable scope
    def model_fn():
      scope_name = tf.get_variable_scope().name
      if 'edge_fn' in scope_name:
        # Edge MLP: input 384, output latent_size
        return self._make_mlp(self._latent_size, cache_key='process_edge_mlp', input_size=edge_input_size)
      else:
        # Node MLP: input 256, output latent_size
        return self._make_mlp(self._latent_size, cache_key='process_node_mlp', input_size=node_input_size)
    
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(model_fn)(latent_graph)
    return self._decoder(latent_graph)

  def __call__(self, graph):
    """Make EncodeProcessDecode callable - routes to _build."""
    return self._build(graph)
