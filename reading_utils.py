import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
import tree

def prepare_data_from_tfds(data_path='data/train.tfrecord', is_rollout=False, batch_size=2, metadata = None):
    
    def prepare_inputs(tensor_dict):
        pos = tensor_dict['position']
        pos = tf.transpose(pos, perm=[1, 0, 2])
        target_position = pos[:, -1]
        tensor_dict['position'] = pos[:, :-1]
        num_particles = tf.shape(pos)[0]
        tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
        if 'step_context' in tensor_dict:
            tensor_dict['step_context'] = tensor_dict['step_context'][-2]
            tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
        return tensor_dict, target_position

    def batch_concat(dataset, batch_size):
        windowed_ds = dataset.window(batch_size)
        initial_state = tree.map_structure(lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),dataset.element_spec)
        def reduce_window(initial_state, ds):
            return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))
        return windowed_ds.map(lambda *x: tree.map_structure(reduce_window, initial_state, x))

    def prepare_rollout_inputs(context, features):
        out_dict = {**context}
        pos = tf.transpose(features['position'], [1, 0, 2])
        target_position = pos[:, -1]
        out_dict['position'] = pos[:, :-1]
        out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
        if 'step_context' in features:
            out_dict['step_context'] = features['step_context']
        out_dict['is_trajectory'] = tf.constant([True], tf.bool)
        return out_dict, target_position

    #metadata = _read_metadata('data/Goop')
    ds = tf.data.TFRecordDataset([data_path])
    ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
    if is_rollout:
        ds = ds.map(prepare_rollout_inputs)
    else:    
        split_with_window = functools.partial(
            split_trajectory,
            window_length=6 + 1)
        ds = ds.flat_map(split_with_window)
        ds = ds.map(prepare_inputs)
        ds = ds.repeat()
        ds = ds.shuffle(512)
        ds = batch_concat(ds, batch_size)
    ds = tfds.as_numpy(ds)
    return ds

# Lint as: python3
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
"""Utilities for reading open sourced Learning Complex Physics data."""

import functools
import numpy as np
import tensorflow as tf

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = tf.convert_to_tensor(np.array(out))
  return out


def parse_serialized_simulation_example(example_proto, metadata):
  """Parses a serialized simulation tf.SequenceExample.
  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.
  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.
  """
  if 'context_mean' in metadata:
    feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
  else:
    feature_description = _FEATURE_DESCRIPTION
  context, parsed_features = tf.io.parse_single_sequence_example(
      example_proto,
      context_features=_CONTEXT_FEATURES,
      sequence_features=feature_description)
  for feature_key, item in parsed_features.items():
    convert_fn = functools.partial(
        convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
    parsed_features[feature_key] = tf.py_function(
        convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

  # There is an extra frame at the beginning so we can calculate pos change
  # for all frames used in the paper.
  position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

  # Reshape positions to correct dim:
  parsed_features['position'] = tf.reshape(parsed_features['position'],
                                           position_shape)
  # Set correct shapes of the remaining tensors.
  sequence_length = metadata['sequence_length'] + 1
  if 'context_mean' in metadata:
    context_feat_len = len(metadata['context_mean'])
    parsed_features['step_context'] = tf.reshape(
        parsed_features['step_context'],
        [sequence_length, context_feat_len])
  # Decode particle type explicitly
  context['particle_type'] = tf.py_function(
      functools.partial(convert_fn, encoded_dtype=np.int64),
      inp=[context['particle_type'].values],
      Tout=[tf.int64])
  context['particle_type'] = tf.reshape(context['particle_type'], [-1])
  return context, parsed_features


def split_trajectory(context, features, window_length=7):
  """Splits trajectory into sliding windows."""
  # Our strategy is to make sure all the leading dimensions are the same size,
  # then we can use from_tensor_slices.

  trajectory_length = features['position'].get_shape().as_list()[0]

  # We then stack window_length position changes so the final
  # trajectory length will be - window_length +1 (the 1 to make sure we get
  # the last split).
  input_trajectory_length = trajectory_length - window_length + 1

  model_input_features = {}
  # Prepare the context features per step.
  model_input_features['particle_type'] = tf.tile(
      tf.expand_dims(context['particle_type'], axis=0),
      [input_trajectory_length, 1])

  if 'step_context' in features:
    global_stack = []
    for idx in range(input_trajectory_length):
      global_stack.append(features['step_context'][idx:idx + window_length])
    model_input_features['step_context'] = tf.stack(global_stack)

  pos_stack = []
  for idx in range(input_trajectory_length):
    pos_stack.append(features['position'][idx:idx + window_length])
  # Get the corresponding positions
  model_input_features['position'] = tf.stack(pos_stack)

  return tf.data.Dataset.from_tensor_slices(model_input_features)