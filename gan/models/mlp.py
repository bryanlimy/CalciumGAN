from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn


@register('mlp')
def get_models(hparams):
  return generator(hparams), discriminator(hparams)


def generator(hparams, units=64):
  shape = (hparams.sequence_length, hparams.noise_dim)
  noise_size = int(np.prod(shape))

  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  outputs = layers.Dense(noise_size)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  # Layer 1
  outputs = layers.Dense(units)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 2
  outputs = layers.Dense(units * 2)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 3
  outputs = layers.Dense(units * 3)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(hparams.num_channels)(outputs)

  if hparams.normalize:
    outputs = activation_fn('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def discriminator(hparams, units=64):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')

  # Layer 1
  outputs = layers.Dense(units * 4)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 2
  outputs = layers.Dense(units * 3)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 3
  outputs = layers.Dense(units * 2)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 4
  outputs = layers.Dense(units)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
