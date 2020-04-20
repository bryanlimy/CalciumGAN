from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn


@register('rnn')
def get_rnn(hparams):
  return generator(hparams), discriminator(hparams)


def calculate_input_config(output_shape, noise_dim, upscale, num_layers):
  w = output_shape[0] / (upscale**(num_layers - 1))
  assert w.is_integer()
  return (int(w), noise_dim)


def generator(hparams, units=64, upscale=4):
  shape = calculate_input_config(
      hparams.signal_shape, hparams.noise_dim, upscale, num_layers=2)
  noise_size = int(np.prod(shape))

  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  outputs = layers.Dense(noise_size)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  # Layer 1
  outputs = layers.GRU(
      units,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.UpSampling1D(upscale)(outputs)

  # Layer 2
  outputs = layers.GRU(
      hparams.num_channels,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.Dense(hparams.num_channels)(outputs)

  if hparams.normalize:
    outputs = activation_fn('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def discriminator(hparams, units=64, pool_size=4):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')

  outputs = layers.GRU(
      units * 2,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(inputs)

  outputs = layers.AveragePooling1D(pool_size=pool_size)(outputs)

  outputs = layers.GRU(
      units,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
