from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import calculate_input_config, activation_fn


@register('rnn')
def get_rnn(hparams):
  # TODO make it works for NWC
  return generator(hparams), discriminator(hparams)


def generator(hparams):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  shape, num_units = calculate_input_config(
      num_neurons=hparams.num_neurons, noise_dim=hparams.noise_dim)
  signal_length = hparams.signal_shape[-1]

  outputs = layers.Dense(num_units, use_bias=False)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  num_units = hparams.signal_shape[-1]

  outputs = layers.GRU(
      signal_length // 9,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.GRU(
      signal_length // 6,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.GRU(
      signal_length // 3,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.Dense(num_units)(outputs)

  if hparams.normalize:
    outputs = activation_fn('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def discriminator(hparams):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')

  signal_length = hparams.signal_shape[-1]

  outputs = layers.GRU(
      signal_length // 3,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(inputs)

  outputs = layers.GRU(
      signal_length // 6,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.GRU(
      signal_length // 9,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=False,
      time_major=False)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
