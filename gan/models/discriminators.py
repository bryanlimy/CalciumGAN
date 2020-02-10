from .registry import discriminator_register as register

import numpy as np
import tensorflow as tf

from .utils import get_activation_fn


@register
def mlp(hparams):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')

  signal_shape = hparams.signal_shape[-1]

  outputs = tf.keras.layers.Dense(signal_shape)(inputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Dense(signal_shape // 3)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Dense(signal_shape // 6)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')


@register
def conv1d(hparams):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  signal_length = hparams.signal_shape[-1]
  kernel_size, strides = 4, 2

  outputs = tf.keras.layers.Conv1D(
      filters=signal_length, kernel_size=kernel_size, strides=strides)(inputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Conv1D(
      filters=signal_length // 2, kernel_size=kernel_size,
      strides=strides)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Conv1D(
      filters=signal_length // 4, kernel_size=kernel_size,
      strides=strides)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')


@register
def rnn(hparams):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')

  signal_length = hparams.signal_shape[-1]

  outputs = tf.keras.layers.GRU(
      signal_length,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(inputs)

  outputs = tf.keras.layers.GRU(
      signal_length // 2,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = tf.keras.layers.GRU(
      signal_length // 4,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=False,
      time_major=False)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
