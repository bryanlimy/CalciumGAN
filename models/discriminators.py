from .registry import discriminator_register as register

import numpy as np
import tensorflow as tf

from .utils import get_activation_fn


@register
def mlp(hparams):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')

  num_units = hparams.signal_shape[-1]

  outputs = tf.keras.layers.Dense(num_units // 3)(inputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Dense(num_units // 6)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')


@register
def conv1d(hparams):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=3, strides=2, padding='causal')(inputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(0.3)(outputs)

  outputs = tf.keras.layers.Conv1D(
      filters=128, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(0.3)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')


@register
def rnn(hparams):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')

  num_units = hparams.signal_shape[-1]

  outputs = tf.keras.layers.GRU(
      num_units,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=False,
      time_major=False)(inputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
