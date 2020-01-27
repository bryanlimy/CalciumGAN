from .utils import get_activation_fn
from .registry import discriminator_register as register

import numpy as np
import tensorflow as tf


@register
def mlp(hparams):
  signals = tf.keras.Input(shape=hparams.signal_shape, name='signals')

  outputs = tf.keras.layers.Flatten()(signals)

  outputs = tf.keras.layers.Dense(512)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(256)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=signals, outputs=outputs, name='discriminator')


@register
def conv1d(hparams):
  signals = tf.keras.Input(hparams.signal_shape, name='signals')

  outputs = tf.keras.layers.Conv1D(
      filters=512, kernel_size=3, strides=2, padding='causal')(signals)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(0.3)(outputs)

  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(0.3)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=signals, outputs=outputs, name='discriminator')


@register
def rnn(hparams):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='signals')

  outputs = tf.keras.layers.LSTM(
      512,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(inputs)
  outputs = tf.keras.layers.LSTM(
      256,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)
  outputs = tf.keras.layers.LSTM(
      128,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      time_major=False)(outputs)

  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
