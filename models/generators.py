from .utils import get_activation_fn
from .registry import generator_register as register

import numpy as np
import tensorflow as tf


@register
def mlp(hparams):
  inputs = tf.keras.Input(shape=hparams.generator_input_shape, name='inputs')

  outputs = tf.keras.layers.Flatten()(inputs)

  outputs = tf.keras.layers.Dense(512)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(256)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(np.prod(hparams.signal_shape))(outputs)
  outputs = tf.keras.layers.Reshape(hparams.signal_shape)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.sigmoid(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


@register
def conv1d(hparams):
  inputs = tf.keras.Input(shape=hparams.generator_input_shape, name='inputs')

  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=3, strides=2, padding='causal')(inputs)
  outputs = tf.keras.layers.BatchNormalization()(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)

  outputs = tf.keras.layers.Conv1D(
      filters=128, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.BatchNormalization()(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)

  outputs = tf.keras.layers.Dense(np.prod(hparams.signal_shape))(outputs)
  outputs = tf.keras.layers.Reshape(hparams.signal_shape)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.sigmoid(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


@register
def rnn(hparams):
  inputs = tf.keras.Input(shape=hparams.generator_input_shape, name='inputs')

  outputs = tf.keras.layers.GRU(
      512,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(inputs)
  outputs = tf.keras.layers.GRU(
      256,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)
  outputs = tf.keras.layers.GRU(
      hparams.signal_shape[-1],
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.sigmoid(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')
