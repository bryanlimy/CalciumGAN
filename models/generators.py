from .registry import generator_register as register

import numpy as np
import tensorflow as tf

from .utils import get_activation_fn, Conv1DTranspose


@register
def mlp(hparams):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  num_units = hparams.signal_shape[-1]

  outputs = tf.keras.layers.Dense(
      hparams.num_neurons * hparams.noise_dim, use_bias=False)(inputs)
  outputs = get_activation_fn(hparams.activation)(outputs)

  outputs = tf.keras.layers.Reshape((hparams.num_neurons,
                                     hparams.noise_dim))(outputs)

  outputs = tf.keras.layers.Dense(num_units // 6)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Dense(num_units // 3)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)

  outputs = tf.keras.layers.Dense(num_units)(outputs)
  outputs = tf.keras.layers.Reshape(hparams.signal_shape)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.sigmoid(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


@register
def conv1d(hparams):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  outputs = tf.keras.layers.Dense(
      hparams.num_neurons * hparams.noise_dim, use_bias=False)(inputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Reshape((hparams.num_neurons,
                                     hparams.noise_dim))(outputs)

  outputs = Conv1DTranspose(filters=128, kernel_size=4, strides=3)(outputs)
  outputs = tf.keras.layers.BatchNormalization()(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)

  outputs = Conv1DTranspose(filters=256, kernel_size=6, strides=2)(outputs)
  outputs = tf.keras.layers.BatchNormalization()(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)

  outputs = tf.keras.layers.Dense(hparams.signal_shape[-1])(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.sigmoid(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


@register
def rnn(hparams):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  outputs = tf.keras.layers.Dense(
      hparams.num_neurons * hparams.noise_dim, use_bias=False)(inputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Reshape((hparams.num_neurons,
                                     hparams.noise_dim))(outputs)

  num_units = hparams.signal_shape[-1]

  outputs = tf.keras.layers.GRU(
      num_units,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = tf.keras.layers.Dense(num_units)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.sigmoid(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')
