from .registry import generator_register as register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import Conv1DTranspose, calculate_input_config


@register
def mlp(hparams):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  shape, num_units = calculate_input_config(
      num_neurons=hparams.num_neurons, noise_dim=hparams.noise_dim)
  signal_length = hparams.signal_shape[-1]

  outputs = layers.Dense(num_units, use_bias=False)(inputs)
  outputs = layers.Activation(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  outputs = layers.Dense(signal_length // 6)(outputs)
  outputs = layers.Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(signal_length // 3)(outputs)
  outputs = layers.Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(signal_length)(outputs)
  outputs = layers.Reshape(hparams.signal_shape)(outputs)

  if hparams.normalize:
    outputs = layers.Activation('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = layers.Activation('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


@register
def conv1d(hparams):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  kernel_size, strides = 4, 2
  shape, num_units = calculate_input_config(
      num_neurons=hparams.num_neurons,
      noise_dim=hparams.noise_dim,
      num_convolution=3,
      kernel_size=kernel_size,
      strides=strides)
  signal_length = hparams.signal_shape[-1]

  outputs = layers.Dense(num_units, use_bias=False)(inputs)
  outputs = layers.Activation(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  outputs = Conv1DTranspose(signal_length // 4, kernel_size, strides)(outputs)
  outputs = layers.BatchNormalization()(outputs)
  outputs = layers.Activation(hparams.activation)(outputs)

  outputs = Conv1DTranspose(signal_length // 2, kernel_size, strides)(outputs)
  outputs = layers.BatchNormalization()(outputs)
  outputs = layers.Activation(hparams.activation)(outputs)

  outputs = Conv1DTranspose(signal_length, kernel_size, strides)(outputs)
  outputs = layers.BatchNormalization()(outputs)
  outputs = layers.Activation(hparams.activation)(outputs)

  outputs = layers.Dense(hparams.signal_shape[-1])(outputs)

  if hparams.normalize:
    outputs = layers.Activation('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = layers.Activation('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


@register
def rnn(hparams):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  shape, num_units = calculate_input_config(
      num_neurons=hparams.num_neurons, noise_dim=hparams.noise_dim)
  signal_length = hparams.signal_shape[-1]

  outputs = layers.Dense(num_units, use_bias=False)(inputs)
  outputs = layers.Activation(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  num_units = hparams.signal_shape[-1]

  outputs = layers.GRU(
      signal_length // 4,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.GRU(
      signal_length // 2,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.GRU(
      signal_length,
      activation=hparams.activation,
      recurrent_initializer='glorot_uniform',
      dropout=hparams.dropout,
      return_sequences=True,
      time_major=False)(outputs)

  outputs = layers.Dense(num_units)(outputs)

  if hparams.normalize:
    outputs = layers.Activation('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = layers.Activation('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')
