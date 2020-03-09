from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn, calculate_input_config, Conv1DTranspose


@register('conv1d')
def get_conv1d(hparams):
  return generator(hparams), discriminator(hparams)


def generator(hparams, kernel_size=4, strides=2, padding='valid'):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  shape, num_units = calculate_input_config(
      output=hparams.sequence_length,
      noise_dim=hparams.noise_dim,
      num_convolution=3,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)
  signal_length = hparams.signal_shape[-1]

  outputs = layers.Dense(num_units, use_bias=False)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  outputs = Conv1DTranspose(signal_length // 4, kernel_size, strides)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = Conv1DTranspose(signal_length // 2, kernel_size, strides)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = Conv1DTranspose(signal_length, kernel_size, strides)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = layers.Dense(hparams.signal_shape[-1])(outputs)

  if hparams.normalize:
    outputs = activation_fn('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def discriminator(hparams, kernel_size=4, strides=2):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  signal_length = hparams.signal_shape[-1]

  outputs = layers.Conv1D(
      filters=signal_length, kernel_size=kernel_size, strides=strides)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Conv1D(
      filters=signal_length // 2, kernel_size=kernel_size,
      strides=strides)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Conv1D(
      filters=signal_length // 4, kernel_size=kernel_size,
      strides=strides)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
