from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn, calculate_input_config, Conv1DTranspose


@register('conv1d')
def get_conv1d(hparams):
  return generator(hparams), discriminator(hparams)


def generator(hparams, filters=32, kernel_size=4, strides=2, padding='same'):
  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  shape, num_units = calculate_input_config(
      output=hparams.sequence_length,
      noise_dim=hparams.noise_dim,
      num_convolution=3,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)

  outputs = layers.Dense(num_units)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  # Layer 1
  outputs = Conv1DTranspose(
      filters * 2, kernel_size, strides, padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 2
  outputs = Conv1DTranspose(
      filters, kernel_size, strides, padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 3
  outputs = Conv1DTranspose(
      hparams.num_neurons, kernel_size, strides, padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = layers.Dense(hparams.num_neurons)(outputs)

  if hparams.normalize:
    outputs = activation_fn('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def discriminator(hparams, filters=32, kernel_size=4, strides=2,
                  padding='same'):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  # Layer 1
  outputs = layers.Conv1D(
      filters, kernel_size=kernel_size, strides=strides,
      padding=padding)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 2
  outputs = layers.Conv1D(
      filters * 2, kernel_size=kernel_size, strides=strides,
      padding=padding)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 3
  outputs = layers.Conv1D(
      filters * 3, kernel_size=kernel_size, strides=strides,
      padding=padding)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
