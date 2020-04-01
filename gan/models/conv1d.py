from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn, Conv1DTranspose


@register('conv1d')
def get_conv1d(hparams):
  return generator(hparams), discriminator(hparams)


def calculate_convolution_width(layer, output, kernel_size, strides, padding):
  if padding == 'same':
    w = output / strides
  else:
    w = (1 / strides) * (output - kernel_size) + 1

  if not w.is_integer():
    raise ValueError('Conv1D: step {} is not an integer.'.format(w))

  if layer > 1:
    w = calculate_convolution_width(
        layer=layer - 1,
        output=w,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)
  return int(w)


def calculate_input_config(output,
                           noise_dim,
                           num_convolution,
                           kernel_size,
                           strides,
                           padding='same'):
  w = calculate_convolution_width(
      layer=num_convolution,
      output=output,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)
  return (w, noise_dim)


def generator(hparams, filters=32, kernel_size=4, strides=2, padding='same'):
  shape = calculate_input_config(
      output=hparams.sequence_length,
      noise_dim=hparams.noise_dim,
      num_convolution=3,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding)

  hparams.noise_shape = shape

  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  outputs = layers.Dense(filters)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 1
  outputs = Conv1DTranspose(
      filters, kernel_size, strides, padding=padding)(outputs)
  if not hparams.no_batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 2
  outputs = Conv1DTranspose(
      filters * 2, kernel_size, strides, padding=padding)(outputs)
  if not hparams.no_batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 3
  outputs = Conv1DTranspose(
      hparams.num_channels, kernel_size, strides, padding=padding)(outputs)
  if not hparams.no_batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

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
      filters * 3, kernel_size=kernel_size, strides=strides,
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
      1, kernel_size=kernel_size, strides=strides, padding=padding)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
