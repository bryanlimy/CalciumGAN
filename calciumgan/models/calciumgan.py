from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn, Conv1DTranspose


@register('calciumgan')
def get_calciumgan(hparams):
  return generator(hparams), discriminator(hparams)


def calculate_noise_shape(output_shape, noise_dim, num_convolutions, strides):
  w = output_shape[0] / (strides**num_convolutions)
  if not w.is_integer():
    raise ValueError('Conv1D: w {} is not an integer.'.format(w))
  return (int(w), noise_dim)


def generator(hparams, padding='same'):
  shape = calculate_noise_shape(
      output_shape=hparams.signal_shape,
      noise_dim=hparams.noise_dim,
      num_convolutions=5,
      strides=hparams.strides)
  noise_shape = int(np.prod(shape))

  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  outputs = layers.Dense(noise_shape)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Reshape(shape)(outputs)

  # Layer 1
  outputs = Conv1DTranspose(
      filters=hparams.num_units * 5,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 2
  outputs = Conv1DTranspose(
      filters=hparams.num_units * 4,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 3
  outputs = Conv1DTranspose(
      filters=hparams.num_units * 3,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 4
  outputs = Conv1DTranspose(
      filters=hparams.num_units * 2,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 5
  outputs = Conv1DTranspose(
      filters=hparams.num_channels,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = layers.Dense(hparams.num_channels)(outputs)

  if hparams.normalize:
    outputs = activation_fn('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


class PhaseShuffle(layers.Layer):
  ''' Phase Shuffle introduced in the WaveGAN paper so that the discriminator 
  are less sensitive toward periodic patterns which occurs quite frequently in
  signal data '''

  def __init__(self, input_shape, m=0, mode='reflect'):
    super().__init__()
    self.shape = input_shape
    self.m = m
    self.mode = mode

  def call(self, inputs):
    w = self.shape[1]

    # shift on the temporal dimension
    shift = tf.random.uniform([],
                              minval=-self.m,
                              maxval=self.m + 1,
                              dtype=tf.int32)

    if shift > 0:
      # shift to the right
      paddings = [[0, 0], [0, shift], [0, 0]]
      start, end = shift, w + shift
    else:
      # shift to the left
      paddings = [[0, 0], [tf.math.abs(shift), 0], [0, 0]]
      start, end = 0, w

    outputs = tf.pad(inputs, paddings=paddings, mode=self.mode)

    outputs = outputs[:, start:end, :]
    return tf.ensure_shape(outputs, shape=self.shape)


def discriminator(hparams, padding='same'):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  # Layer 1
  outputs = layers.Conv1D(
      filters=hparams.num_units,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)

  # Layer 2
  outputs = layers.Conv1D(
      filters=hparams.num_units * 2,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)

  # Layer 3
  outputs = layers.Conv1D(
      filters=hparams.num_units * 3,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)

  # Layer 4
  outputs = layers.Conv1D(
      filters=hparams.num_units * 4,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)

  # Layer 5
  outputs = layers.Conv1D(
      filters=hparams.num_units * 5,
      kernel_size=hparams.kernel_size,
      strides=hparams.strides,
      padding=padding)(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
