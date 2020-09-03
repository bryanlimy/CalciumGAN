from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn, Conv1DTranspose


@register('wavegan2d')
def get_wavegan(hparams):
  return generator(hparams), discriminator(hparams)


def calculate_noise_shape(output_shape, noise_dim, num_convolutions, strides):
  w = output_shape[0] / (strides**num_convolutions)
  if not w.is_integer():
    raise ValueError('Conv2D: w {} is not an integer.'.format(w))
  return (int(w), output_shape[1] // 2, noise_dim)


def generator(hparams, padding='same'):
  kernel_size = (hparams.kernel_size, hparams.kernel_size)

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
  outputs = layers.Conv2DTranspose(
      filters=hparams.num_units * 5,
      kernel_size=kernel_size,
      strides=(hparams.strides, 1),
      padding=padding,
  )(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 2
  outputs = layers.Conv2DTranspose(
      filters=hparams.num_units * 3,
      kernel_size=kernel_size,
      strides=(hparams.strides, 1),
      padding=padding,
  )(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 3
  outputs = layers.Conv2DTranspose(
      filters=hparams.num_units * 2,
      kernel_size=kernel_size,
      strides=(hparams.strides, 2),
      padding=padding,
  )(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 4
  outputs = layers.Conv2DTranspose(
      filters=hparams.num_units,
      kernel_size=kernel_size,
      strides=(hparams.strides, 1),
      padding=padding,
  )(outputs)
  if hparams.batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  if hparams.layer_norm:
    outputs = layers.LayerNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 5
  outputs = layers.Conv2DTranspose(
      filters=hparams.num_channels,
      kernel_size=kernel_size,
      strides=(hparams.strides, 1),
      padding=padding,
  )(outputs)
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

  def __init__(self, input_shape, m=0, n=0, mode='reflect'):
    super().__init__()
    self.shape = input_shape
    self.m = m
    self.n = n
    self.mode = mode

  def call(self, inputs):

    def get_paddings(shift, length):
      if shift > 0:
        paddings = [0, shift]
        start, end = shift, length + shift
      else:
        paddings = [tf.math.abs(shift), 0]
        start, end = 0, length
      return paddings, start, end

    # shift on the temporal and spatial dimensions
    w_shift = tf.random.uniform([],
                                minval=-self.m,
                                maxval=self.m + 1,
                                dtype=tf.int32)
    c_shift = tf.random.uniform([],
                                minval=-self.n,
                                maxval=self.n + 1,
                                dtype=tf.int32)

    w_padding, w_start, w_end = get_paddings(w_shift, self.shape[1])
    c_padding, c_start, c_end = get_paddings(c_shift, self.shape[2])
    paddings = [[0, 0], w_padding, c_padding, [0, 0]]

    outputs = tf.pad(inputs, paddings=paddings, mode=self.mode)

    outputs = outputs[:, w_start:w_end, c_start:c_end, :]
    return tf.ensure_shape(outputs, shape=self.shape)


def discriminator(hparams, kernel_size=(16, 16), strides=(4, 1),
                  padding='same'):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  # Layer 1
  outputs = layers.Conv2D(
      filters=hparams.num_units,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m, n=hparams.n)(outputs)

  # Layer 2
  outputs = layers.Conv2D(
      filters=hparams.num_units * 2,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m, n=hparams.n)(outputs)

  # Layer 3
  outputs = layers.Conv2D(
      filters=hparams.num_units * 3,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m, n=hparams.n)(outputs)

  # Layer 4
  outputs = layers.Conv2D(
      filters=hparams.num_units * 4,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=0, n=hparams.n)(outputs)

  # Layer 5
  outputs = layers.Conv2D(
      filters=hparams.num_units * 5,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
