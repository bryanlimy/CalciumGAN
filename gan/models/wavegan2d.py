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

  def __init__(self, input_shape, shuffle=0, mode='reflect'):
    super().__init__()
    self.shape = input_shape
    self.shuffle = shuffle
    self.mode = mode

  def call(self, inputs):
    if self.shuffle == 0:
      return inputs

    w_phase = tf.random.uniform([],
                                minval=-self.shuffle,
                                maxval=self.shuffle + 1,
                                dtype=tf.int32)
    w_left_pad = tf.maximum(w_phase, 0)
    w_right_pad = tf.maximum(-w_phase, 0)

    c_phase = tf.random.uniform([],
                                minval=-self.shuffle,
                                maxval=self.shuffle + 1,
                                dtype=tf.int32)
    c_left_pad = tf.maximum(c_phase, 0)
    c_right_pad = tf.maximum(-c_phase, 0)

    outputs = tf.pad(
        inputs,
        paddings=[[0, 0], [w_left_pad, w_right_pad], [c_left_pad, c_right_pad],
                  [0, 0]],
        mode=self.mode)

    outputs = outputs[:, w_right_pad:w_right_pad +
                      self.shape[1], c_right_pad:c_right_pad + self.shape[2], :]
    return tf.ensure_shape(outputs, shape=self.shape)


def discriminator(hparams,
                  kernel_size=(16, 16),
                  strides=(4, 1),
                  padding='same',
                  shuffle=2):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  # Layer 1
  outputs = layers.Conv2D(
      filters=hparams.num_units,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(inputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, shuffle=shuffle)(outputs)

  # Layer 2
  outputs = layers.Conv2D(
      filters=hparams.num_units * 2,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, shuffle=shuffle)(outputs)

  # Layer 3
  outputs = layers.Conv2D(
      filters=hparams.num_units * 3,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, shuffle=shuffle)(outputs)

  # Layer 4
  outputs = layers.Conv2D(
      filters=hparams.num_units * 4,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = PhaseShuffle(outputs.shape, shuffle=shuffle)(outputs)

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
