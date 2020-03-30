from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import activation_fn


@register('conv2d')
def get_conv2d(hparams):
  return generator(hparams), discriminator(hparams)


def calculate_input_config(output, noise_dim, num_convolution, strides):
  w = output[0] / (strides[0]**num_convolution)
  if not w.is_integer():
    raise ValueError('Conv1D: w {} is not an integer.'.format(w))
  return (int(w), output[1], noise_dim)


def generator(hparams,
              filters=32,
              kernel_size=(4, 2),
              strides=(2, 1),
              padding='same'):
  shape = calculate_input_config(
      output=(hparams.sequence_length, hparams.num_neurons, 1),
      noise_dim=hparams.noise_dim,
      num_convolution=3,
      strides=strides)

  hparams.noise_shape = shape

  inputs = tf.keras.Input(shape=hparams.noise_shape, name='inputs')

  # outputs = layers.Dense(np.prod(shape))(inputs)
  # outputs = activation_fn(hparams.activation)(outputs)
  # outputs = layers.Reshape(shape)(outputs)

  outputs = layers.Dense(filters)(inputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 1
  outputs = layers.Conv2DTranspose(
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  if not hparams.no_batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 2
  outputs = layers.Conv2DTranspose(
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  if not hparams.no_batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  # Layer 3
  outputs = layers.Conv2DTranspose(
      1,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  if not hparams.no_batch_norm:
    outputs = layers.BatchNormalization()(outputs)
  outputs = activation_fn(hparams.activation)(outputs)

  outputs = layers.Dense(1)(outputs)

  if hparams.normalize:
    outputs = activation_fn('sigmoid', dtype=tf.float32)(outputs)
  else:
    outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  outputs = tf.squeeze(outputs, axis=-1)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def discriminator(hparams,
                  filters=32,
                  kernel_size=(4, 2),
                  strides=(2, 1),
                  padding='same'):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  outputs = tf.expand_dims(inputs, axis=-1)

  # Layer 1
  outputs = layers.Conv2D(
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 2
  outputs = layers.Conv2D(
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  # Layer 3
  outputs = layers.Conv2D(
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
  )(outputs)
  outputs = activation_fn(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Flatten()(outputs)
  outputs = layers.Dense(1)(outputs)
  outputs = activation_fn('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')
