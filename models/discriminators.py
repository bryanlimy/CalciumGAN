from .registry import discriminator_register as register

import numpy as np
import tensorflow as tf


@register
def conv1d(hparams):
  signal = tf.keras.Input(hparams.signal_shape, name='signal')

  outputs = tf.keras.layers.Reshape((signal.shape[-1] // 4, 4))(signal)

  outputs = tf.keras.layers.Conv1D(
      filters=128, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)

  return tf.keras.Model(inputs=signal, outputs=outputs, name='discriminator')
