import numpy as np
import tensorflow as tf


def get_generator(hparams):
  inputs = tf.keras.Input(shape=hparams.generator_input_shape, name='inputs')
  shape = inputs.shape[1:]

  outputs = tf.keras.layers.Reshape((shape[0] // 4, 4))(inputs)
  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Conv1D(
      filters=128, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(np.prod(hparams.signal_shape))(outputs)
  outputs = tf.keras.layers.Reshape(hparams.signal_shape)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.sigmoid(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def get_discriminator(hparams):
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
