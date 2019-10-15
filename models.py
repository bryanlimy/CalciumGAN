import numpy as np
import tensorflow as tf


def get_generator(hparams):
  inputs = tf.keras.Input(
      shape=(hparams.noise_dim, hparams.noise_dim), name='inputs')

  outputs = tf.keras.layers.Conv1D(
      filters=512, kernel_size=10, strides=2, padding='causal')(inputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=10, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(hparams.sequence_length)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def get_discriminator(hparams):
  inputs = tf.keras.Input(shape=(hparams.sequence_length,), name='inputs')

  outputs = tf.keras.layers.Dense(hparams.sequence_length * 5)(inputs)
  outputs = tf.keras.layers.Reshape((hparams.sequence_length, 5))(outputs)
  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=10, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Conv1D(
      filters=512, kernel_size=10, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')

  return model
