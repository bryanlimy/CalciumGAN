import numpy as np
import tensorflow as tf


def get_generator(hparams):
  inputs = tf.keras.Input(shape=hparams.generator_input_shape, name='inputs')

  outputs = tf.keras.layers.Conv1D(
      filters=512, kernel_size=10, strides=2, padding='causal')(inputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=10, strides=2, padding='causal')(outputs)
  outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(np.prod(
      hparams.generator_output_shape))(outputs)
  outputs = tf.keras.layers.Reshape(hparams.generator_output_shape)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


def get_discriminator(hparams):
  inputs = tf.keras.Input(hparams.generator_output_shape, name='inputs')

  outputs = tf.keras.layers.Reshape((inputs.shape[1] * inputs.shape[2],
                                     inputs.shape[3]))(inputs)
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
