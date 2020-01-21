from .registry import generator_register as register

import numpy as np
import tensorflow as tf


@register
def mlp(hparams):
  inputs = tf.keras.Input(shape=hparams.generator_input_shape, name='inputs')

  outputs = tf.keras.layers.Flatten()(inputs)

  outputs = tf.keras.layers.Dense(512, activation='tanh')(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(256, activation='tanh')(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(np.prod(hparams.signal_shape))(outputs)
  outputs = tf.keras.layers.Reshape(hparams.signal_shape)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.tanh(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')


@register
def conv1d(hparams):
  inputs = tf.keras.Input(shape=hparams.generator_input_shape, name='inputs')

  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=3, strides=2, padding='valid')(inputs)
  outputs = tf.keras.layers.BatchNormalization()(outputs)
  outputs = tf.keras.activations.tanh(outputs)
  outputs = tf.keras.layers.Conv1D(
      filters=128, kernel_size=3, strides=2, padding='valid')(outputs)
  outputs = tf.keras.layers.BatchNormalization()(outputs)
  outputs = tf.keras.activations.tanh(outputs)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(np.prod(hparams.signal_shape))(outputs)
  outputs = tf.keras.layers.Reshape(hparams.signal_shape)(outputs)

  if hparams.normalize:
    outputs = tf.keras.activations.tanh(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')
