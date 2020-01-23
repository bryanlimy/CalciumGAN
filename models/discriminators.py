from .utils import get_activation_fn
from .registry import discriminator_register as register

import numpy as np
import tensorflow as tf


@register
def mlp(hparams):
  signals = tf.keras.Input(shape=hparams.signal_shape, name='signals')

  outputs = tf.keras.layers.Flatten()(signals)

  outputs = tf.keras.layers.Dense(512)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(256)(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=signals, outputs=outputs, name='discriminator')


@register
def conv1d(hparams):
  signals = tf.keras.Input(hparams.signal_shape, name='signals')

  outputs = tf.keras.layers.Conv1D(
      filters=512, kernel_size=3, strides=2, padding='causal')(signals)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(0.3)(outputs)

  outputs = tf.keras.layers.Conv1D(
      filters=256, kernel_size=3, strides=2, padding='causal')(outputs)
  outputs = get_activation_fn(hparams.activation)(outputs)
  outputs = tf.keras.layers.Dropout(0.3)(outputs)

  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  return tf.keras.Model(inputs=signals, outputs=outputs, name='discriminator')
