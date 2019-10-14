import numpy as np
import tensorflow as tf


def get_generator(hparams):
  inputs = tf.keras.Input(shape=(hparams.noise_dim,), name='inputs')

  outputs = tf.keras.layers.Dense(hparams.num_units, activation='relu')(inputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(hparams.num_units, activation='relu')(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(hparams.sequence_length)(outputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')

  return model


def get_discriminator(hparams):
  inputs = tf.keras.Input(shape=(hparams.sequence_length,), name='inputs')

  outputs = tf.keras.layers.Dense(hparams.num_units, activation='relu')(inputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(hparams.num_units, activation='relu')(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.Dense(1)(outputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='discriminator')

  return model
